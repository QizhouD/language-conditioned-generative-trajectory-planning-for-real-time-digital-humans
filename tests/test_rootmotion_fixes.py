import json
import math
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from gtp.rootmotion import HISTORY_STEPS, HORIZON_STEPS, MAX_PATH_VERTICES
from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.service import PlanService, previous_plan_tensor, serve
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.train import augment_history
from gtp.rootmotion.windows import HISTORY_DIM, history_features, path_tokens

STATS = RootMotionStats(0.095, 0.105, (0.9, 2.0, 2.5), ((0.3, 2.6), (0.9, 4.4), (0.3, 5.2)))


def model(speed=True, deterministic=False, history_steps=HISTORY_STEPS):
    torch.manual_seed(0)
    return RootMotionModel(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2, deterministic=deterministic,
                           speed_conditioning=speed, history_steps=history_steps).eval()


def batch(speed=(0.9,), rows=1):
    tokens, valid = path_tokens(np.array([[0.0, 6.0]]), True, MAX_PATH_VERTICES)
    local = np.stack([np.zeros(HISTORY_STEPS), (np.arange(HISTORY_STEPS) - HISTORY_STEPS + 1) * 0.05], axis=1)
    history = torch.from_numpy(history_features(local, np.zeros(HISTORY_STEPS)))[None].repeat(rows, 1, 1)
    return {"history": history, "path": torch.from_numpy(tokens)[None].repeat(rows, 1, 1),
            "path_valid": torch.from_numpy(valid)[None].repeat(rows, 1), "gait": torch.zeros(rows, dtype=torch.long),
            "target_speed": torch.tensor(speed, dtype=torch.float32)}


# --------------------------------------------------------------------------- A: guidance


def test_guidance_one_is_exactly_the_unguided_sample():
    net, data = model(), batch()
    plain = net.sample(data, steps=2, samples=3, generator=torch.Generator().manual_seed(1))
    one = net.sample(data, steps=2, samples=3, generator=torch.Generator().manual_seed(1), guidance=1.0)
    assert torch.equal(plain, one)


def test_guidance_extrapolates_from_unspecified_towards_the_given_speed():
    for deterministic in (False, True):
        net, data = model(deterministic=deterministic), batch()
        gen = lambda: torch.Generator().manual_seed(1)
        cond = net.sample(data, steps=1, samples=2, generator=gen())
        uncond = net.sample({**data, "target_speed": torch.tensor([float("nan")])}, steps=1, samples=2, generator=gen())
        guided = net.sample(data, steps=1, samples=2, generator=gen(), guidance=2.0)
        # One Euler step from the same noise: guided = uncond + 2 (cond - uncond).
        assert torch.allclose(guided, uncond + 2.0 * (cond - uncond), atol=1e-5)


def test_guidance_leaves_rows_without_a_speed_and_unconditioned_models_alone():
    net = model()
    nan = batch(speed=(float("nan"),))
    gen = lambda: torch.Generator().manual_seed(1)
    assert torch.allclose(net.sample(nan, steps=2, generator=gen(), guidance=3.0),
                          net.sample(nan, steps=2, generator=gen()), atol=1e-6)
    plain = model(speed=False)
    assert torch.equal(plain.sample(batch(), steps=2, generator=gen(), guidance=3.0),
                       plain.sample(batch(), steps=2, generator=gen()))


def test_planner_passes_its_guidance_weight():
    planner = RootMotionPlanner(model(), STATS, PlannerConfig(candidates=2, flow_steps=1, speed_guidance=2.0))
    assert planner.plan_batch(batch())["deltas"].shape == (1, HORIZON_STEPS, 3)


# --------------------------------------------------------------------------- B: history noise


def test_history_noise_keeps_the_current_pose_and_recomputes_velocities():
    data = batch()
    noisy = augment_history(data, 0.05, 0.1, torch.Generator().manual_seed(0))["history"]
    assert torch.equal(noisy[:, -1, 0:4], data["history"][:, -1, 0:4])  # current pose untouched
    assert not torch.allclose(noisy[:, :-1, 0:2], data["history"][:, :-1, 0:2])
    positions = noisy[0, :, 0:2] * 5.0
    velocity = (positions[1:] - positions[:-1]) * 10.0
    assert torch.allclose(noisy[0, 1:, 4:6] * 5.0, velocity, atol=1e-4)
    assert torch.allclose(noisy[0, :, 2] ** 2 + noisy[0, :, 3] ** 2, torch.ones(HISTORY_STEPS), atol=1e-5)
    unchanged = augment_history(data, 0.0, 0.0)["history"]
    assert torch.allclose(unchanged, data["history"], atol=1e-6)


# --------------------------------------------------------------------------- C: shorter history


def test_a_short_history_model_reads_only_the_last_steps():
    short = model(history_steps=5, deterministic=True)
    data = batch()
    reference = short.predict(data)
    older = {**data, "history": data["history"].clone()}
    older["history"][:, :-5] = 7.0  # change what lies beyond the last 5 steps
    assert torch.equal(short.predict(older), reference)
    recent = {**data, "history": data["history"].clone()}
    recent["history"][:, -2, 0] += 0.5
    assert not torch.allclose(short.predict(recent), reference)
    # The default 20-step model still reads everything.
    full = model(deterministic=True)
    assert not torch.allclose(full.predict(older), full.predict(data))


# --------------------------------------------------------------------------- service previous_plan


def request(**extra):
    body = {"history": [{"position": [2.0, 0.0, 3.0], "yaw": math.pi / 2}],
            "path": [[2.0, 0.0, 3.0], [12.0, 0.0, 3.0]], "gait": "walk", "target_speed": 0.9}
    body.update(extra)
    return body


def test_previous_plan_is_converted_into_the_local_frame():
    local = previous_plan_tensor([[2.1, 0.0, 3.0], [2.2, 0.0, 3.0]], {"origin": (2.0, 0.0, 3.0), "yaw": math.pi / 2})
    assert local.shape == (1, 2, 2) and local[0, 0].tolist() == pytest.approx([0.0, 0.1], abs=1e-6)
    assert previous_plan_tensor(None, {}) is None and previous_plan_tensor([], {}) is None


def test_service_accepts_an_optional_previous_plan():
    service = PlanService(RootMotionPlanner(model(), STATS, PlannerConfig(candidates=2, flow_steps=1)))
    without = service.plan(request())
    assert without["previous_plan_used"] is False
    previous = [[2.0 + 0.09 * k, 0.0, 3.0] for k in range(1, 21)]
    used = service.plan(request(previous_plan=previous))
    assert used["previous_plan_used"] is True and len(used["trajectory"]) == HORIZON_STEPS + 1


def test_service_rejects_a_malformed_previous_plan():
    planner = RootMotionPlanner(model(), STATS, PlannerConfig(candidates=2, flow_steps=1))
    server = serve(planner, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for bad in ([[1.0, 2.0]], [[1.0, 0.0, float("nan")]], "points", [[0.0, 0.0, 0.0]] * 65):
            body = json.dumps(request(previous_plan=bad)).replace("NaN", "null").encode()
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/plan", data=body,
                                         method="POST", headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req, timeout=30)
            assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()

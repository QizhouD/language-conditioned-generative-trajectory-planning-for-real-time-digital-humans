import math

import numpy as np
import pytest
import torch

from gtp.rootmotion import GAITS, HISTORY_STEPS, HORIZON_STEPS, MAX_PATH_VERTICES
from gtp.rootmotion.evaluate import run_baseline
from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner, load_checkpoint
from gtp.rootmotion.request import RequestError, build_batch
from gtp.rootmotion.service import PlanService
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.train import drop_target_speed
from gtp.rootmotion.windows import (
    HISTORY_DIM, WindowConfig, WindowSet, path_tokens, target_speed_label, windows_from_track,
)

RANGED = RootMotionStats(0.095, 0.105, (0.9, 2.0, 2.5), ((0.3, 2.6), (0.9, 4.4), (0.3, 5.2)))


def tiny(speed_conditioning=False, deterministic=True):
    torch.manual_seed(0)
    return RootMotionModel(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2,
                           deterministic=deterministic, speed_conditioning=speed_conditioning).eval()


def batch(rows=2, speed=None):
    tokens, valid = path_tokens(np.array([[0.0, 4.0]]), True, MAX_PATH_VERTICES)
    history = torch.zeros(rows, HISTORY_STEPS, HISTORY_DIM)
    history[..., 3] = 1.0
    out = {
        "history": history,
        "path": torch.from_numpy(tokens)[None].repeat(rows, 1, 1),
        "path_valid": torch.from_numpy(valid)[None].repeat(rows, 1),
        "gait": torch.zeros(rows, dtype=torch.long),
    }
    if speed is not None:
        out["target_speed"] = torch.as_tensor(speed, dtype=torch.float32)
    return out


def request(**extra):
    body = {"history": [{"position": [0.0, 0.0, 0.0], "yaw": 0.0}],
            "path": [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], "path_ends_at_goal": True, "gait": "walk"}
    body.update(extra)
    return body


# --------------------------------------------------------------------------- label


def test_label_is_the_90th_percentile_of_moving_speeds_and_zero_when_still():
    speeds = np.array([0.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    moving = speeds[speeds > 0.2]
    assert target_speed_label(speeds, 0.2, 90.0) == pytest.approx(np.percentile(moving, 90))
    assert target_speed_label(np.array([0.0, 0.05, 0.1]), 0.2, 90.0) == 0.0


def test_windows_carry_the_label_of_the_capture_their_path_describes():
    # Walk +z at 1.2 m/s for 30 s, then stand for 10 s.
    walk = np.stack([np.zeros(300), np.arange(300) * 0.12, np.zeros(300)], axis=1)
    stand = np.repeat(walk[-1:], 100, axis=0)
    track = np.concatenate([walk, stand]).astype(np.float32)
    out = WindowSet()
    windows_from_track(track, GAITS.index("walk"), WindowConfig(stride=20), np.random.default_rng(0), out, 0.25)
    assert out.target_speed and len(out.target_speed) == len(out.target)
    moving = [s for s in out.target_speed if s > 0]
    assert moving and all(s == pytest.approx(1.2, abs=1e-3) for s in moving)


# --------------------------------------------------------------------------- model


def test_without_the_switch_the_model_is_unchanged_and_ignores_target_speed():
    plain, switched_off = tiny(), tiny(speed_conditioning=False)
    assert set(plain.state_dict()) == set(switched_off.state_dict())
    assert not any(key.startswith("speed_") for key in plain.state_dict())
    reference = plain.predict(batch())
    assert torch.equal(plain.predict(batch(speed=[0.6, float("nan")])), reference)


def test_old_checkpoints_load_and_reproduce(tmp_path):
    model = tiny()
    config = {"model": {"hidden_dim": 32, "encoder_layers": 1, "decoder_layers": 1, "heads": 2,
                        "deterministic": True}}  # written before speed_conditioning existed
    stats = {"delta_std": 0.095, "turn_std": 0.105, "gait_speed": [0.9, 2.0, 2.5]}  # no speed_range
    torch.save({"model": model.state_dict(), "config": config, "stats": stats}, tmp_path / "old.pt")
    loaded, loaded_stats, _ = load_checkpoint(tmp_path / "old.pt")
    assert not loaded.speed_conditioning and loaded_stats.speed_range is None
    assert torch.equal(loaded.predict(batch()), model.predict(batch()))


def test_unspecified_speed_is_a_learned_input_and_differs_from_a_given_speed():
    model = tiny(speed_conditioning=True)
    missing = model.predict(batch())
    nan = model.predict(batch(speed=[float("nan"), float("nan")]))
    assert torch.equal(missing, nan)
    given = model.predict(batch(speed=[0.6, 1.3]))
    assert not torch.allclose(given[0], missing[0])
    assert not torch.allclose(given[0], given[1])
    # Mixed rows: each row gets its own input.
    mixed = model.predict(batch(speed=[0.6, float("nan")]))
    assert torch.allclose(mixed[0], given[0]) and torch.allclose(mixed[1], missing[1])
    mixed.sum().backward()
    assert model.speed_unspecified.grad is not None and model.speed_unspecified.grad.abs().sum() > 0


def test_training_drops_the_speed_with_the_given_probability():
    data = {"target_speed": torch.linspace(0.5, 1.5, 1000)}
    assert torch.equal(drop_target_speed(data, 0.0)["target_speed"], data["target_speed"])
    assert torch.isnan(drop_target_speed(data, 1.0)["target_speed"]).all()
    dropped = torch.isnan(drop_target_speed(data, 0.1, torch.Generator().manual_seed(0))["target_speed"])
    assert 0.05 < float(dropped.float().mean()) < 0.15


# --------------------------------------------------------------------------- stats


def test_stats_record_the_label_range_and_stay_compatible():
    target = torch.zeros(200, HORIZON_STEPS, 3)
    target[..., 1] = 0.1
    labels = torch.cat([torch.linspace(0.4, 2.0, 100), torch.linspace(1.0, 4.0, 100)])
    gaits = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)])
    stats = RootMotionStats.from_batch({"target": target, "gait": gaits, "target_speed": labels})
    assert stats.speed_range[0][0] == pytest.approx(float(torch.quantile(labels[:100], 0.01)))
    assert stats.speed_range[1][1] == pytest.approx(float(torch.quantile(labels[100:], 0.99)))
    assert RootMotionStats.from_dict(stats.to_dict()) == stats
    old = RootMotionStats.from_dict({"delta_std": 0.1, "turn_std": 0.1, "gait_speed": [1, 2, 3]})
    assert old.speed_range is None and old.clip_speed(9.0, 0) == 9.0
    assert RANGED.clip_speed(9.0, 0) == 2.6 and RANGED.clip_speed(0.1, 1) == 0.9


# --------------------------------------------------------------------------- planner, request, service


def test_planner_clamps_given_speeds_and_leaves_unspecified_ones():
    planner = RootMotionPlanner(tiny(True, deterministic=False), RANGED, PlannerConfig(candidates=2, flow_steps=1))
    result = planner.plan_batch(batch(speed=[9.0, float("nan")]))
    assert result["target_speed"][0] == pytest.approx(2.6)
    assert math.isnan(float(result["target_speed"][1]))
    assert result["target_speed_clipped"].tolist() == [True, False]


def test_an_unconditioned_planner_ignores_target_speed():
    planner = RootMotionPlanner(tiny(False, deterministic=False), RANGED, PlannerConfig(candidates=2, flow_steps=1))
    generator = lambda: torch.Generator().manual_seed(0)
    without = planner.plan_batch(batch(), generator())
    with_speed = planner.plan_batch(batch(speed=[2.5, 2.5]), generator())
    assert torch.equal(without["deltas"], with_speed["deltas"])  # same spring fallback speed too
    assert "target_speed" not in with_speed


def test_request_target_speed_is_optional_and_validated():
    data, frame = build_batch(request())
    assert math.isnan(float(data["target_speed"][0])) and frame["target_speed"] is None
    data, frame = build_batch(request(target_speed=1.1))
    assert float(data["target_speed"][0]) == pytest.approx(1.1) and frame["target_speed"] == 1.1
    for bad in (-0.5, "fast", float("inf")):
        with pytest.raises(RequestError):
            build_batch(request(target_speed=bad))


def test_service_reports_what_happened_to_the_target_speed():
    conditioned = PlanService(RootMotionPlanner(tiny(True, deterministic=False), RANGED,
                                                PlannerConfig(candidates=2, flow_steps=1)))
    assert conditioned.plan(request())["target_speed"]["requested"] is None  # unspecified works
    used = conditioned.plan(request(target_speed=1.1))["target_speed"]
    assert used["used"] == pytest.approx(1.1) and used["clipped"] is False
    clamped = conditioned.plan(request(target_speed=7.0))["target_speed"]
    assert clamped["used"] == pytest.approx(2.6) and clamped["clipped"] is True and "note" in clamped
    plain = PlanService(RootMotionPlanner(tiny(False, deterministic=False), RANGED,
                                          PlannerConfig(candidates=2, flow_steps=1)))
    ignored = plain.plan(request(target_speed=1.1))["target_speed"]
    assert ignored["used"] is None and "ignored" in ignored["note"]


def test_baselines_can_be_told_each_windows_target_speed():
    data = batch(speed=[0.5, 1.5])
    data["target"] = torch.zeros(2, HORIZON_STEPS, 3)
    median = run_baseline("constant", data, (0.9, 2.0, 2.5))
    told = run_baseline("constant", data, (0.9, 2.0, 2.5), per_window_speed=True)
    speed = lambda d: torch.linalg.vector_norm(d[..., :2], dim=-1).amax(dim=-1) * 10
    assert speed(median).tolist() == pytest.approx([0.9, 0.9], abs=1e-4)
    assert speed(told).tolist() == pytest.approx([0.5, 1.5], abs=1e-4)

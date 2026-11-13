import json
import math
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from gtp.rootmotion import GAITS, HISTORY_STEPS, HORIZON_STEPS, MAX_PATH_VERTICES
from gtp.rootmotion.baselines import constant_speed_follow, spring_follow
from gtp.rootmotion.constraints import (
    KinematicLimits, corridor_deviation, corridor_violation, kinematic_violation,
)
from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.planner import (
    SOURCE_HOLD, SOURCE_MODEL, SOURCE_SPRING, PlannerConfig, RootMotionPlanner,
)
from gtp.rootmotion.request import RequestError, build_batch, build_plan, clip_path
from gtp.rootmotion.service import serve
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.windows import HISTORY_DIM, PATH_DIM, path_tokens, trajectory_positions

STATS = RootMotionStats(0.095, 0.105, (0.9, 2.0, 2.5))


def tiny_model(deterministic=False):
    torch.manual_seed(0)
    return RootMotionModel(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2,
                           deterministic=deterministic).eval()


def make_batch(vertices, ends_at_goal=True, gait=0, velocity=(0.0, 0.0)):
    tokens, valid = path_tokens(np.asarray(vertices, dtype=np.float64), ends_at_goal, MAX_PATH_VERTICES)
    history = torch.zeros(1, HISTORY_STEPS, HISTORY_DIM)
    history[..., 3] = 1.0
    history[..., 4] = velocity[0] / 5.0
    history[..., 5] = velocity[1] / 5.0
    return {
        "history": history, "path": torch.from_numpy(tokens)[None],
        "path_valid": torch.from_numpy(valid)[None], "gait": torch.tensor([gait]),
    }


def straight(speed, steps=HORIZON_STEPS, lateral=0.0):
    deltas = torch.zeros(steps, 3)
    deltas[:, 1] = speed / 10.0
    deltas[0, 0] = lateral
    return deltas


def test_model_shapes_and_sampling_are_seeded():
    model = tiny_model()
    batch = make_batch([[0.0, 4.0]])
    batch = {key: value.repeat(3, *[1] * (value.dim() - 1)) for key, value in batch.items()}
    state, time = torch.randn(3, HORIZON_STEPS, 3), torch.rand(3)
    assert model.velocity(state, time, batch).shape == (3, HORIZON_STEPS, 3)
    first = model.sample(batch, steps=2, samples=4, generator=torch.Generator().manual_seed(1))
    again = model.sample(batch, steps=2, samples=4, generator=torch.Generator().manual_seed(1))
    assert first.shape == (3, 4, HORIZON_STEPS, 3)
    assert torch.equal(first, again)
    assert tiny_model(deterministic=True).sample(batch, samples=4).shape == (3, 1, HORIZON_STEPS, 3)


def test_padded_path_vertices_do_not_change_the_output():
    model = tiny_model(deterministic=True)
    batch = make_batch([[0.0, 2.0], [1.0, 3.0]])
    reference = model.predict(batch)
    noisy = {**batch, "path": batch["path"].clone()}
    noisy["path"][0, 2:] = 123.0
    assert torch.allclose(model.predict(noisy), reference, atol=1e-5)


def test_kinematic_limits_cover_speed_acceleration_turning_and_nan():
    limits = KinematicLimits()
    assert not bool(kinematic_violation(straight(1.4), limits, torch.tensor([0.0, 1.4])))
    assert bool(kinematic_violation(straight(9.0), limits, torch.tensor([0.0, 9.0])))
    # Fine in isolation, but an impossible jump from the motion already under way.
    assert not bool(kinematic_violation(straight(3.0), limits))
    assert bool(kinematic_violation(straight(3.0), limits, torch.tensor([0.0, 0.0])))
    spin = straight(1.0)
    spin[5, 2] = 1.5
    assert bool(kinematic_violation(spin, limits))
    broken = straight(1.0)
    broken[3, 0] = float("nan")
    assert bool(kinematic_violation(broken, limits))


def test_corridor_check_is_swept_and_starts_at_the_origin():
    valid = torch.ones(3, dtype=torch.bool)
    corner = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # Both end points sit exactly on the L-shaped path; the segment between them cuts the corner.
    shortcut = torch.tensor([[0.0, 0.2, 0.0], [0.8, 0.8, 0.0]])
    positions = trajectory_positions(shortcut)
    assert positions[-1].tolist() == pytest.approx([0.8, 1.0])
    assert float(corridor_deviation(shortcut, corner, valid)) > 0.25
    assert bool(corridor_violation(shortcut, corner, valid, half_width=0.25))
    # The bound is conservative: it never under-reports the true swept deviation.
    dense = positions[0] + torch.linspace(0, 1, 2001)[:, None] * (positions[1] - positions[0])
    dense = torch.cat([dense, positions[1] + torch.linspace(0, 1, 2001)[:, None] * (positions[2] - positions[1])])
    from gtp.rootmotion.paths import distance_to_polyline
    true_max = float(distance_to_polyline(dense, corner, valid).max())
    assert float(corridor_deviation(shortcut, corner, valid)) >= true_max

    line = torch.tensor([[0.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    assert not bool(corridor_violation(straight(1.4), line, valid, half_width=0.5))
    assert bool(corridor_violation(straight(1.4, lateral=0.7), line, valid, half_width=0.5))


def test_spring_follower_rounds_a_corner_and_settles_on_the_goal():
    polyline = np.array([[0.0, 0.0], [0.0, 2.0], [1.5, 2.0]])
    deltas = spring_follow(polyline, True, np.zeros(2), 1.2, steps=80)
    positions = np.cumsum(deltas[:, :2], axis=0)
    assert np.hypot(*(positions[-1] - polyline[-1])) < 0.1
    assert np.hypot(*deltas[-1, :2]) * 10 < 0.1
    speed = np.hypot(deltas[:, 0], deltas[:, 1]) * 10
    assert speed[0] < 0.6 and speed.max() <= 1.2 + 1e-6  # eases in, never exceeds the target
    assert float(corridor_deviation(
        torch.tensor(deltas), torch.tensor(polyline), torch.ones(3, dtype=torch.bool)
    )) < 0.5
    assert abs(deltas[:, 2].sum() - math.pi / 2) < 0.2  # ends up facing along the second leg


def test_spring_follower_keeps_moving_when_the_path_continues():
    deltas = spring_follow(np.array([[0.0, 0.0], [0.0, 1.0]]), False, np.array([0.0, 1.2]), 1.2)
    assert np.cumsum(deltas[:, 1])[-1] > 3.0


def test_constant_speed_follower_respects_its_acceleration_limit():
    deltas = constant_speed_follow(np.array([[0.0, 0.0], [0.0, 10.0]]), True, 0.0, 3.5)
    speed = np.hypot(deltas[:, 0], deltas[:, 1]) * 10
    assert np.all(np.diff(np.concatenate([[0.0], speed])) * 10 <= 8.0 + 1e-6)
    assert speed.max() == pytest.approx(3.5)
    short = constant_speed_follow(np.array([[0.0, 0.0], [0.0, 1.0]]), True, 0.0, 3.5)
    assert np.cumsum(short[:, 1])[-1] == pytest.approx(1.0, abs=1e-6)


def test_planner_falls_back_to_the_spring_then_to_holding():
    # An untrained network produces noise, so every candidate must be rejected.
    planner = RootMotionPlanner(tiny_model(), STATS, PlannerConfig(candidates=4, flow_steps=2))
    result = planner.plan_batch(make_batch([[0.0, 4.0]]))
    assert bool(result["candidate_invalid"].all())
    assert int(result["source"][0]) == SOURCE_SPRING
    assert not bool(planner.invalid(result["deltas"][None], make_batch([[0.0, 4.0]]))[0, 0])
    # A corridor too tight for the spring's corner rounding leaves only standing still.
    hairpin = make_batch([[0.0, 1.0], [0.3, 0.0]], velocity=(0.0, 2.0), gait=1)
    held = planner.plan_batch(hairpin, half_width=0.06)
    assert int(held["source"][0]) == SOURCE_HOLD
    assert float(held["deltas"].abs().max()) == 0.0


def test_planner_uses_the_first_valid_candidate():
    class Scripted(RootMotionModel):
        def sample(self, batch, steps=8, samples=1, generator=None):
            good = STATS.normalise(straight(1.2))
            bad = STATS.normalise(straight(1.2, lateral=2.0))
            return torch.stack([bad, good, good * 0.5])[None]

    torch.manual_seed(0)
    model = Scripted(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2)
    planner = RootMotionPlanner(model, STATS, PlannerConfig(candidates=3, selection="first_valid"))
    result = planner.plan_batch(make_batch([[0.0, 6.0]], velocity=(0.0, 1.2)))
    assert result["candidate_invalid"][0].tolist() == [True, False, False]
    assert int(result["source"][0]) == SOURCE_MODEL
    assert torch.allclose(result["deltas"][0], straight(1.2), atol=1e-6)


def walk_then_stop(moving_steps, speed=1.0, lateral=0.0):
    deltas = torch.zeros(HORIZON_STEPS, 3)
    deltas[:moving_steps, 1] = speed / 10.0
    deltas[0, 0] = lateral
    return deltas


class ScriptedCandidates(RootMotionModel):
    """Returns fixed candidates: [overshoots off the path, stops at 1.5 m, stops at 2.0 m]."""

    def sample(self, batch, steps=8, samples=1, generator=None):
        candidates = [walk_then_stop(30, lateral=2.0), walk_then_stop(15), walk_then_stop(20)]
        return torch.stack([STATS.normalise(c) for c in candidates])[None]


def scripted_planner(selection):
    torch.manual_seed(0)
    model = ScriptedCandidates(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2)
    return RootMotionPlanner(model, STATS, PlannerConfig(candidates=3, selection=selection))


def test_goal_aware_picks_the_candidate_that_stops_on_a_reachable_goal():
    near = make_batch([[0.0, 2.0]])  # goal 2 m ahead: within walk speed 0.9 m/s x 3 s
    first = scripted_planner("first_valid").plan_batch(near)
    aware = scripted_planner("goal_aware").plan_batch(near)
    assert first["candidate_invalid"][0].tolist() == [True, False, False]
    assert torch.allclose(first["deltas"][0], walk_then_stop(15), atol=1e-6)  # first valid
    assert torch.allclose(aware["deltas"][0], walk_then_stop(20), atol=1e-6)  # ends on the goal
    assert int(aware["source"][0]) == SOURCE_MODEL


def test_goal_aware_falls_back_to_first_valid_when_the_goal_is_far_or_open():
    far = make_batch([[0.0, 10.0]])  # 10 m > 0.9 m/s x 3 s
    assert torch.allclose(scripted_planner("goal_aware").plan_batch(far)["deltas"][0],
                          walk_then_stop(15), atol=1e-6)
    open_path = make_batch([[0.0, 2.0]], ends_at_goal=False)
    assert torch.allclose(scripted_planner("goal_aware").plan_batch(open_path)["deltas"][0],
                          walk_then_stop(15), atol=1e-6)


def test_goal_aware_prefers_the_slower_finish_at_equal_distance():
    from gtp.rootmotion.planner import GOAL_AWARE_SPEED_WEIGHT

    class Finishes(RootMotionModel):
        def sample(self, batch, steps=8, samples=1, generator=None):
            # Both end 2.0 m ahead; the first is still moving at 0.67 m/s, the second has stopped.
            moving = torch.zeros(HORIZON_STEPS, 3)
            moving[:, 1] = 2.0 / HORIZON_STEPS
            return torch.stack([STATS.normalise(moving), STATS.normalise(walk_then_stop(20))])[None]

    torch.manual_seed(0)
    model = Finishes(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2)
    planner = RootMotionPlanner(model, STATS, PlannerConfig(candidates=2, selection="goal_aware"))
    result = planner.plan_batch(make_batch([[0.0, 2.0]]))
    assert GOAL_AWARE_SPEED_WEIGHT > 0
    assert torch.allclose(result["deltas"][0], walk_then_stop(20), atol=1e-6)


def test_goal_aware_is_the_default_selection():
    assert PlannerConfig().selection == "goal_aware"
    near = make_batch([[0.0, 2.0]])
    default = RootMotionPlanner(scripted_planner("goal_aware").model, STATS, PlannerConfig(candidates=3))
    assert torch.allclose(default.plan_batch(near)["deltas"][0], walk_then_stop(20), atol=1e-6)


def test_unknown_selection_rule_is_rejected():
    with pytest.raises(ValueError):
        RootMotionPlanner(tiny_model(), STATS, PlannerConfig(selection="nearest"))


def request_body(**overrides):
    body = {
        "history": [{"position": [10.0, 0.5, 20.0], "yaw": math.pi / 2}],
        "path": [[10.0, 0.5, 20.0], [14.0, 0.5, 20.0]],
        "path_ends_at_goal": True,
        "gait": "walk",
    }
    body.update(overrides)
    return body


def test_request_is_expressed_in_the_character_frame():
    batch, frame = build_batch(request_body())
    assert batch["history"].shape == (1, HISTORY_STEPS, HISTORY_DIM)
    assert batch["path"].shape == (1, MAX_PATH_VERTICES, PATH_DIM)
    # Facing +x (yaw 90 degrees), a point 4 m along +x is 4 m straight ahead; the
    # corner on top of the character is dropped.
    assert batch["path_valid"][0].tolist() == [True] + [False] * (MAX_PATH_VERTICES - 1)
    assert (batch["path"][0, 0] * 5.0).tolist() == pytest.approx([0.0, 4.0, 4.0, 5.0], abs=1e-5)
    assert frame["origin"] == (10.0, 0.5, 20.0) and frame["gait"] == "walk"
    with pytest.raises(RequestError):
        build_batch(request_body(gait="moonwalk"))
    with pytest.raises(RequestError):
        build_batch(request_body(history=[]))
    with pytest.raises(RequestError):
        build_batch(request_body(path=[[1.0, 2.0]]))


def test_long_paths_are_clipped_and_lose_their_goal_flag():
    vertices, ends = clip_path(np.array([[0.0, 10.0], [0.0, 40.0]]), True)
    assert ends is False
    assert vertices[-1].tolist() == pytest.approx([0.0, 25.0])
    many = np.stack([np.zeros(30), np.arange(1, 31) * 0.5], axis=1)
    vertices, ends = clip_path(many, True)
    assert len(vertices) == MAX_PATH_VERTICES and ends is False


def test_plan_round_trips_to_world_space_and_labels_segments():
    deltas = straight(1.2).tolist()
    for step in deltas[20:]:
        step[1] = 0.0
    frame = {"origin": (10.0, 0.5, 20.0), "yaw": math.pi / 2, "gait": "run"}
    plan = build_plan(deltas, frame).to_dict()
    assert len(plan["trajectory"]) == HORIZON_STEPS + 1
    assert plan["horizon"] == pytest.approx(3.0)
    first, last = plan["trajectory"][0], plan["trajectory"][-1]
    assert first["t"] == 0.0 and first["position"] == pytest.approx([10.0, 0.5, 20.0])
    # 20 steps of 0.12 m straight ahead, and straight ahead is world +x.
    assert last["position"] == pytest.approx([12.4, 0.5, 20.0], abs=1e-6)
    assert first["velocity"] == pytest.approx([1.2, 0.0, 0.0], abs=1e-6)
    assert first["yaw"] == pytest.approx(math.pi / 2)
    assert [segment["action"] for segment in plan["segments"]] == ["walk", "stop"]
    assert "runs" in plan["segments"][0]["prompt"]
    assert plan["segments"][0]["end"] == pytest.approx(plan["segments"][1]["start"])
    assert plan["fallback"] is False


def test_service_answers_plan_requests_over_http():
    planner = RootMotionPlanner(tiny_model(), STATS, PlannerConfig(candidates=2, flow_steps=1))
    server = serve(planner, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as response:
            assert json.loads(response.read())["status"] == "ok"

        def post(body):
            request = urllib.request.Request(
                f"{base}/plan", data=json.dumps(body).encode(), method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())

        plan = post(request_body(seed=3))
        assert plan["source"] == "spring_fallback" and plan["fallback"] is False
        assert plan["rejected_candidates"] == plan["candidates"] == 2
        assert len(plan["trajectory"]) == HORIZON_STEPS + 1
        assert all(len(point["position"]) == 3 and len(point["velocity"]) == 3 for point in plan["trajectory"])
        assert plan["trajectory"][-1]["position"][0] > 10.5  # heading for the goal along +x
        assert set(GAITS) >= {segment["action"] for segment in plan["segments"]} - {"turn", "stop", "idle"}

        assert post(request_body(candidates=5))["candidates"] == 5
        with pytest.raises(urllib.error.HTTPError) as error:
            post(request_body(candidates=500))
        assert error.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as error:
            post(request_body(gait="moonwalk"))
        assert error.value.code == 400
        assert "gait" in json.loads(error.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()

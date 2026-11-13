import math

import numpy as np
import pytest
import torch

from gtp.rootmotion import HORIZON_STEPS
from gtp.rootmotion.closed_loop import (
    Route, aggregate, baseline_plan_function, model_plan_function, open_loop, remaining_path,
    rollout_metrics, simulate, standard_routes, unity_route,
)
from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.paths import arclength
from gtp.rootmotion.planner import SOURCE_HOLD, SOURCE_MODEL, PlannerConfig, RootMotionPlanner
from gtp.rootmotion.request import build_batch
from gtp.rootmotion.stats import RootMotionStats

STATS = RootMotionStats(0.095, 0.105, (1.0, 2.0, 2.5))
STRAIGHT = Route("straight", ((0.0, 0.0), (0.0, 3.0)), 0.0, "walk")


class Forward:
    """Deterministic planner: 0.1 m per step straight ahead, stopping at the path's end."""

    def __init__(self, source=SOURCE_MODEL):
        self.calls = []
        self.source = source

    def __call__(self, request):
        batch, _ = build_batch(request)
        self.calls.append(request)
        count = int(batch["path_valid"][0].sum())
        remaining = float(batch["path"][0, count - 1, 2]) * 5.0 if count else 0.0
        deltas = np.zeros((HORIZON_STEPS, 3))
        travelled = 0.0
        for step in range(HORIZON_STEPS):
            move = min(0.1, max(0.0, remaining - travelled))
            deltas[step, 1] = move
            travelled += move
        return deltas, self.source, {}


def test_rollout_replans_on_schedule_and_reaches_the_goal():
    planner = Forward()
    rollout = simulate(STRAIGHT, planner, replan_interval=1.0, time_limit=6.0, settle_early_stop=False)
    assert len(planner.calls) == 6  # one request per second
    assert int(rollout.seams.sum()) == 5  # every adoption after the first is a seam
    # Facing +z with the goal 3 m ahead: 1 m/s in a straight line, then rest.
    assert rollout.poses[10, :2] == pytest.approx([0.0, 1.0], abs=1e-6)
    metrics = rollout_metrics(rollout)
    assert metrics["arrival_time"] == pytest.approx(2.8)  # within 0.25 m of the goal
    assert metrics["settled"] and metrics["stop_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["final_distance"] == pytest.approx(0.0, abs=1e-6) and metrics["final_speed"] == 0.0
    assert metrics["moving_mean_speed"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["backward_steps"] == 0 and metrics["max_deviation"] == pytest.approx(0.0, abs=1e-9)
    # Requests carry 20 poses of history, the current position first on the path, and the goal.
    last = planner.calls[1]
    assert len(last["history"]) == 20 and last["path_ends_at_goal"] is True
    assert last["path"][0] == pytest.approx([0.0, 0.0, 1.0]) and last["path"][-1] == pytest.approx([0.0, 0.0, 3.0])


def test_latency_keeps_playing_the_old_plan_and_starts_the_new_one_late():
    planner = Forward()
    rollout = simulate(STRAIGHT, planner, replan_interval=1.0, latency_steps=2, time_limit=2.0,
                       settle_early_stop=False)
    # No plan for the first two steps: the character waits.
    assert rollout.poses[2, :2] == pytest.approx([0.0, 0.0])
    # Adopted at step 2 and started at its index 2, so the plan's timing is kept.
    assert rollout.poses[3, :2] == pytest.approx([0.0, 0.3], abs=1e-6)


def test_lock_on_arrival_stops_at_the_tolerance_like_the_engine():
    planner = Forward()
    rollout = simulate(STRAIGHT, planner, replan_interval=1.0, time_limit=6.0, lock_on_arrival=True)
    # 0.1 m per step: the first pose within 0.25 m of the goal is 2.8 m along; it stays there.
    assert rollout.locked_at == 28
    assert np.allclose(rollout.poses[28:, :2], [0.0, 2.8])
    assert len(planner.calls) == 3  # no replanning after the lock
    metrics = rollout_metrics(rollout)
    assert metrics["locked"] and metrics["settled"]
    assert metrics["final_distance"] == pytest.approx(0.2, abs=1e-6) and metrics["final_speed"] == 0.0
    unlocked = rollout_metrics(simulate(STRAIGHT, Forward(), replan_interval=1.0, time_limit=6.0))
    assert not unlocked["locked"] and unlocked["final_distance"] == pytest.approx(0.0, abs=1e-6)


def test_request_extras_reach_the_planner():
    planner = Forward()
    simulate(STRAIGHT, planner, replan_interval=1.0, time_limit=1.0, settle_early_stop=False,
             request_extras={"target_speed": 0.6})
    assert planner.calls[0]["target_speed"] == 0.6


def test_a_hold_answer_stands_still_and_counts_as_hold():
    rollout = simulate(STRAIGHT, Forward(SOURCE_HOLD), replan_interval=1.0, time_limit=3.0,
                       settle_early_stop=False)
    assert np.allclose(rollout.poses[:, :2], 0.0)
    assert rollout_metrics(rollout)["hold_rate"] == 1.0


def test_remaining_path_only_moves_forward_along_a_fold_back():
    route = np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    cumulative = arclength(route)
    # Near the start on the way back: the apex is behind, only the goal remains.
    path = remaining_path(route, cumulative, np.array([0.0, 0.5]), progress=7.5)
    assert path.tolist() == [[0.0, 0.5], [0.0, 0.0]]
    path = remaining_path(route, cumulative, np.array([0.0, 0.5]), progress=0.5)
    assert path.tolist() == [[0.0, 0.5], [0.0, 4.0], [0.0, 0.0]]


def test_spring_baseline_closes_the_loop_through_the_service_path():
    route = Route("corner", ((0.0, 0.0), (0.0, 3.0), (3.0, 3.0)), 0.0, "walk")
    rollout = simulate(route, baseline_plan_function("spring", STATS.gait_speed), gait_speed=1.0)
    metrics = rollout_metrics(rollout)
    assert metrics["arrived_in_time"] and metrics["settled"]
    assert metrics["stop_error"] < 0.15
    assert metrics["max_deviation"] < 0.5
    assert metrics["backward_steps"] == 0


def test_a_fold_back_is_not_arrived_at_its_start():
    route = Route("uturn", ((0.0, 0.0), (0.0, 4.0), (0.0, 0.0)), 0.0, "walk")
    metrics = rollout_metrics(simulate(route, baseline_plan_function("spring", STATS.gait_speed), gait_speed=1.0))
    # The goal is the start: arrival only counts once the route has been walked.
    assert metrics["arrival_time"] is not None and metrics["arrival_time"] > 6.0
    assert metrics["follow_mean_speed"] > 0.5


def test_model_plan_function_reports_selection_details():
    torch.manual_seed(0)
    model = RootMotionModel(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2).eval()
    planner = RootMotionPlanner(model, STATS, PlannerConfig(candidates=3, flow_steps=1))
    plan = model_plan_function(planner, torch.Generator().manual_seed(0))
    rollout = simulate(STRAIGHT, plan, replan_interval=1.0, time_limit=2.0, settle_early_stop=False)
    assert len(rollout.details) == 2
    assert {"candidate_speed", "valid_candidate_speed", "rejected"} <= set(rollout.details[0])
    # An untrained network is rejected every time and the spring takes over.
    assert rollout_metrics(rollout)["fallback_rate"] == 1.0
    assert open_loop(STRAIGHT, plan)["mean_speed_3s"] > 0.0


def test_route_grid_and_aggregate():
    routes = standard_routes()
    assert len(routes) == 6 * 2 * 2
    turned = next(r for r in routes if r.name == "corner_4_4/turned90/walk")
    assert turned.start_yaw == pytest.approx(math.pi / 2)
    assert unity_route().length == pytest.approx(2.0 * math.hypot(2.96, 1.28) / 2 + 2.08 + math.hypot(5.46, 1.28), rel=1e-3)
    rows = [{"routes": [{"route": "a", "stop_error": 1.0, "settled": True, "arrived_in_time": True}], "w1": {}},
            {"routes": [{"route": "a", "stop_error": 3.0, "settled": False, "arrived_in_time": True}], "w1": {}}]
    summary = aggregate(rows)["routes"]["a"]
    assert summary["stop_error"]["mean"] == 2.0 and summary["stop_error"]["std"] == pytest.approx(math.sqrt(2))
    assert summary["settled"] == 0.5 and summary["arrived_in_time"] == 1.0

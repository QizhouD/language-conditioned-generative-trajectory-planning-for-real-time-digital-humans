import math

import numpy as np
import pytest
import torch

from gtp.rootmotion import HISTORY_STEPS, HORIZON_STEPS, MAX_PATH_VERTICES
from gtp.rootmotion.closed_loop import (
    Route, capture_segments, model_plan_function, phase_breakdown, simulate, speed_by_progress,
)
from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.request import build_batch
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.windows import HISTORY_DIM, path_tokens, trajectory_positions

STATS = RootMotionStats(0.095, 0.105, (0.9, 2.0, 2.5))


def walk(speed, steps=HORIZON_STEPS, lateral=0.0):
    deltas = torch.zeros(steps, 3)
    deltas[:, 1] = speed / 10.0
    deltas[:, 0] = lateral / 10.0
    return deltas


class Scripted(RootMotionModel):
    """Three valid candidates on a long path: 1.2 m/s straight, 0.8 m/s straight, 0.8 m/s drifting."""

    def sample(self, batch, steps=8, samples=1, generator=None):
        candidates = [walk(1.2), walk(0.8), walk(0.8, lateral=0.1)]
        return torch.stack([STATS.normalise(c) for c in candidates])[None]


def planner(selection="consistent"):
    torch.manual_seed(0)
    model = Scripted(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2)
    return RootMotionPlanner(model, STATS, PlannerConfig(candidates=3, selection=selection))


def batch(vertices=((0.0, 10.0),), velocity=(0.0, 1.0)):
    tokens, valid = path_tokens(np.asarray(vertices, dtype=np.float64), True, MAX_PATH_VERTICES)
    history = torch.zeros(1, HISTORY_STEPS, HISTORY_DIM)
    history[..., 3] = 1.0
    history[..., 4], history[..., 5] = velocity[0] / 5.0, velocity[1] / 5.0
    return {"history": history, "path": torch.from_numpy(tokens)[None],
            "path_valid": torch.from_numpy(valid)[None], "gait": torch.tensor([0])}


def previous_of(deltas, steps=20):
    return trajectory_positions(deltas)[1:steps + 1][None]  # [1, M, 2], local, steps 1..M


def test_closed_loop_passes_the_previous_plans_remaining_positions():
    calls = []

    def plan(request, previous=None):
        calls.append(previous)
        deltas = np.zeros((HORIZON_STEPS, 3))
        deltas[:, 1] = 0.05
        return deltas, 0, {}

    route = Route("straight", ((0.0, 0.0), (0.0, 20.0)), 0.0, "walk")
    simulate(route, plan, replan_interval=1.0, time_limit=2.0, settle_early_stop=False)
    assert calls[0] is None
    # After 10 executed steps (0.5 m) the previous plan still had steps 11..30 to visit.
    assert calls[1].shape == (20, 2)
    assert calls[1][0] == pytest.approx([0.0, 0.55]) and calls[1][-1] == pytest.approx([0.0, 1.5])


def test_model_plan_function_hands_the_previous_plan_to_the_planner_in_the_local_frame():
    seen = []

    class Recording(RootMotionPlanner):
        def plan_batch(self, batch, generator=None, half_width=None, candidates=None, previous=None):
            seen.append(previous)
            return super().plan_batch(batch, generator, half_width, candidates, previous)

    torch.manual_seed(0)
    model = Scripted(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2)
    recording = Recording(model, STATS, PlannerConfig(candidates=3, selection="consistent"))
    plan = model_plan_function(recording, None)
    request = {"history": [{"position": [2.0, 0.0, 3.0], "yaw": math.pi / 2}],
               "path": [[2.0, 0.0, 3.0], [12.0, 0.0, 3.0]], "gait": "walk"}
    plan(request, np.array([[2.1, 3.0], [2.2, 3.0]]))
    # Facing +x from (2, 3): world (2.1, 3) is 0.1 m straight ahead.
    assert seen[-1].shape == (1, 2, 2)
    assert seen[-1][0, 0].tolist() == pytest.approx([0.0, 0.1], abs=1e-6)


def test_initial_history_starts_from_a_moving_state():
    history = [(0.0, -1.9 + 0.1 * k, 0.0) for k in range(HISTORY_STEPS)]  # walking +z at 1 m/s
    seen = []

    def plan(request, previous=None):
        seen.append(request)
        return np.zeros((HORIZON_STEPS, 3)), 0, {}

    route = Route("straight", ((0.0, 0.0), (0.0, 5.0)), 0.0, "walk")
    rollout = simulate(route, plan, time_limit=0.5, settle_early_stop=False, initial_history=history)
    assert rollout.poses[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    batch_, _ = build_batch(seen[0])
    assert float(batch_["history"][0, -1, 5]) * 5.0 == pytest.approx(1.0, abs=1e-5)  # vz in the history


def test_phase_breakdown_splits_start_cruise_and_deceleration():
    speed = np.array([0.1, 0.3, 0.6, 0.85, 0.9, 0.95, 0.88, 0.9, 0.5, 0.2])
    phases = phase_breakdown(speed, asked=0.9)
    assert phases["start_s"] == pytest.approx(0.3) and phases["cruise_s"] == pytest.approx(0.5)
    assert phases["decel_s"] == pytest.approx(0.2)
    assert phases["cruise_speed"] == pytest.approx(np.mean([0.85, 0.9, 0.95, 0.88, 0.9]))
    assert phase_breakdown(np.full(10, 0.5), asked=0.9)["reached"] is False


def test_speed_by_progress_bins_speeds_along_the_route():
    route = np.array([[0.0, 0.0], [0.0, 4.0]])
    positions = np.stack([np.zeros(41), np.concatenate([np.arange(21) * 0.05, 1.0 + np.arange(1, 21) * 0.15])], axis=1)
    profile = speed_by_progress(positions, route, bins=4)
    # Bins hold the steps that end inside them: the first metre at 0.5 m/s, the last two at 1.5.
    assert profile[0] == pytest.approx(0.5) and profile[2] == pytest.approx(1.5) and profile[3] == pytest.approx(1.5)


def test_capture_segments_cut_rest_to_rest_walks_with_their_history_and_label():
    rest = np.zeros((40, 3))
    walk_out = np.stack([np.zeros(50), 0.1 * np.arange(1, 51), np.zeros(50)], axis=1)  # 1 m/s for 5 s
    rest_after = np.repeat(walk_out[-1:], 30, axis=0)
    track = np.concatenate([rest, walk_out, rest_after]).astype(np.float32)
    segments = capture_segments(track, "walk1_subject5")
    assert len(segments) == 1
    segment = segments[0]
    assert segment.start == 39 and segment.end == 89
    assert segment.route.points[0].tolist() == [0.0, 0.0] and segment.route.points[-1].tolist() == pytest.approx([0.0, 5.0])
    assert len(segment.history) == HISTORY_STEPS and segment.history[-1] == (0.0, 0.0, 0.0)
    assert segment.target_speed == pytest.approx(1.0, abs=1e-5)
    assert capture_segments(track, "short", min_length=6.0) == []

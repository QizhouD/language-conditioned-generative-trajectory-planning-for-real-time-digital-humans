import math

import numpy as np
import pytest
import torch

from gtp.rootmotion import HISTORY_STEPS, HORIZON_STEPS, MAX_PATH_VERTICES
from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
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


def test_consistent_follows_the_unexecuted_rest_of_the_previous_plan():
    plain = planner("goal_aware").plan_batch(batch())
    assert torch.allclose(plain["deltas"][0], walk(1.2), atol=1e-6)  # first valid
    chosen = planner().plan_batch(batch(), previous=previous_of(walk(0.8)))
    assert torch.allclose(chosen["deltas"][0], walk(0.8), atol=1e-6)
    drifting = planner().plan_batch(batch(), previous=previous_of(walk(0.8, lateral=0.1)))
    assert torch.allclose(drifting["deltas"][0], walk(0.8, lateral=0.1), atol=1e-6)


def test_consistent_without_a_previous_plan_or_near_the_goal_is_goal_aware():
    for result in (planner().plan_batch(batch()),
                   planner().plan_batch(batch(), previous=torch.full((1, 20, 2), float("nan")))):
        assert torch.allclose(result["deltas"][0], walk(1.2), atol=1e-6)
    near = batch(vertices=((0.0, 2.4),))  # 2.4 m <= 0.9 m/s x 3 s: goal-aware's rule decides
    aware = planner("goal_aware").plan_batch(near)
    consistent = planner().plan_batch(near, previous=previous_of(walk(1.2)))
    assert torch.equal(aware["deltas"], consistent["deltas"])


def test_consistency_cost_compares_overlapping_steps_and_first_velocity():
    candidates = torch.stack([walk(1.0), walk(0.5)])[None]
    cost = RootMotionPlanner.consistency_cost(candidates, previous_of(walk(1.0), steps=5))
    assert float(cost[0, 0]) == pytest.approx(0.0, abs=1e-6)
    # 0.5 m/s slower: positions fall behind by 0.05 m per step (mean over 5 steps 0.15 m),
    # and the first-step velocity differs by 0.5 m/s, weighted 0.1 s.
    assert float(cost[0, 1]) == pytest.approx(0.15 + 0.1 * 0.5, abs=1e-5)
    assert math.isnan(float(RootMotionPlanner.consistency_cost(candidates, torch.full((1, 5, 2), float("nan")))[0, 0]))

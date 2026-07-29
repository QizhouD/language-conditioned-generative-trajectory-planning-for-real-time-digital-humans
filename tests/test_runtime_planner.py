import math
from types import MethodType

import pytest
import torch

from gtp.models import ConditionalTrajectoryFlow
from gtp.runtime import DUMMY_OBSTACLE_POSITION, RuntimePlanner, build_runtime_batch
from gtp.schema import MotionState, PlanningGoal


def _history():
    return [
        MotionState(
            position=[0.0, 0.0, 0.0],
            yaw=0.0,
            velocity=[0.0, 0.0],
            angular_velocity=0.0,
            action=0,
            action_phase=0.0,
            foot_contacts=[1.0, 1.0],
        )
    ]


def _safe_sample(self, batch, steps, samples):
    count = batch["history"].shape[0] * samples
    trajectory = torch.zeros(count, 3, 8)
    trajectory[..., 1] = 0.1
    trajectory[..., 3] = 1.0
    trajectory[..., 5] = 1.0
    trajectory[..., 7] = 1.0
    logits = torch.zeros(count, 3, 5)
    logits[..., 1] = 10.0
    return {"trajectory": trajectory, "action_logits": logits}


def test_runtime_batch_uses_distant_dummy_obstacle():
    batch = build_runtime_batch(
        _history(),
        PlanningGoal("walk", [0.0, 0.0, 1.0], 0.0),
        [],
    )
    assert batch["history"].shape == (1, 1, 12)
    assert batch["goal"].shape == (1, 5)
    assert batch["obstacle_valid"].tolist() == [[True]]
    assert batch["obstacles"][0, 0, 0].item() == DUMMY_OBSTACLE_POSITION


def test_runtime_planner_connects_function_call_to_world_plan():
    model = ConditionalTrajectoryFlow(
        state_dim=12,
        goal_dim=5,
        obstacle_dim=4,
        trajectory_dim=8,
        hidden_dim=32,
        layers=1,
        heads=4,
        horizon_steps=3,
    )
    model.sample = MethodType(_safe_sample, model)
    planner = RuntimePlanner(model, candidates=2, flow_steps=1)
    plan = planner.plan_function_call(
        {
            "function": "move_to",
            "arguments": {"target_id": "visitor", "stop_distance": 0.5},
        },
        {"visitor": {"position": [11.0, 0.0, 5.0], "yaw": math.pi / 2}},
        _history(),
        [],
        character_position=[10.0, 0.0, 5.0],
        character_yaw=math.pi / 2,
    )
    assert not plan.fallback
    assert plan.trajectory[-1]["position"] == pytest.approx([10.3, 0.0, 5.0])
    assert [segment.action for segment in plan.segments] == ["walk"]

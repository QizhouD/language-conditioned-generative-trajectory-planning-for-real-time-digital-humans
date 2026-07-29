"""In-process runtime facade joining domain contracts, inference, and plan export."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from gtp.integrations.plan_export import build_trajectory_plan
from gtp.integrations.simpletool_adapter import planning_goal_from_function_call
from gtp.models import ConditionalTrajectoryFlow
from gtp.planning import RollingTrajectoryPlanner
from gtp.schema import MOTION_INTENTS, MotionState, Obstacle, PlanningGoal, TrajectoryPlan
from gtp.train import build_model

DUMMY_OBSTACLE_POSITION = 1_000_000.0


def build_runtime_batch(
    history: Sequence[MotionState],
    goal: PlanningGoal,
    obstacles: Sequence[Obstacle],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build the exact model input contract without fabricating training targets."""
    if not history:
        raise ValueError("runtime history must be non-empty")
    history_tensor = torch.tensor(
        [state.features() for state in history], dtype=torch.float32, device=device
    ).unsqueeze(0)
    goal_tensor = torch.tensor(
        [
            goal.target_position[0],
            goal.target_position[2],
            math.sin(goal.target_yaw),
            math.cos(goal.target_yaw),
            goal.stop_distance,
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    if obstacles:
        obstacle_tensor = torch.tensor(
            [[*item.center, item.radius, float(item.kind)] for item in obstacles],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
    else:
        obstacle_tensor = torch.tensor(
            [[[DUMMY_OBSTACLE_POSITION, DUMMY_OBSTACLE_POSITION, 0.0, 0.0]]],
            dtype=torch.float32,
            device=device,
        )
    return {
        "history": history_tensor,
        "history_valid": torch.ones(
            1, len(history), dtype=torch.bool, device=device
        ),
        "goal": goal_tensor,
        "behavior": torch.tensor(
            [MOTION_INTENTS.index(goal.behavior)], dtype=torch.long, device=device
        ),
        "obstacles": obstacle_tensor,
        "obstacle_valid": torch.ones(
            1, obstacle_tensor.shape[1], dtype=torch.bool, device=device
        ),
    }


class RuntimePlanner:
    """One-character rolling planner suitable for an engine or service adapter."""

    def __init__(
        self,
        model: ConditionalTrajectoryFlow,
        *,
        candidates: int = 16,
        flow_steps: int = 4,
        execute_steps: int = 3,
        control_hz: int = 10,
        device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(model, ConditionalTrajectoryFlow):
            raise TypeError("RuntimePlanner currently requires ConditionalTrajectoryFlow")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.control_hz = control_hz
        self.planner = RollingTrajectoryPlanner(
            self.model,
            candidates=candidates,
            flow_steps=flow_steps,
            execute_steps=execute_steps,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        candidates: int = 16,
        flow_steps: int = 4,
        execute_steps: int = 3,
        control_hz: int = 10,
        device: torch.device | str = "cpu",
    ) -> "RuntimePlanner":
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
        return cls(
            model,
            candidates=candidates,
            flow_steps=flow_steps,
            execute_steps=execute_steps,
            control_hz=control_hz,
            device=device,
        )

    def plan(
        self,
        history: Sequence[MotionState],
        goal: PlanningGoal,
        obstacles: Sequence[Obstacle],
        *,
        world_origin: Sequence[float],
        world_yaw: float,
    ) -> TrajectoryPlan:
        batch = build_runtime_batch(
            history, goal, obstacles, device=self.device
        )
        result = self.planner.plan(batch)
        return build_trajectory_plan(
            result["trajectory"][0].cpu().tolist(),
            result["actions"][0].cpu().tolist(),
            origin=world_origin,
            origin_yaw=world_yaw,
            control_hz=self.control_hz,
            score=result["cost"][0].cpu().item(),
            fallback=result["fallback"][0].cpu().item(),
        )

    def plan_function_call(
        self,
        call: Mapping[str, Any],
        scene_objects: Mapping[str, Mapping[str, Any]],
        history: Sequence[MotionState],
        obstacles: Sequence[Obstacle],
        *,
        character_position: list[float],
        character_yaw: float,
    ) -> TrajectoryPlan:
        goal = planning_goal_from_function_call(
            call,
            scene_objects,
            character_position=character_position,
            character_yaw=character_yaw,
        )
        return self.plan(
            history,
            goal,
            obstacles,
            world_origin=character_position,
            world_yaw=character_yaw,
        )

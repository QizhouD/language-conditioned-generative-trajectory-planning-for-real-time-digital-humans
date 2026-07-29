"""Remote SimpleTool decision followed by local GTP trajectory planning."""

from __future__ import annotations

from typing import Mapping, Sequence

from gtp.integrations.simpletool_client import SimpleToolClient
from gtp.runtime import RuntimePlanner
from gtp.schema import MotionState, Obstacle, TrajectoryPlan


class RemoteRuntimePipeline:
    def __init__(
        self, simpletool: SimpleToolClient, planner: RuntimePlanner
    ) -> None:
        self.simpletool = simpletool
        self.planner = planner

    def plan_command(
        self,
        command: str,
        *,
        scene_objects: Mapping[str, Mapping],
        motion_history: Sequence[MotionState],
        obstacles: Sequence[Obstacle],
        character_position: list[float],
        character_yaw: float,
        decision_history: Sequence[str] = (),
    ) -> TrajectoryPlan:
        environment = [
            f"{object_id}: position={item.get('position')}, yaw={item.get('yaw', 0.0)}"
            for object_id, item in scene_objects.items()
        ]
        call = self.simpletool.function_call(
            command,
            history=decision_history,
            environment=environment,
        )
        return self.planner.plan_function_call(
            call,
            scene_objects,
            motion_history,
            obstacles,
            character_position=character_position,
            character_yaw=character_yaw,
        )

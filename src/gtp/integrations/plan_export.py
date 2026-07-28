"""Convert model-space trajectories into a versioned runtime plan."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Sequence

from gtp.integrations.prompt_compiler import compile_segments
from gtp.schema import TrajectoryPlan


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def build_trajectory_plan(
    trajectory: Sequence[Sequence[float]],
    actions: Sequence[int],
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    origin_yaw: float = 0.0,
    control_hz: int = 10,
    score: float = 0.0,
    fallback: bool = False,
    plan_id: str | None = None,
) -> TrajectoryPlan:
    if len(origin) != 3:
        raise ValueError("origin must contain xyz")
    if len(trajectory) != len(actions):
        raise ValueError("trajectory and actions must have equal length")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")

    cosine, sine = math.cos(origin_yaw), math.sin(origin_yaw)
    local_x = local_z = 0.0
    points: list[dict] = []
    for index, features in enumerate(trajectory):
        if len(features) != 8:
            raise ValueError("each trajectory point must have eight features")
        local_x += float(features[0])
        local_z += float(features[1])
        world_x = origin[0] + cosine * local_x + sine * local_z
        world_z = origin[2] - sine * local_x + cosine * local_z
        local_yaw = math.atan2(float(features[2]), float(features[3]))
        velocity_x = cosine * float(features[4]) + sine * float(features[5])
        velocity_z = -sine * float(features[4]) + cosine * float(features[5])
        points.append(
            {
                "t": (index + 1) / control_hz,
                "position": [world_x, float(origin[1]), world_z],
                "yaw": _normalize_angle(origin_yaw + local_yaw),
                "velocity": [velocity_x, 0.0, velocity_z],
                "angular_velocity": float(features[6]),
                "speed": float(features[7]),
            }
        )
    return TrajectoryPlan(
        plan_id=plan_id or f"plan-{uuid.uuid4().hex[:12]}",
        horizon=len(points) / control_hz,
        trajectory=points,
        segments=compile_segments(list(actions), control_hz),
        score=float(score),
        fallback=bool(fallback),
    )


def write_plan(path: str | Path, plan: TrajectoryPlan) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

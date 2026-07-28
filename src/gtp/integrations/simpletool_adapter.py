"""Normalize SimpleTool function calls into the stable GTP planning contract."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gtp.schema import PlanningGoal

SUPPORTED_FUNCTIONS = {
    "move_to": "walk",
    "approach": "walk",
    "greet": "wave",
    "greet_person": "wave",
    "look_at": "turn",
    "idle": "idle",
}


def _world_to_local(
    point: list[float], origin: list[float], origin_yaw: float
) -> list[float]:
    dx, dy, dz = point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]
    cosine, sine = math.cos(origin_yaw), math.sin(origin_yaw)
    return [cosine * dx - sine * dz, dy, sine * dx + cosine * dz]


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def planning_goal_from_function_call(
    call: Mapping[str, Any],
    scene_objects: Mapping[str, Mapping[str, Any]],
    character_position: list[float],
    character_yaw: float,
) -> PlanningGoal:
    """Resolve a SimpleTool call and convert world geometry to character-local geometry."""
    function = call.get("function", call.get("name"))
    if function not in SUPPORTED_FUNCTIONS:
        raise ValueError(f"unsupported SimpleTool function: {function}")
    arguments = dict(call.get("arguments") or {})
    target_id = arguments.get("target_id", arguments.get("target"))

    if function == "idle":
        world_position = character_position
        world_yaw = character_yaw
    else:
        if not isinstance(target_id, str) or target_id not in scene_objects:
            raise ValueError(f"unknown or missing target object: {target_id}")
        target = scene_objects[target_id]
        world_position = target.get("position")
        if not isinstance(world_position, list) or len(world_position) != 3:
            raise ValueError(f"target {target_id} has no xyz position")
        world_yaw = float(target.get("yaw", character_yaw))

    local_position = _world_to_local(world_position, character_position, character_yaw)
    local_yaw = _normalize_angle(world_yaw - character_yaw)
    stop_distance = float(
        arguments.get(
            "stop_distance",
            arguments.get("distance", arguments.get("preferred_distance", 0.0)),
        )
    )
    behavior = SUPPORTED_FUNCTIONS[function]
    instruction = {
        "walk": f"Walk toward {target_id} and stop at the requested distance.",
        "wave": f"Approach {target_id}, face them, and wave.",
        "turn": f"Turn naturally to look at {target_id}.",
        "idle": "Stand naturally and remain attentive.",
    }[behavior]
    return PlanningGoal(
        behavior=behavior,
        target_position=local_position,
        target_yaw=local_yaw,
        stop_distance=stop_distance,
        instruction=instruction,
        arguments={"source_function": function, **arguments},
    )

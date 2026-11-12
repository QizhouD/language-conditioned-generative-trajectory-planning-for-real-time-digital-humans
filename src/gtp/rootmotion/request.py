"""Convert a world-space plan request into model inputs and back into a plan."""

from __future__ import annotations

import math
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gtp.rootmotion import CONTROL_HZ, GAITS, HISTORY_STEPS, MAX_PATH_VERTICES
from gtp.rootmotion.paths import arclength
from gtp.rootmotion.windows import history_features, path_tokens, to_local, wrap
from gtp.schema import MotionSegment, TrajectoryPlan

MAX_PATH_LENGTH = 25.0
WALK_SPEED_THRESHOLD = 0.15  # m/s, shared with the Unity mock source
TURN_RATE_THRESHOLD = 0.35  # rad/s
GAIT_VERBS = {"walk": "walks", "run": "runs", "sprint": "sprints"}


class RequestError(ValueError):
    """The request is malformed; maps to HTTP 400."""


def _ground(point: Sequence[float], name: str) -> tuple[float, float, float]:
    if not isinstance(point, (list, tuple)) or len(point) != 3:
        raise RequestError(f"{name} must be [x, y, z]")
    values = [float(v) for v in point]
    if not all(math.isfinite(v) for v in values):
        raise RequestError(f"{name} contains a non-finite value")
    return values[0], values[1], values[2]


def clip_path(vertices: np.ndarray, ends_at_goal: bool,
              max_vertices: int = MAX_PATH_VERTICES,
              max_length: float = MAX_PATH_LENGTH) -> tuple[np.ndarray, bool]:
    """Trim a local polyline (origin excluded) to what the model was trained on."""
    if len(vertices) == 0:
        return vertices, True
    with_origin = np.concatenate([np.zeros((1, 2)), vertices])
    along = arclength(with_origin)
    if along[-1] > max_length:
        index = int(np.searchsorted(along, max_length))
        span = along[index] - along[index - 1]
        ratio = (max_length - along[index - 1]) / max(span, 1e-9)
        cut = with_origin[index - 1] + ratio * (with_origin[index] - with_origin[index - 1])
        vertices = np.concatenate([with_origin[1:index], cut[None]])
        ends_at_goal = False
    if len(vertices) > max_vertices:
        vertices, ends_at_goal = vertices[:max_vertices], False
    return vertices, ends_at_goal


def build_batch(request: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Validate a /plan request and express it in the character's local frame."""
    history = request.get("history")
    if not isinstance(history, list) or not history:
        raise RequestError("history must be a non-empty list of {position, yaw}")
    gait = str(request.get("gait", "walk")).lower()
    if gait not in GAITS:
        raise RequestError(f"gait must be one of {list(GAITS)}")
    poses = []
    for index, state in enumerate(history[-HISTORY_STEPS:]):
        if not isinstance(state, Mapping) or "yaw" not in state:
            raise RequestError(f"history[{index}] must have position and yaw")
        x, y, z = _ground(state.get("position"), f"history[{index}].position")
        poses.append((x, y, z, float(state["yaw"])))
    # Too little history means the character has only just appeared: treat it as
    # having stood at its oldest known pose.
    poses = [poses[0]] * (HISTORY_STEPS - len(poses)) + poses
    world = np.asarray(poses, dtype=np.float64)
    origin, height, yaw = world[-1, [0, 2]], float(world[-1, 1]), float(world[-1, 3])

    corners = request.get("path", [])
    if not isinstance(corners, list):
        raise RequestError("path must be a list of [x, y, z] corners")
    points = [_ground(corner, f"path[{index}]") for index, corner in enumerate(corners)]
    local = to_local(np.asarray([[p[0], p[2]] for p in points]).reshape(-1, 2), origin, yaw)
    # NavMesh paths start at the agent; a vertex on top of the origin carries no information.
    while len(local) and float(np.hypot(*local[0])) < 0.05:
        local = local[1:]
    vertices, ends_at_goal = clip_path(local, bool(request.get("path_ends_at_goal", True)))
    tokens, valid = path_tokens(vertices, ends_at_goal, MAX_PATH_VERTICES)

    past = to_local(world[:, [0, 2]], origin, yaw)
    batch = {
        "history": torch.from_numpy(history_features(past, wrap(world[:, 3] - yaw)))[None],
        "path": torch.from_numpy(tokens)[None],
        "path_valid": torch.from_numpy(valid)[None],
        "gait": torch.tensor([GAITS.index(gait)], dtype=torch.long),
    }
    frame = {"origin": (float(origin[0]), height, float(origin[1])), "yaw": yaw, "gait": gait}
    return batch, frame


def _label(speed: float, turn_rate: float, moved_before: bool) -> str:
    if speed > WALK_SPEED_THRESHOLD:
        return "walk"
    if abs(turn_rate) > TURN_RATE_THRESHOLD:
        return "turn"
    return "stop" if moved_before else "idle"


def _prompt(action: str, gait: str) -> str:
    return {
        "walk": f"A person {GAIT_VERBS[gait]} along the path.",
        "turn": "A person turns in place.",
        "stop": "A person slows down and comes to a settled stop.",
        "idle": "A person stands naturally and remains attentive.",
    }[action]


def build_plan(
    deltas: Sequence[Sequence[float]], frame: Mapping[str, Any], *, fallback: bool = False,
    score: float = 0.0, control_hz: int = CONTROL_HZ, plan_id: str | None = None,
) -> TrajectoryPlan:
    """World-space plan whose first point is the current pose at t = 0."""
    ox, oy, oz = frame["origin"]
    yaw0, gait = float(frame["yaw"]), str(frame["gait"])
    cosine, sine = math.cos(yaw0), math.sin(yaw0)
    local_x = local_z = local_yaw = 0.0
    points: list[dict[str, Any]] = []
    labels: list[str] = []
    moved = False
    steps = [(0.0, 0.0, 0.0)] + [tuple(float(v) for v in step) for step in deltas]
    for index, (dx, dz, turn) in enumerate(steps):
        local_x, local_z, local_yaw = local_x + dx, local_z + dz, local_yaw + turn
        # The pose at t = 0 inherits the first step's velocity so playback starts smoothly.
        vx, vz, rate = steps[index] if index else steps[min(1, len(steps) - 1)]
        velocity = (vx * control_hz, vz * control_hz)
        speed = math.hypot(*velocity)
        turn_rate = rate * control_hz
        points.append(
            {
                "t": index / control_hz,
                "position": [ox + cosine * local_x + sine * local_z, oy,
                             oz - sine * local_x + cosine * local_z],
                "yaw": math.atan2(math.sin(yaw0 + local_yaw), math.cos(yaw0 + local_yaw)),
                "velocity": [cosine * velocity[0] + sine * velocity[1], 0.0,
                             -sine * velocity[0] + cosine * velocity[1]],
                "angular_velocity": turn_rate,
                "speed": speed,
            }
        )
        labels.append(_label(speed, turn_rate, moved))
        moved = moved or speed > WALK_SPEED_THRESHOLD
    segments: list[MotionSegment] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            segments.append(
                MotionSegment(
                    start / control_hz, index / control_hz, labels[start],
                    _prompt(labels[start], gait),
                )
            )
            start = index
    return TrajectoryPlan(
        plan_id=plan_id or f"plan-{uuid.uuid4().hex[:12]}",
        horizon=(len(points) - 1) / control_hz,
        trajectory=points,
        segments=segments,
        score=float(score),
        fallback=bool(fallback),
    )

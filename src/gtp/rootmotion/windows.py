"""Slice root tracks into (history, path, future) training windows.

The conditioning path is what a NavMesh query would hand the planner: a coarse
polyline with no timing. It is recovered from the capture itself by simplifying
the trace the actor went on to walk, so the model has to supply everything a
polyline lacks: the speed profile, how corners are rounded, where the body faces
and how it comes to rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from gtp.rootmotion import (
    CONTROL_HZ, GAITS, HISTORY_STEPS, HORIZON_STEPS, MAX_PATH_VERTICES,
)
from gtp.rootmotion.extract import parse_clip_name
from gtp.rootmotion.paths import arclength, rdp

HISTORY_DIM = 6  # x, z, sin(yaw), cos(yaw), vx, vz
PATH_DIM = 4  # x, z, arclength, is_goal
TARGET_DIM = 3  # dx, dz, dyaw per step
POSITION_SCALE = 5.0  # metres; keeps history and path tokens O(1)


@dataclass(frozen=True)
class WindowConfig:
    history_steps: int = HISTORY_STEPS
    horizon_steps: int = HORIZON_STEPS
    max_vertices: int = MAX_PATH_VERTICES
    stride: int = 2
    lookahead_steps: int = 250
    max_path_length: float = 25.0
    stop_speed: float = 0.2  # m/s
    stop_steps: int = 10  # a stop is one second below stop_speed
    epsilon_range: tuple[float, float] = (0.1, 0.4)
    idle_keep_probability: float = 0.15
    seed: int = 0


@dataclass
class WindowSet:
    history: list = field(default_factory=list)
    path: list = field(default_factory=list)
    path_valid: list = field(default_factory=list)
    gait: list = field(default_factory=list)
    target: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def to_local(points: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    """World ground-plane points to the frame whose +z is the facing direction."""
    delta = points - origin
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.stack(
        [cosine * delta[:, 0] - sine * delta[:, 1], sine * delta[:, 0] + cosine * delta[:, 1]],
        axis=1,
    )


def wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def history_features(local: np.ndarray, relative_yaw: np.ndarray) -> np.ndarray:
    """Encode recent root states; the last row is the current pose at the origin."""
    velocity = np.zeros_like(local)
    velocity[1:] = (local[1:] - local[:-1]) * CONTROL_HZ
    velocity[0] = velocity[1] if len(local) > 1 else 0.0
    return np.concatenate(
        [
            local / POSITION_SCALE,
            np.sin(relative_yaw)[:, None],
            np.cos(relative_yaw)[:, None],
            velocity / POSITION_SCALE,
        ],
        axis=1,
    ).astype(np.float32)


def path_tokens(vertices: np.ndarray, ends_at_goal: bool, max_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack polyline vertices (local frame, origin excluded) into padded tokens."""
    tokens = np.zeros((max_vertices, PATH_DIM), dtype=np.float32)
    valid = np.zeros(max_vertices, dtype=bool)
    count = len(vertices)
    if count > max_vertices:
        raise ValueError("path has more vertices than the model accepts")
    with_origin = np.concatenate([np.zeros((1, 2)), vertices]) if count else np.zeros((1, 2))
    along = arclength(with_origin)[1:] if count else np.zeros(0)
    tokens[:count, :2] = vertices / POSITION_SCALE
    tokens[:count, 2] = along / POSITION_SCALE
    if count and ends_at_goal:
        tokens[count - 1, 3] = 1.0
    valid[:count] = True
    return tokens, valid


def stopped_from(speed: np.ndarray, threshold: float, run: int) -> np.ndarray:
    """stopped_from[k] is True when speed stays under threshold for ``run`` steps from k."""
    slow = (speed < threshold).astype(np.int32)
    if len(slow) < run:
        return np.zeros(len(slow), dtype=bool)
    sums = np.convolve(slow, np.ones(run, dtype=np.int32), mode="valid")
    result = np.zeros(len(slow), dtype=bool)
    result[: len(sums)] = sums == run
    return result


def build_path(
    trace: np.ndarray, stopped: np.ndarray, config: WindowConfig, epsilon: float
) -> tuple[np.ndarray, bool, int] | None:
    """Simplify the upcoming trace into a polyline.

    trace: [L + 1, 2] local positions starting at the origin.
    stopped: [L] flags aligned with the steps of the trace.
    Returns (vertices without the origin, ends_at_goal, steps covered by the path).
    """
    steps = len(trace) - 1
    cut, ends_at_goal = steps, False
    along = arclength(trace)
    moving = np.flatnonzero(along > 0.3)
    onset = int(moving[0]) if len(moving) else steps
    later_stops = np.flatnonzero(stopped[onset:]) + onset
    if len(later_stops):
        cut, ends_at_goal = int(later_stops[0]), True
    elif not len(moving):
        ends_at_goal = True  # never leaves the spot: an idle window
        cut = 0
    over = np.flatnonzero(along > config.max_path_length)
    if len(over) and over[0] < cut:
        cut, ends_at_goal = int(over[0]), False
    if cut == 0:
        return np.zeros((0, 2)), True, steps
    kept = rdp(trace[: cut + 1], epsilon)[1:]
    if len(kept) > config.max_vertices:
        kept = kept[: config.max_vertices]
        cut, ends_at_goal = int(kept[-1]), False
    return trace[kept], ends_at_goal, cut


def windows_from_track(
    track: np.ndarray, gait: int, config: WindowConfig, rng: np.random.Generator,
    out: WindowSet, epsilon: float | None = None,
) -> None:
    history, horizon = config.history_steps, config.horizon_steps
    positions, yaw = track[:, :2].astype(np.float64), track[:, 2].astype(np.float64)
    speed = np.hypot(*np.diff(positions, axis=0).T) * CONTROL_HZ
    stopped = stopped_from(speed, config.stop_speed, config.stop_steps)
    for now in range(history - 1, len(track) - horizon, config.stride):
        last = min(len(track) - 1, now + config.lookahead_steps)
        trace = to_local(positions[now : last + 1], positions[now], yaw[now])
        chosen = epsilon if epsilon is not None else float(rng.uniform(*config.epsilon_range))
        vertices, ends_at_goal, covered = build_path(trace, stopped[now:last], config, chosen)
        future = trace[: horizon + 1]
        travelled = arclength(future)[-1]
        if len(vertices) == 0:
            if travelled > 0.3:
                out.drop("idle_path_but_moves")
                continue
            if rng.random() > config.idle_keep_probability:
                out.drop("idle_subsampled")
                continue
        elif covered < horizon:
            if not ends_at_goal:
                out.drop("path_shorter_than_horizon")
                continue
            # Arrived inside the horizon: the capture must stay put afterwards.
            if np.hypot(*(future[-1] - trace[covered])) > 0.3:
                out.drop("leaves_goal_within_horizon")
                continue
        if len(vertices) and stopped[now] and arclength(trace[: 6])[-1] < 0.1:
            out.drop("dawdles_before_start")
            continue

        past = to_local(positions[now - history + 1 : now + 1], positions[now], yaw[now])
        out.history.append(history_features(past, wrap(yaw[now - history + 1 : now + 1] - yaw[now])))
        tokens, valid = path_tokens(vertices, ends_at_goal, config.max_vertices)
        out.path.append(tokens)
        out.path_valid.append(valid)
        out.gait.append(gait)
        deltas = np.diff(future, axis=0)
        turn = wrap(np.diff(yaw[now : now + horizon + 1]))
        out.target.append(np.concatenate([deltas, turn[:, None]], axis=1).astype(np.float32))


class RootMotionWindows(Dataset):
    """All windows of the selected subjects, held in memory as tensors."""

    def __init__(
        self, tracks: dict[str, np.ndarray], subjects: set[int], config: WindowConfig,
        epsilon: float | None = None,
    ) -> None:
        rng = np.random.default_rng(config.seed)
        collected = WindowSet()
        self.clips: list[str] = []
        for name in sorted(tracks):
            parsed = parse_clip_name(name)
            if parsed is None or parsed[0] not in GAITS or parsed[1] not in subjects:
                continue
            self.clips.append(name)
            windows_from_track(
                tracks[name], GAITS.index(parsed[0]), config, rng, collected, epsilon
            )
        if not collected.target:
            raise ValueError(f"no windows for subjects {sorted(subjects)}")
        self.history = torch.from_numpy(np.stack(collected.history))
        self.path = torch.from_numpy(np.stack(collected.path))
        self.path_valid = torch.from_numpy(np.stack(collected.path_valid))
        self.gait = torch.tensor(collected.gait, dtype=torch.long)
        self.target = torch.from_numpy(np.stack(collected.target))
        self.dropped = collected.dropped

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.history[index],
            "path": self.path[index],
            "path_valid": self.path_valid[index],
            "gait": self.gait[index],
            "target": self.target[index],
        }

    def as_batch(self) -> dict[str, torch.Tensor]:
        return {
            "history": self.history, "path": self.path, "path_valid": self.path_valid,
            "gait": self.gait, "target": self.target,
        }


def mirror(batch: dict[str, torch.Tensor], flip: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reflect selected samples left-to-right. Doubles the data at no capture cost."""
    sign = torch.where(flip, -1.0, 1.0)
    history = batch["history"].clone()
    history[..., 0] *= sign[:, None]
    history[..., 2] *= sign[:, None]
    history[..., 4] *= sign[:, None]
    path = batch["path"].clone()
    path[..., 0] *= sign[:, None]
    target = batch["target"].clone()
    target[..., 0] *= sign[:, None]
    target[..., 2] *= sign[:, None]
    return {**batch, "history": history, "path": path, "target": target}


def path_polyline(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the metric polyline (origin first) and its validity mask from tokens."""
    vertices = batch["path"][..., :2] * POSITION_SCALE
    origin = torch.zeros_like(vertices[..., :1, :])
    valid = torch.cat([torch.ones_like(batch["path_valid"][..., :1]), batch["path_valid"]], dim=-1)
    return torch.cat([origin, vertices], dim=-2), valid


def trajectory_positions(deltas: torch.Tensor) -> torch.Tensor:
    """Cumulative local positions including the origin: [..., T + 1, 2]."""
    cumulative = deltas[..., :2].cumsum(dim=-2)
    return torch.cat([torch.zeros_like(cumulative[..., :1, :]), cumulative], dim=-2)

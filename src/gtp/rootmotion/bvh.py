"""Minimal BVH reader that extracts only what the root-motion layer needs.

Only the ROOT joint's six channels are decoded. Decoding the full skeleton would
be wasted work: the layer models the planar root trajectory and facing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RootMotion:
    """Root translation, rotation and the hip axis in the BVH's own frame."""

    positions: np.ndarray  # [frames, 3], file units
    rotations: np.ndarray  # [frames, 3, 3]
    across_local: np.ndarray  # unit vector from the right hip to the left hip
    frame_time: float


def _axis_rotation(axis: str, degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    zero, one = np.zeros_like(radians), np.ones_like(radians)
    if axis == "X":
        rows = [[one, zero, zero], [zero, cosine, -sine], [zero, sine, cosine]]
    elif axis == "Y":
        rows = [[cosine, zero, sine], [zero, one, zero], [-sine, zero, cosine]]
    elif axis == "Z":
        rows = [[cosine, -sine, zero], [sine, cosine, zero], [zero, zero, one]]
    else:
        raise ValueError(f"unknown rotation axis: {axis}")
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def read_root_motion(path: str | Path) -> RootMotion:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    channels: list[str] | None = None
    depth = 0
    pending: str | None = None
    hip_offsets: dict[str, np.ndarray] = {}
    motion_start = -1
    for index, raw in enumerate(lines):
        tokens = raw.split()
        if not tokens:
            continue
        head = tokens[0]
        if head == "MOTION":
            motion_start = index
            break
        if head in {"ROOT", "JOINT"}:
            pending = tokens[1] if len(tokens) > 1 else ""
        elif head == "{":
            depth += 1
        elif head == "}":
            depth -= 1
        elif head == "OFFSET" and depth == 2 and pending is not None:
            # Direct children of the root: the two hip joints define the lateral axis.
            hip_offsets[pending.lower()] = np.array([float(v) for v in tokens[1:4]])
        elif head == "CHANNELS" and depth == 1 and channels is None:
            channels = tokens[2 : 2 + int(tokens[1])]
    if motion_start < 0 or channels is None:
        raise ValueError(f"{path} is not a BVH file with a ROOT joint")
    if len(channels) != 6:
        raise ValueError(f"{path}: ROOT must have six channels, found {len(channels)}")

    left = [offset for name, offset in hip_offsets.items() if "left" in name]
    right = [offset for name, offset in hip_offsets.items() if "right" in name]
    if len(left) != 1 or len(right) != 1:
        raise ValueError(f"{path}: expected one left and one right child of the root")
    across = left[0] - right[0]
    across_norm = float(np.linalg.norm(across))
    if across_norm < 1e-6:
        raise ValueError(f"{path}: hip joints share an offset; facing is undefined")

    frame_time = 0.0
    data_start = motion_start + 1
    for index in range(motion_start + 1, min(motion_start + 4, len(lines))):
        if lines[index].strip().lower().startswith("frame time"):
            frame_time = float(lines[index].split(":")[1])
            data_start = index + 1
    if frame_time <= 0:
        raise ValueError(f"{path}: missing frame time")
    rows = [
        [float(value) for value in line.split(None, 6)[:6]]
        for line in lines[data_start:]
        if line.strip()
    ]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"{path}: motion block is malformed")

    positions = np.zeros((len(values), 3))
    rotation = np.broadcast_to(np.eye(3), (len(values), 3, 3)).copy()
    for column, channel in enumerate(channels):
        axis, kind = channel[0].upper(), channel[1:].lower()
        if kind == "position":
            positions[:, "XYZ".index(axis)] = values[:, column]
        elif kind == "rotation":
            # BVH composes rotations in channel order, left to right.
            rotation = rotation @ _axis_rotation(axis, values[:, column])
        else:
            raise ValueError(f"{path}: unsupported channel {channel}")
    return RootMotion(positions, rotation, across / across_norm, frame_time)

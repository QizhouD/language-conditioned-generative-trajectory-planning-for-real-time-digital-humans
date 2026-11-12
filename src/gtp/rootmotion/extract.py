"""Turn BVH clips into smoothed 10 Hz planar root tracks.

Output convention matches the Unity side: ground plane (x, z), y up, left-handed,
yaw = atan2(forward_x, forward_z) so yaw 0 faces +z and positive yaw turns right.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from gtp.rootmotion import CONTROL_HZ, GAITS
from gtp.rootmotion.bvh import read_root_motion

_CLIP_NAME = re.compile(r"^([a-zA-Z]+?)(\d+)_subject(\d+)$")


def savgol_kernel(window: int, order: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing weights for the window centre."""
    if window % 2 == 0 or window <= order:
        raise ValueError("window must be odd and larger than the polynomial order")
    half = window // 2
    design = np.vander(np.arange(-half, half + 1, dtype=np.float64), order + 1, increasing=True)
    return np.linalg.pinv(design)[0]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Cubic Savitzky-Golay along axis 0 with edge replication."""
    if window <= 3 or len(values) < window:
        return values.copy()
    kernel = savgol_kernel(window)
    half = window // 2
    padded = np.concatenate(
        [np.repeat(values[:1], half, axis=0), values, np.repeat(values[-1:], half, axis=0)]
    )
    columns = [np.convolve(padded[:, c], kernel[::-1], mode="valid") for c in range(values.shape[1])]
    return np.stack(columns, axis=1)


def root_track(
    path: str | Path,
    *,
    unit_scale: float = 0.01,
    smooth_seconds: float = 0.5,
    control_hz: int = CONTROL_HZ,
) -> np.ndarray:
    """Return [steps, 3] rows of (x, z, yaw) at the control rate."""
    motion = read_root_motion(path)
    source_hz = 1.0 / motion.frame_time
    stride = int(round(source_hz / control_hz))
    if stride < 1 or abs(source_hz / stride - control_hz) > 0.05 * control_hz:
        raise ValueError(f"{path}: {source_hz:.2f} fps is not a multiple of {control_hz} Hz")
    window = int(round(smooth_seconds * source_hz)) | 1

    # A BVH is right-handed with y up. Negating x gives Unity's left-handed ground plane.
    ground = np.stack([-motion.positions[:, 0], motion.positions[:, 2]], axis=1) * unit_scale
    across = motion.rotations @ motion.across_local  # points to the character's left
    # In a right-handed frame, forward = left x up.
    forward = np.cross(across, np.array([0.0, 1.0, 0.0]))
    facing = np.stack([-forward[:, 0], forward[:, 2]], axis=1)

    ground = smooth(ground, window)
    facing = smooth(facing, window)
    norms = np.linalg.norm(facing, axis=1, keepdims=True)
    facing = facing / np.clip(norms, 1e-8, None)
    yaw = np.unwrap(np.arctan2(facing[:, 0], facing[:, 1]))
    track = np.concatenate([ground, yaw[:, None]], axis=1)
    return track[::stride].astype(np.float32)


def parse_clip_name(stem: str) -> tuple[str, int] | None:
    match = _CLIP_NAME.match(stem)
    if not match:
        return None
    return match.group(1).lower(), int(match.group(3))


def extract_directory(bvh_dir: Path, output: Path, gaits: tuple[str, ...] = GAITS) -> dict:
    tracks: dict[str, np.ndarray] = {}
    for path in sorted(bvh_dir.glob("*.bvh")):
        parsed = parse_clip_name(path.stem)
        if parsed is None or parsed[0] not in gaits:
            continue
        tracks[path.stem] = root_track(path)
    if not tracks:
        raise SystemExit(f"no clips for gaits {gaits} under {bvh_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **tracks)
    seconds = sum(len(track) for track in tracks.values()) / CONTROL_HZ
    return {"clips": len(tracks), "minutes": seconds / 60.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/lafan1_root.npz"))
    args = parser.parse_args()
    summary = extract_directory(args.bvh_dir, args.output)
    print(f"{args.output}: {summary['clips']} clips, {summary['minutes']:.1f} minutes")


if __name__ == "__main__":
    main()

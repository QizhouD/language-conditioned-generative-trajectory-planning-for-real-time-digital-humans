import math

import numpy as np
import pytest
import torch

from gtp.rootmotion.bvh import read_root_motion
from gtp.rootmotion.extract import root_track
from gtp.rootmotion.paths import arclength, distance_to_polyline, rdp
from gtp.rootmotion.windows import (
    POSITION_SCALE, WindowConfig, WindowSet, mirror, path_polyline, trajectory_positions,
    windows_from_track,
)

HEADER = """HIERARCHY
ROOT Hips
{{
\tOFFSET 0 0 0
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
\tJOINT LeftUpLeg
\t{{
\t\tOFFSET 10 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tEnd Site
\t\t{{
\t\t\tOFFSET 0 -40 0
\t\t}}
\t}}
\tJOINT RightUpLeg
\t{{
\t\tOFFSET -10 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tEnd Site
\t\t{{
\t\t\tOFFSET 0 -40 0
\t\t}}
\t}}
}}
MOTION
Frames: {frames}
Frame Time: 0.033333
"""


def write_bvh(path, rows):
    lines = [" ".join(f"{value:.6f}" for value in (*row, 0, 0, 0, 0, 0, 0)) for row in rows]
    path.write_text(HEADER.format(frames=len(rows)) + "\n".join(lines) + "\n", encoding="utf-8")


def test_bvh_reader_decodes_root_channels(tmp_path):
    path = tmp_path / "walk1_subject1.bvh"
    write_bvh(path, [(1.0, 90.0, 2.0, 0.0, 90.0, 0.0), (3.0, 90.0, 4.0, 0.0, 0.0, 0.0)])
    motion = read_root_motion(path)
    assert motion.positions.shape == (2, 3)
    assert motion.positions[1].tolist() == [3.0, 90.0, 4.0]
    assert motion.across_local.tolist() == [1.0, 0.0, 0.0]
    # A +90 degree yaw in a right-handed y-up frame carries +z onto +x.
    assert np.allclose(motion.rotations[0] @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(motion.rotations[1], np.eye(3))


def test_root_track_uses_unity_ground_convention(tmp_path):
    # Walk along BVH +z at 1 m/s (centimetres, 30 fps) facing +z; left hip is on +x.
    forward = tmp_path / "walk1_subject1.bvh"
    write_bvh(forward, [(0.0, 90.0, 100.0 * frame / 30.0, 0.0, 0.0, 0.0) for frame in range(90)])
    track = root_track(forward)
    assert track.shape == (30, 3)
    assert np.allclose(track[:, 0], 0.0, atol=1e-4)
    assert np.allclose(np.diff(track[5:-5, 1]), 0.1, atol=1e-3)
    assert np.allclose(track[:, 2], 0.0, atol=1e-4)

    # Facing BVH +x is facing Unity -x, which is yaw -90 degrees.
    turned = tmp_path / "walk2_subject1.bvh"
    write_bvh(turned, [(100.0 * frame / 30.0, 90.0, 0.0, 0.0, 90.0, 0.0) for frame in range(90)])
    track = root_track(turned)
    assert np.allclose(track[:, 2], -math.pi / 2, atol=1e-4)
    assert np.all(np.diff(track[5:-5, 0]) < 0)


def test_rdp_keeps_corners_and_bounds_error():
    leg = np.linspace(0.0, 3.0, 31)
    points = np.concatenate(
        [np.stack([np.zeros(31), leg], 1), np.stack([leg[1:], np.full(30, 3.0)], 1)]
    )
    kept = rdp(points, 0.05)
    assert kept.tolist() == [0, 30, 60]
    wavy = np.stack([0.2 * np.sin(np.linspace(0, 12, 200)), np.linspace(0, 10, 200)], 1)
    simplified = wavy[rdp(wavy, 0.1)]
    distance = distance_to_polyline(
        torch.tensor(wavy), torch.tensor(simplified), torch.ones(len(simplified), dtype=torch.bool)
    )
    assert float(distance.max()) <= 0.1 + 1e-6
    assert arclength(points)[-1] == pytest.approx(6.0)


def test_distance_to_polyline_respects_padding():
    polyline = torch.tensor([[0.0, 0.0], [0.0, 2.0], [50.0, 50.0]])
    valid = torch.tensor([True, True, False])
    points = torch.tensor([[1.0, 1.0], [0.0, 3.0], [0.5, -0.5]])
    distance = distance_to_polyline(points, polyline, valid)
    assert distance.tolist() == pytest.approx([1.0, 1.0, math.sqrt(0.5)])
    single = distance_to_polyline(points, polyline, torch.tensor([True, False, False]))
    assert single.tolist() == pytest.approx([math.sqrt(2.0), 3.0, math.sqrt(0.5)])


def straight_then_stop(moving_steps=80, rest_steps=60, speed=1.2):
    z = np.concatenate([np.zeros(45), np.arange(1, moving_steps + 1) * speed / 10.0])
    z = np.concatenate([z, np.full(rest_steps, z[-1])])
    return np.stack([np.zeros_like(z), z, np.zeros_like(z)], axis=1).astype(np.float32)


def test_windows_mark_the_goal_and_stay_on_the_path():
    out = WindowSet()
    config = WindowConfig(stride=5, idle_keep_probability=0.0)
    windows_from_track(straight_then_stop(), 0, config, np.random.default_rng(0), out, epsilon=0.25)
    assert len(out.target) > 10
    batch = {
        "path": torch.from_numpy(np.stack(out.path)),
        "path_valid": torch.from_numpy(np.stack(out.path_valid)),
        "target": torch.from_numpy(np.stack(out.target)),
    }
    count = batch["path_valid"].sum(1)
    assert int(count.max()) == 1  # a straight walk simplifies to its end point
    rows = torch.arange(len(count))
    assert bool((batch["path"][rows, count - 1, 3] == 1.0).all())  # every path ends at the stop
    polyline, valid = path_polyline(batch)
    deviation = distance_to_polyline(trajectory_positions(batch["target"]), polyline, valid)
    assert float(deviation.max()) < 0.25
    # The goal is where the capture actually came to rest.
    assert float(polyline[0, 1, 1]) == pytest.approx(batch["path"][0, 0, 2].item() * POSITION_SCALE)
    assert out.dropped.get("dawdles_before_start", 0) > 0


def test_mirror_is_an_involution_and_flips_lateral_terms():
    generator = torch.Generator().manual_seed(0)
    batch = {
        "history": torch.randn(4, 20, 6, generator=generator),
        "path": torch.randn(4, 16, 4, generator=generator),
        "path_valid": torch.ones(4, 16, dtype=torch.bool),
        "gait": torch.zeros(4, dtype=torch.long),
        "target": torch.randn(4, 30, 3, generator=generator),
    }
    flip = torch.tensor([True, False, True, False])
    once = mirror(batch, flip)
    assert torch.equal(once["target"][1], batch["target"][1])
    assert torch.equal(once["target"][0, :, 0], -batch["target"][0, :, 0])
    assert torch.equal(once["target"][0, :, 1], batch["target"][0, :, 1])
    assert torch.equal(once["target"][0, :, 2], -batch["target"][0, :, 2])
    twice = mirror(once, flip)
    for key in ("history", "path", "target"):
        assert torch.equal(twice[key], batch[key])

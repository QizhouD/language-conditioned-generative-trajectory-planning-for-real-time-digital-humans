"""Hard runtime checks for root-motion plans.

The corridor check is a swept test with an explicit bound. Sub-sampling each
0.1 s segment ``subsamples`` times leaves at most half a sample spacing between
any point of the segment and a checked point, and distance-to-polyline is
1-Lipschitz. So if every sample is within ``half_width - spacing / 2`` of the
path, the whole continuous piecewise-linear trajectory is within ``half_width``.
The check starts at the character's current position, not at the first step.

This bounds deviation from the path the engine supplied. Whether that corridor
is walkable is the engine's knowledge, so the Unity side re-verifies every
segment against the NavMesh before executing a plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gtp.rootmotion import CONTROL_HZ
from gtp.rootmotion.paths import distance_to_polyline
from gtp.rootmotion.windows import trajectory_positions


@dataclass(frozen=True)
class KinematicLimits:
    max_speed: float = 8.0  # m/s
    max_acceleration: float = 20.0  # m/s^2
    max_turn_rate: float = 10.0  # rad/s


def kinematic_violation(
    deltas: torch.Tensor, limits: KinematicLimits = KinematicLimits(),
    initial_velocity: torch.Tensor | None = None, control_hz: int = CONTROL_HZ,
) -> torch.Tensor:
    """True per trajectory when any step breaks a limit or is non-finite.

    deltas: [..., T, 3]. initial_velocity: [..., 2] makes the first step's
    acceleration count, which ties a plan to the motion already under way.
    """
    velocity = deltas[..., :2] * control_hz
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    change = velocity[..., 1:, :] - velocity[..., :-1, :]
    if initial_velocity is not None:
        change = torch.cat((velocity[..., :1, :] - initial_velocity.unsqueeze(-2), change), dim=-2)
    acceleration = torch.linalg.vector_norm(change, dim=-1) * control_hz
    turn_rate = deltas[..., 2].abs() * control_hz
    finite = torch.isfinite(deltas).all(dim=-1).all(dim=-1)
    # NaN compares False, so test for "within limits" and negate.
    within = (
        (speed.amax(-1) <= limits.max_speed)
        & (acceleration.amax(-1) <= limits.max_acceleration)
        & (turn_rate.amax(-1) <= limits.max_turn_rate)
    )
    return ~(finite & within)


def corridor_deviation(
    deltas: torch.Tensor, polyline: torch.Tensor, valid: torch.Tensor, subsamples: int = 4,
) -> torch.Tensor:
    """Upper bound on the swept trajectory's distance from the path: [...]."""
    positions = trajectory_positions(deltas)  # [..., T + 1, 2], origin first
    start, end = positions[..., :-1, :], positions[..., 1:, :]
    fractions = torch.arange(subsamples, device=deltas.device, dtype=deltas.dtype) / subsamples
    samples = start.unsqueeze(-2) + fractions[:, None] * (end - start).unsqueeze(-2)
    samples = torch.cat((samples.flatten(-3, -2), positions[..., -1:, :]), dim=-2)
    sampled = distance_to_polyline(samples, polyline, valid).amax(-1)
    spacing = torch.linalg.vector_norm(end - start, dim=-1).amax(-1) / subsamples
    return sampled + 0.5 * spacing


def corridor_violation(
    deltas: torch.Tensor, polyline: torch.Tensor, valid: torch.Tensor,
    half_width: float = 0.5, subsamples: int = 4,
) -> torch.Tensor:
    bound = corridor_deviation(deltas, polyline, valid, subsamples)
    return ~(bound <= half_width)

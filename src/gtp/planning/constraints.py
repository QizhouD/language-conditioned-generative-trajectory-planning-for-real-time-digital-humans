"""Hard runtime checks for generated root trajectories."""

from __future__ import annotations

import torch


def cumulative_positions(trajectory: torch.Tensor) -> torch.Tensor:
    return trajectory[..., :2].cumsum(dim=-2)


def collision_mask(
    trajectory: torch.Tensor,
    obstacles: torch.Tensor,
    obstacle_valid: torch.Tensor,
    clearance: float = 0.3,
) -> torch.Tensor:
    """Return one boolean per candidate; True means at least one collision."""
    positions = cumulative_positions(trajectory)
    centers = obstacles[..., :2]
    radii = obstacles[..., 2] + clearance
    distances = torch.linalg.vector_norm(
        positions.unsqueeze(-2) - centers.unsqueeze(-3), dim=-1
    )
    collisions = (distances < radii.unsqueeze(-2)) & obstacle_valid.unsqueeze(-2)
    return collisions.any(dim=(-1, -2))


def kinematic_violation_mask(
    trajectory: torch.Tensor,
    max_speed: float = 3.0,
    max_angular_speed: float = 4.0,
    control_hz: float = 10.0,
) -> torch.Tensor:
    speed_invalid = trajectory[..., 7].abs().amax(dim=-1) > max_speed
    negative_speed = trajectory[..., 7].amin(dim=-1) < 0.0
    implied_speed = torch.linalg.vector_norm(trajectory[..., :2], dim=-1) * control_hz
    displacement_invalid = implied_speed.amax(dim=-1) > max_speed
    angular_invalid = trajectory[..., 6].abs().amax(dim=-1) > max_angular_speed
    finite_invalid = ~torch.isfinite(trajectory).all(dim=(-1, -2))
    return speed_invalid | negative_speed | displacement_invalid | angular_invalid | finite_invalid

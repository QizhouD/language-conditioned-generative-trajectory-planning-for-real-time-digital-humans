"""Cost-based candidate selection."""

from __future__ import annotations

import torch

from gtp.planning.constraints import collision_mask, cumulative_positions


def candidate_costs(
    trajectories: torch.Tensor,
    goal: torch.Tensor,
    obstacles: torch.Tensor,
    obstacle_valid: torch.Tensor,
) -> torch.Tensor:
    positions = cumulative_positions(trajectories)
    endpoint = positions[..., -1, :]
    goal_error = torch.linalg.vector_norm(endpoint - goal[..., None, :2], dim=-1)
    path_length = torch.linalg.vector_norm(trajectories[..., :2], dim=-1).sum(dim=-1)
    velocity = trajectories[..., 4:6]
    acceleration = velocity[..., 1:, :] - velocity[..., :-1, :]
    jerk = acceleration[..., 1:, :] - acceleration[..., :-1, :]
    smoothness = jerk.square().mean(dim=(-1, -2))
    final_yaw = torch.atan2(trajectories[..., -1, 2], trajectories[..., -1, 3])
    desired_yaw = torch.atan2(goal[..., 2], goal[..., 3]).unsqueeze(-1)
    facing_error = torch.atan2(
        torch.sin(final_yaw - desired_yaw), torch.cos(final_yaw - desired_yaw)
    ).abs()
    collision = collision_mask(
        trajectories,
        obstacles.unsqueeze(1),
        obstacle_valid.unsqueeze(1),
    )
    return (
        2.0 * goal_error
        + 0.2 * path_length
        + 0.5 * smoothness
        + facing_error
        + collision.float() * 10_000.0
    )

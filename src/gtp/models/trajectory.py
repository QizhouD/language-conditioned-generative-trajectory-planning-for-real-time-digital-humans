"""Conditional models for fixed-horizon root trajectories."""

from __future__ import annotations

import math

import torch
from torch import nn

from gtp.schema import MOTION_INTENTS


class ConditionEncoder(nn.Module):
    def __init__(
        self, state_dim: int, goal_dim: int, obstacle_dim: int, hidden_dim: int,
        layers: int, heads: int,
    ) -> None:
        super().__init__()
        self.history_projection = nn.Linear(state_dim, hidden_dim)
        history_layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, batch_first=True
        )
        self.history_encoder = nn.TransformerEncoder(history_layer, layers)
        self.goal_encoder = nn.Sequential(nn.Linear(goal_dim, hidden_dim), nn.SiLU())
        self.behavior_embedding = nn.Embedding(len(MOTION_INTENTS), hidden_dim)
        self.obstacle_projection = nn.Linear(obstacle_dim, hidden_dim)
        obstacle_layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 2, batch_first=True
        )
        self.obstacle_encoder = nn.TransformerEncoder(obstacle_layer, max(1, layers // 2))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.SiLU())

    @staticmethod
    def _pool(encoded: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).float()
        return (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history = self.history_encoder(
            self.history_projection(batch["history"]),
            src_key_padding_mask=~batch["history_valid"],
        )
        obstacles = self.obstacle_encoder(
            self.obstacle_projection(batch["obstacles"]),
            src_key_padding_mask=~batch["obstacle_valid"],
        )
        return self.fusion(
            torch.cat(
                (
                    self._pool(history, batch["history_valid"]),
                    self.goal_encoder(batch["goal"]),
                    self.behavior_embedding(batch["behavior"]),
                    self._pool(obstacles, batch["obstacle_valid"]),
                ),
                dim=-1,
            )
        )


class DeterministicTrajectoryBaseline(nn.Module):
    def __init__(
        self, state_dim: int, goal_dim: int, obstacle_dim: int, trajectory_dim: int,
        hidden_dim: int, layers: int, heads: int, horizon_steps: int,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.trajectory_dim = trajectory_dim
        self.encoder = ConditionEncoder(
            state_dim, goal_dim, obstacle_dim, hidden_dim, layers, heads
        )
        self.trajectory_head = nn.Linear(hidden_dim, horizon_steps * trajectory_dim)
        self.action_head = nn.Linear(hidden_dim, horizon_steps * len(MOTION_INTENTS))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition = self.encoder(batch)
        return {
            "trajectory": self.trajectory_head(condition).view(
                -1, self.horizon_steps, self.trajectory_dim
            ),
            "action_logits": self.action_head(condition).view(
                -1, self.horizon_steps, len(MOTION_INTENTS)
            ),
        }


class ConditionalTrajectoryFlow(nn.Module):
    def __init__(
        self, state_dim: int, goal_dim: int, obstacle_dim: int, trajectory_dim: int,
        hidden_dim: int, layers: int, heads: int, horizon_steps: int,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.trajectory_dim = trajectory_dim
        self.hidden_dim = hidden_dim
        self.encoder = ConditionEncoder(
            state_dim, goal_dim, obstacle_dim, hidden_dim, layers, heads
        )
        features = horizon_steps * trajectory_dim
        self.vector_field = nn.Sequential(
            nn.Linear(features + hidden_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, features),
        )
        self.action_head = nn.Sequential(
            nn.Linear(trajectory_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(MOTION_INTENTS)),
        )

    def _time_embedding(self, time: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        angles = time[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding

    def velocity(
        self, state: torch.Tensor, time: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        condition = self.encoder(batch)
        features = torch.cat(
            (state.flatten(1), condition, self._time_embedding(time)), dim=-1
        )
        return self.vector_field(features).view(
            -1, self.horizon_steps, self.trajectory_dim
        )

    def action_logits(
        self, trajectory: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        condition = self.encoder(batch).unsqueeze(1).expand(-1, self.horizon_steps, -1)
        return self.action_head(torch.cat((trajectory, condition), dim=-1))

    @torch.no_grad()
    def sample(
        self, batch: dict[str, torch.Tensor], steps: int = 4, samples: int = 1
    ) -> dict[str, torch.Tensor]:
        expanded = {
            key: value.repeat_interleave(samples, dim=0)
            for key, value in batch.items()
            if key not in {"trajectory", "actions"}
        }
        count = expanded["history"].shape[0]
        state = torch.randn(
            count, self.horizon_steps, self.trajectory_dim,
            device=expanded["history"].device,
        )
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full((count,), index * delta, device=state.device)
            state = state + delta * self.velocity(state, time, expanded)
        return {
            "trajectory": state,
            "action_logits": self.action_logits(state, expanded),
        }

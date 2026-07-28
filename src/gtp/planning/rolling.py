"""Rolling-horizon wrapper around a conditional trajectory model."""

from __future__ import annotations

import torch

from gtp.planning.constraints import kinematic_violation_mask
from gtp.planning.ranker import candidate_costs


class RollingTrajectoryPlanner:
    def __init__(
        self, model, candidates: int = 16, flow_steps: int = 4, execute_steps: int = 3
    ) -> None:
        self.model = model
        self.candidates = candidates
        self.flow_steps = flow_steps
        self.execute_steps = execute_steps

    @torch.no_grad()
    def plan(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.model.sample(
            batch, steps=self.flow_steps, samples=self.candidates
        )
        batch_size = batch["history"].shape[0]
        horizon, dimensions = output["trajectory"].shape[-2:]
        trajectories = output["trajectory"].view(
            batch_size, self.candidates, horizon, dimensions
        )
        logits = output["action_logits"].view(
            batch_size, self.candidates, horizon, -1
        )
        costs = candidate_costs(
            trajectories, batch["goal"], batch["obstacles"], batch["obstacle_valid"]
        )
        invalid = kinematic_violation_mask(trajectories)
        costs = costs + invalid.float() * 10_000.0
        selected = costs.argmin(dim=1)
        rows = torch.arange(batch_size, device=selected.device)
        best = trajectories[rows, selected]
        best_logits = logits[rows, selected]
        fallback = costs[rows, selected] >= 10_000.0
        if fallback.any():
            best[fallback] = 0.0
            best[fallback, :, 3] = 1.0
        return {
            "trajectory": best,
            "actions": best_logits.argmax(dim=-1),
            "cost": costs[rows, selected],
            "fallback": fallback,
            "execute_prefix": best[:, : self.execute_steps],
        }

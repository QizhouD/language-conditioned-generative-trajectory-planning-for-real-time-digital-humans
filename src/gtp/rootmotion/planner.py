"""Sample root-motion candidates, reject the invalid, fall back to a spring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gtp.rootmotion.baselines import spring_follow
from gtp.rootmotion.constraints import (
    KinematicLimits, corridor_deviation, corridor_violation, kinematic_violation,
)
from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.windows import POSITION_SCALE, path_polyline

SOURCE_MODEL, SOURCE_SPRING, SOURCE_HOLD = 0, 1, 2
SOURCE_NAMES = ("model", "spring_fallback", "hold")


@dataclass(frozen=True)
class PlannerConfig:
    candidates: int = 16
    flow_steps: int = 8
    half_width: float = 0.5
    subsamples: int = 4
    # "first_valid" is rejection sampling: it keeps the model's distribution,
    # truncated by the constraints. "closest" prefers hugging the path.
    selection: str = "first_valid"
    limits: KinematicLimits = field(default_factory=KinematicLimits)


def initial_velocity(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["history"][:, -1, 4:6] * POSITION_SCALE


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RootMotionModel(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), RootMotionStats.from_dict(checkpoint["stats"]), checkpoint


class RootMotionPlanner:
    def __init__(
        self, model: RootMotionModel, stats: RootMotionStats,
        config: PlannerConfig = PlannerConfig(), device: str | torch.device = "cpu",
    ) -> None:
        if config.selection not in {"first_valid", "closest"}:
            raise ValueError(f"unknown selection rule: {config.selection}")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.stats = stats
        self.config = config

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, config: PlannerConfig = PlannerConfig(),
        device: str | torch.device = "cpu",
    ) -> "RootMotionPlanner":
        model, stats, _ = load_checkpoint(path, device)
        return cls(model, stats, config, device)

    def invalid(
        self, deltas: torch.Tensor, batch: dict[str, torch.Tensor], half_width: float | None = None,
    ) -> torch.Tensor:
        """Constraint check for candidates shaped [B, K, T, 3]."""
        polyline, valid = path_polyline(batch)
        width = self.config.half_width if half_width is None else half_width
        corridor = corridor_violation(
            deltas, polyline.unsqueeze(1), valid.unsqueeze(1), width, self.config.subsamples
        )
        kinematic = kinematic_violation(
            deltas, self.config.limits, initial_velocity(batch).unsqueeze(1)
        )
        return corridor | kinematic

    def spring_fallback(self, batch: dict[str, torch.Tensor], row: int) -> np.ndarray:
        polyline, valid = path_polyline(batch)
        vertices = polyline[row][valid[row]].cpu().numpy()
        count = int(batch["path_valid"][row].sum())
        ends_at_goal = count == 0 or bool(batch["path"][row, count - 1, 3] > 0.5)
        return spring_follow(
            vertices, ends_at_goal, initial_velocity(batch)[row].cpu().numpy(),
            self.stats.gait_speed[int(batch["gait"][row])],
            steps=self.model.horizon_steps,
        )

    @torch.no_grad()
    def plan_batch(
        self, batch: dict[str, torch.Tensor], generator: torch.Generator | None = None,
        half_width: float | None = None, candidates: int | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = {key: value.to(self.device) for key, value in batch.items()}
        normalised = self.model.sample(
            batch, steps=self.config.flow_steps,
            samples=self.config.candidates if candidates is None else candidates,
            generator=generator,
        )
        candidates = self.stats.denormalise(normalised)
        invalid = self.invalid(candidates, batch, half_width)
        rows = torch.arange(candidates.shape[0], device=self.device)
        if self.config.selection == "closest":
            polyline, valid = path_polyline(batch)
            cost = corridor_deviation(
                candidates, polyline.unsqueeze(1), valid.unsqueeze(1), self.config.subsamples
            )
            choice = cost.masked_fill(invalid, float("inf")).argmin(dim=1)
        else:
            choice = (~invalid).float().argmax(dim=1)
        deltas = candidates[rows, choice].clone()
        source = torch.full((len(rows),), SOURCE_MODEL, device=self.device)
        for row in invalid.all(dim=1).nonzero(as_tuple=True)[0].tolist():
            fallback = torch.as_tensor(
                self.spring_fallback(batch, row), dtype=deltas.dtype, device=self.device
            )
            single = {key: value[row : row + 1] for key, value in batch.items()}
            if bool(self.invalid(fallback[None, None], single, half_width)[0, 0]):
                deltas[row] = 0.0
                source[row] = SOURCE_HOLD
            else:
                deltas[row] = fallback
                source[row] = SOURCE_SPRING
        return {
            "deltas": deltas, "source": source, "candidates": candidates,
            "candidate_invalid": invalid,
        }

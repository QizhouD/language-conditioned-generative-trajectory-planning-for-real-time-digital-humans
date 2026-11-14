"""Sample root-motion candidates, reject the invalid, fall back to a spring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gtp.rootmotion import CONTROL_HZ
from gtp.rootmotion.baselines import spring_follow
from gtp.rootmotion.constraints import (
    KinematicLimits, corridor_deviation, corridor_violation, kinematic_violation,
)
from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.windows import POSITION_SCALE, path_polyline, trajectory_positions

SOURCE_MODEL, SOURCE_SPRING, SOURCE_HOLD = 0, 1, 2
SOURCE_NAMES = ("model", "spring_fallback", "hold")
SELECTION_RULES = ("first_valid", "closest", "goal_aware", "consistent")
# goal_aware cost: endpoint distance to the goal (m) + this many seconds of final speed (m/s).
GOAL_AWARE_SPEED_WEIGHT = 0.5
# consistent cost: mean distance (m) to the previous plan's unexecuted positions over the
# overlapping steps + this many seconds of first-step velocity difference (m/s).
CONSISTENCY_VELOCITY_WEIGHT = 0.1


@dataclass(frozen=True)
class PlannerConfig:
    candidates: int = 16
    flow_steps: int = 8
    half_width: float = 0.5
    subsamples: int = 4
    # "first_valid" is rejection sampling: it keeps the model's distribution,
    # truncated by the constraints. "closest" prefers hugging the path.
    # "goal_aware" (the default) is first_valid, except when the goal is within
    # reach of this plan: then it takes the valid candidate that ends nearest the
    # goal and slowest (see goal_within_reach and GOAL_AWARE_SPEED_WEIGHT). With a
    # single candidate (the regressor) every rule picks the same plan.
    # "consistent" takes the valid candidate closest to the unexecuted rest of the
    # previous plan (plan_batch's ``previous``); near the goal, or without a previous
    # plan, it is goal_aware.
    selection: str = "goal_aware"
    limits: KinematicLimits = field(default_factory=KinematicLimits)


def goal_within_reach(
    batch: dict[str, torch.Tensor], speed: torch.Tensor, horizon_steps: int,
    control_hz: int = CONTROL_HZ,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows whose path ends at the goal no farther than speed x plan duration, and the goal.

    ``speed`` is per row (m/s). An empty path means standing at the goal (the origin).
    """
    count = batch["path_valid"].sum(dim=1)
    rows = torch.arange(len(count), device=count.device)
    last = (count - 1).clamp_min(0)
    ends = (count == 0) | (batch["path"][rows, last, 3] > 0.5)
    remaining = torch.where(count > 0, batch["path"][rows, last, 2] * POSITION_SCALE,
                            torch.zeros_like(batch["path"][:, 0, 2]))
    polyline, _ = path_polyline(batch)
    goal = polyline[rows, count]
    return ends & (remaining <= speed.to(remaining.dtype) * horizon_steps / control_hz), goal


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
        if config.selection not in SELECTION_RULES:
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

    def reference_speed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-row speed (m/s) for the spring fallback and goal-aware selection.

        The gait median, or the row's target speed when the model is speed-conditioned
        and one is given. An unconditioned model ignores target_speed entirely, so old
        checkpoints behave exactly as before.
        """
        speed = torch.as_tensor(self.stats.gait_speed, dtype=torch.float32,
                                device=batch["gait"].device)[batch["gait"]]
        target = batch.get("target_speed")
        if self.model.speed_conditioning and target is not None:
            target = target.reshape(-1).to(speed.dtype)
            speed = torch.where(torch.isfinite(target), target, speed)
        return speed

    def clip_target_speed(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Clamp given target speeds into the training range of their gait.

        Returns the batch with clamped speeds and a per-row mask of the rows that changed.
        """
        speed = batch["target_speed"].reshape(-1).to(torch.float32)
        given = torch.isfinite(speed)
        if self.stats.speed_range is None:
            return batch, torch.zeros_like(given)
        ranges = torch.as_tensor(self.stats.speed_range, dtype=speed.dtype, device=speed.device)
        low, high = ranges[batch["gait"], 0], ranges[batch["gait"], 1]
        clamped = torch.minimum(torch.maximum(speed, low), high)
        changed = given & (clamped != speed)
        return {**batch, "target_speed": torch.where(given, clamped, speed)}, changed

    @staticmethod
    def consistency_cost(candidates: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Per candidate [B, K]: how far it departs from the previous plan's unexecuted part.

        Mean distance between the candidate's and the previous plan's positions over the
        steps both cover, plus CONSISTENCY_VELOCITY_WEIGHT x the difference between their
        first-step velocities. Rows with no previous positions get NaN.
        """
        positions = trajectory_positions(candidates)[..., 1:, :]  # [B, K, T, 2]
        overlap = min(positions.shape[-2], previous.shape[-2])
        reference = previous[:, None, :overlap, :].to(positions.dtype)
        present = torch.isfinite(reference).all(dim=-1)  # [B, 1, M]
        gap = torch.linalg.vector_norm(positions[..., :overlap, :] - torch.nan_to_num(reference), dim=-1)
        count = present.sum(dim=-1)
        mean_gap = (gap * present).sum(dim=-1) / count.clamp_min(1)
        first = positions[..., 0, :] * CONTROL_HZ  # velocity of the first step (origin is now)
        before = torch.nan_to_num(previous[:, None, 0, :].to(positions.dtype)) * CONTROL_HZ
        velocity_gap = torch.linalg.vector_norm(first - before, dim=-1)
        cost = mean_gap + CONSISTENCY_VELOCITY_WEIGHT * velocity_gap
        return cost.masked_fill(count == 0, float("nan"))

    def spring_fallback(self, batch: dict[str, torch.Tensor], row: int) -> np.ndarray:
        polyline, valid = path_polyline(batch)
        vertices = polyline[row][valid[row]].cpu().numpy()
        count = int(batch["path_valid"][row].sum())
        ends_at_goal = count == 0 or bool(batch["path"][row, count - 1, 3] > 0.5)
        speed = self.stats.gait_speed[int(batch["gait"][row])]  # float64, exactly as before
        target = batch.get("target_speed")
        if self.model.speed_conditioning and target is not None:
            value = float(target.reshape(-1)[row])
            if np.isfinite(value):
                speed = value
        return spring_follow(
            vertices, ends_at_goal, initial_velocity(batch)[row].cpu().numpy(), speed,
            steps=self.model.horizon_steps,
        )

    @torch.no_grad()
    def plan_batch(
        self, batch: dict[str, torch.Tensor], generator: torch.Generator | None = None,
        half_width: float | None = None, candidates: int | None = None,
        previous: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample, filter and select one plan per row.

        ``previous`` ([B, M, 2], current local frame, NaN where absent) holds the positions
        the previous plan still had to visit at steps 1..M from now; only the "consistent"
        rule reads it.
        """
        batch = {key: value.to(self.device) for key, value in batch.items()}
        speed_clipped = None
        if self.model.speed_conditioning and "target_speed" in batch:
            batch, speed_clipped = self.clip_target_speed(batch)
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
            if self.config.selection in ("goal_aware", "consistent"):
                near, goal = goal_within_reach(batch, self.reference_speed(batch), candidates.shape[-2])
                endpoint = trajectory_positions(candidates)[..., -1, :]
                miss = torch.linalg.vector_norm(endpoint - goal[:, None, :], dim=-1)
                final_speed = torch.linalg.vector_norm(
                    candidates[..., -3:, :2], dim=-1).mean(dim=-1) * CONTROL_HZ
                cost = (miss + GOAL_AWARE_SPEED_WEIGHT * final_speed).masked_fill(invalid, float("inf"))
                # All-invalid rows keep first_valid's index; the fallback replaces them anyway.
                choice = torch.where(near & (~invalid).any(dim=1), cost.argmin(dim=1), choice)
                if self.config.selection == "consistent" and previous is not None:
                    consistent = self.consistency_cost(candidates, previous.to(self.device))
                    has_previous = torch.isfinite(consistent).any(dim=1)
                    consistent = torch.nan_to_num(consistent, nan=float("inf")).masked_fill(invalid, float("inf"))
                    use = has_previous & ~near & (~invalid).any(dim=1)
                    choice = torch.where(use, consistent.argmin(dim=1), choice)
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
        result = {
            "deltas": deltas, "source": source, "candidates": candidates,
            "candidate_invalid": invalid,
        }
        if speed_clipped is not None:
            result["target_speed"] = batch["target_speed"]  # as used, after clamping; NaN = unspecified
            result["target_speed_clipped"] = speed_clipped
        return result

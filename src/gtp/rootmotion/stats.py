"""Dataset statistics carried inside every checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from gtp.rootmotion import CONTROL_HZ, GAITS


SPEED_RANGE_PERCENTILES = (1.0, 99.0)


@dataclass(frozen=True)
class RootMotionStats:
    delta_std: float
    turn_std: float
    gait_speed: tuple[float, ...]  # median moving speed per gait, m/s
    # Per gait (low, high) of the training target_speed labels (1st-99th percentile of the
    # moving windows); None in checkpoints trained before target speeds existed.
    speed_range: tuple[tuple[float, float], ...] | None = None

    def clip_speed(self, speed: float, gait: int) -> float:
        """A requested target speed clamped into the training range of its gait."""
        if self.speed_range is None:
            return float(speed)
        low, high = self.speed_range[gait]
        return float(min(max(speed, low), high))

    def normalise(self, target: torch.Tensor) -> torch.Tensor:
        scale = target.new_tensor([self.delta_std, self.delta_std, self.turn_std])
        return target / scale

    def denormalise(self, target: torch.Tensor) -> torch.Tensor:
        scale = target.new_tensor([self.delta_std, self.delta_std, self.turn_std])
        return target * scale

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "RootMotionStats":
        ranges = values.get("speed_range")
        return cls(
            float(values["delta_std"]), float(values["turn_std"]),
            tuple(float(v) for v in values["gait_speed"]),
            None if ranges is None else tuple((float(lo), float(hi)) for lo, hi in ranges),
        )

    @classmethod
    def from_batch(cls, batch: dict[str, torch.Tensor]) -> "RootMotionStats":
        target = batch["target"]
        speed = torch.linalg.vector_norm(target[..., :2], dim=-1) * CONTROL_HZ
        medians = []
        for index in range(len(GAITS)):
            moving = speed[batch["gait"] == index]
            moving = moving[moving > 0.5]
            medians.append(float(moving.median()) if moving.numel() else 1.3)
        ranges = None
        if "target_speed" in batch:
            labels, ranges = batch["target_speed"], []
            low_q, high_q = (p / 100.0 for p in SPEED_RANGE_PERCENTILES)
            for index in range(len(GAITS)):
                values = labels[(batch["gait"] == index) & (labels > 0)]
                ranges.append((float(torch.quantile(values, low_q)), float(torch.quantile(values, high_q)))
                              if values.numel() else (0.0, 0.0))
            ranges = tuple(ranges)
        return cls(float(target[..., :2].std()), float(target[..., 2].std()), tuple(medians), ranges)

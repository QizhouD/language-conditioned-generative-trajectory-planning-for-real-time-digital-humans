"""Dataset statistics carried inside every checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from gtp.rootmotion import CONTROL_HZ, GAITS


@dataclass(frozen=True)
class RootMotionStats:
    delta_std: float
    turn_std: float
    gait_speed: tuple[float, ...]  # median moving speed per gait, m/s

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
        return cls(
            float(values["delta_std"]), float(values["turn_std"]),
            tuple(float(v) for v in values["gait_speed"]),
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
        return cls(float(target[..., :2].std()), float(target[..., 2].std()), tuple(medians))

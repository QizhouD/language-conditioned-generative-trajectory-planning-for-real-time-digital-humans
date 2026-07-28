"""Evaluate trajectory quality, safety, diversity, and inference latency."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gtp.dataset import DigitalHumanTrajectoryDataset, collate_trajectories
from gtp.models import ConditionalTrajectoryFlow
from gtp.planning.constraints import collision_mask, cumulative_positions
from gtp.train import build_model


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[min(len(values) - 1, round((len(values) - 1) * quantile))]


def _batch_without_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in batch.items() if key not in {"trajectory", "actions"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--latency-repetitions", type=int, default=30)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = DigitalHumanTrajectoryDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_trajectories)
    totals = {"ade": 0.0, "fde": 0.0, "facing": 0.0, "collision": 0.0,
              "action": 0.0, "diversity": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = _batch_without_targets(batch)
            if isinstance(model, ConditionalTrajectoryFlow):
                output = model.sample(inputs, steps=args.flow_steps, samples=args.samples)
                shape = (batch["history"].shape[0], args.samples, *output["trajectory"].shape[-2:])
                candidates = output["trajectory"].view(shape)
                prediction = candidates[:, 0]
                logits = output["action_logits"].view(
                    batch["history"].shape[0], args.samples,
                    output["action_logits"].shape[-2], output["action_logits"].shape[-1],
                )[:, 0]
                endpoints = cumulative_positions(candidates)[..., -1, :]
                diversity = torch.pdist(endpoints.flatten(0, 1)).mean().item() if endpoints.numel() > 2 else 0.0
            else:
                output = model(inputs)
                prediction, logits, diversity = output["trajectory"], output["action_logits"], 0.0
            predicted_positions = cumulative_positions(prediction)
            target_positions = cumulative_positions(batch["trajectory"])
            distances = torch.linalg.vector_norm(predicted_positions - target_positions, dim=-1)
            predicted_yaw = torch.atan2(prediction[:, -1, 2], prediction[:, -1, 3])
            target_yaw = torch.atan2(batch["trajectory"][:, -1, 2], batch["trajectory"][:, -1, 3])
            facing = torch.atan2(
                torch.sin(predicted_yaw - target_yaw), torch.cos(predicted_yaw - target_yaw)
            ).abs()
            collisions = collision_mask(
                prediction, batch["obstacles"], batch["obstacle_valid"]
            )
            size = prediction.shape[0]
            totals["ade"] += distances.mean(dim=1).sum().item()
            totals["fde"] += distances[:, -1].sum().item()
            totals["facing"] += facing.sum().item()
            totals["collision"] += collisions.float().sum().item()
            totals["action"] += (
                logits.argmax(dim=-1) == batch["actions"]
            ).float().mean(dim=1).sum().item()
            totals["diversity"] += diversity * size
            count += size

    batch = _batch_without_targets(next(iter(loader)))
    timings: list[float] = []
    for _ in range(args.latency_repetitions):
        started = time.perf_counter()
        if isinstance(model, ConditionalTrajectoryFlow):
            model.sample(batch, steps=args.flow_steps, samples=args.samples)
        else:
            model(batch)
        timings.append((time.perf_counter() - started) * 1_000)
    print(
        {
            "ade": totals["ade"] / count,
            "fde": totals["fde"] / count,
            "final_facing_degrees": math.degrees(totals["facing"] / count),
            "collision_rate": totals["collision"] / count,
            "action_accuracy": totals["action"] / count,
            "endpoint_diversity": totals["diversity"] / count,
            "latency_ms_p50": percentile(timings, 0.5),
            "latency_ms_p95": percentile(timings, 0.95),
        }
    )


if __name__ == "__main__":
    main()

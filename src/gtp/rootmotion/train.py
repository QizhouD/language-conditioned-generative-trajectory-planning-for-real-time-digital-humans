"""Train the root-motion flow model or its deterministic counterpart."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.windows import (
    POSITION_SCALE, RootMotionWindows, WindowConfig, mirror, trajectory_positions,
)


def build_model(config: dict) -> RootMotionModel:
    return RootMotionModel(**config["model"])


def load_splits(config: dict) -> dict[str, RootMotionWindows]:
    tracks = dict(np.load(config["data_path"]))
    data = config["data"]
    train_config = WindowConfig(stride=data.get("train_stride", 1), seed=config["seed"])
    eval_config = WindowConfig(stride=data.get("eval_stride", 5), seed=config["seed"])
    epsilon = data.get("eval_epsilon", 0.25)
    return {
        "train": RootMotionWindows(tracks, set(data["train_subjects"]), train_config),
        "val": RootMotionWindows(tracks, set(data["val_subjects"]), eval_config, epsilon),
        "test": RootMotionWindows(tracks, set(data["test_subjects"]), eval_config, epsilon),
    }


SPEED_DROPOUT = 0.1


def drop_target_speed(batch: dict[str, torch.Tensor], probability: float,
                      generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Replace each row's target speed by NaN ("unspecified") with the given probability."""
    speed = batch["target_speed"]
    drop = torch.rand(speed.shape, generator=generator, device=speed.device) < probability
    return {**batch, "target_speed": torch.where(drop, torch.full_like(speed, float("nan")), speed)}


def position_loss_weight(config: dict) -> float:
    """training.position_loss_weight; 0 (the default) leaves training exactly as before."""
    return float(config.get("training", {}).get("position_loss_weight", 0.0))


def position_loss(
    normalised_prediction: torch.Tensor, target: torch.Tensor, stats: RootMotionStats,
) -> torch.Tensor:
    """MSE between predicted and true positions, both accumulated from the per-step deltas.

    Positions are divided by POSITION_SCALE so the term is O(1) like the delta loss.
    Per-step errors that the delta loss treats as independent add up along the plan
    here, which is what the stop error and ADE measure.
    """
    predicted = trajectory_positions(stats.denormalise(normalised_prediction))[..., 1:, :]
    actual = trajectory_positions(target)[..., 1:, :]
    return nn.functional.mse_loss(predicted / POSITION_SCALE, actual / POSITION_SCALE)


def batch_loss(
    model: RootMotionModel, batch: dict[str, torch.Tensor], stats: RootMotionStats,
    generator: torch.Generator | None = None, position_weight: float = 0.0,
    parts: dict[str, float] | None = None,
) -> torch.Tensor:
    """Training loss. With position_weight > 0 a position-space term is added:

    * regressor: on the prediction itself;
    * flow: on the clean sample implied by the predicted velocity,
      x1_hat = state + (1 - t) * v_pred.

    ``parts``, when given, receives the two terms separately for logging.
    """
    target = stats.normalise(batch["target"])
    if model.deterministic:
        estimate = model.predict(batch)
        loss = nn.functional.mse_loss(estimate, target)
    else:
        noise = torch.randn(target.shape, generator=generator, device=target.device)
        time = torch.rand(target.shape[0], generator=generator, device=target.device)
        mix = time[:, None, None]
        state = (1.0 - mix) * noise + mix * target
        velocity = model.velocity(state, time, batch)
        loss = nn.functional.mse_loss(velocity, target - noise)
        estimate = None if position_weight <= 0 else state + (1.0 - mix) * velocity
    if parts is not None:
        parts["base"] = parts.get("base", 0.0) + float(loss.detach())
    if position_weight > 0:
        # Skipped entirely at weight 0, so existing configs train bit-for-bit as before.
        term = position_loss(estimate, batch["target"], stats)
        if parts is not None:
            parts["position"] = parts.get("position", 0.0) + float(term.detach())
        loss = loss + position_weight * term
    return loss


@torch.no_grad()
def validation_loss(
    model: RootMotionModel, data: RootMotionWindows, stats: RootMotionStats,
    batch_size: int, device: torch.device, repeats: int = 4, position_weight: float = 0.0,
    parts: dict[str, float] | None = None,
) -> float:
    """Mean total loss (including the weighted position term) over the data.

    ``parts``, when given, receives the size-weighted means of the two terms.
    """
    generator = torch.Generator(device=device).manual_seed(1234)
    total, count = 0.0, 0
    sums: dict[str, float] = {}
    for _ in range(1 if model.deterministic else repeats):
        for start in range(0, len(data), batch_size):
            batch = {k: v[start : start + batch_size].to(device) for k, v in data.as_batch().items()}
            size = batch["target"].shape[0]
            step_parts: dict[str, float] = {}
            total += batch_loss(model, batch, stats, generator, position_weight, step_parts).item() * size
            for key, value in step_parts.items():
                sums[key] = sums.get(key, 0.0) + value * size
            count += size
    if parts is not None:
        parts.update({key: value / count for key, value in sums.items()})
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(config)
    train = splits["train"]
    stats = RootMotionStats.from_batch(train.as_batch())
    print(f"windows: train={len(train)} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"stats: {json.dumps(stats.to_dict())}")

    model = build_model(config).to(device)
    ema = copy.deepcopy(model).eval()
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training.get("weight_decay", 0.0)
    )
    batch_size, epochs = training["batch_size"], training["epochs"]
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps, warmup = steps_per_epoch * epochs, training.get("warmup_steps", 200)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup)
        * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / total_steps))),
    )
    decay = training.get("ema_decay", 0.999)
    weight = position_loss_weight(config)
    if weight > 0:
        print(f"position_loss_weight={weight}", flush=True)
    speed_dropout = float(training.get("speed_dropout", SPEED_DROPOUT))
    if model.speed_conditioning:
        print(f"speed_conditioning: dropout={speed_dropout} range={stats.speed_range}", flush=True)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.to(device) for key, value in train.as_batch().items()}
    best, started = float("inf"), time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), device=device)
        for start in range(0, len(train), batch_size):
            index = order[start : start + batch_size]
            batch = {key: value[index] for key, value in tensors.items()}
            batch = mirror(batch, torch.rand(len(index), device=device) < 0.5)
            if model.speed_conditioning:
                # Only conditioned models draw this, so other configs train exactly as before.
                batch = drop_target_speed(batch, speed_dropout)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, stats, position_weight=weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters()):
                    averaged.lerp_(current, 1.0 - decay)
        if epoch % training.get("validate_every", 5) == 0 or epoch == epochs:
            parts: dict[str, float] = {}
            value = validation_loss(ema, splits["val"], stats, batch_size, device,
                                    position_weight=weight, parts=parts)
            # best.pt is chosen on this total, which includes the weighted position term.
            detail = (f" base={parts['base']:.5f} position={parts['position']:.5f}"
                      if weight > 0 else "")
            print(f"epoch={epoch:03d} val_loss={value:.5f}{detail} elapsed={time.time() - started:.0f}s",
                  flush=True)
            if value < best:
                best = value
                torch.save(
                    {"model": ema.state_dict(), "config": config, "stats": stats.to_dict(),
                     "validation_loss": value, "epoch": epoch},
                    run_dir / "best.pt",
                )


if __name__ == "__main__":
    main()

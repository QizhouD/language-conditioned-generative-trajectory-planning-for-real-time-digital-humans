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
from gtp.rootmotion.windows import RootMotionWindows, WindowConfig, mirror


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


def batch_loss(
    model: RootMotionModel, batch: dict[str, torch.Tensor], stats: RootMotionStats,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    target = stats.normalise(batch["target"])
    if model.deterministic:
        return nn.functional.mse_loss(model.predict(batch), target)
    noise = torch.randn(target.shape, generator=generator, device=target.device)
    time = torch.rand(target.shape[0], generator=generator, device=target.device)
    mix = time[:, None, None]
    state = (1.0 - mix) * noise + mix * target
    return nn.functional.mse_loss(model.velocity(state, time, batch), target - noise)


@torch.no_grad()
def validation_loss(
    model: RootMotionModel, data: RootMotionWindows, stats: RootMotionStats,
    batch_size: int, device: torch.device, repeats: int = 4,
) -> float:
    generator = torch.Generator(device=device).manual_seed(1234)
    total, count = 0.0, 0
    for _ in range(1 if model.deterministic else repeats):
        for start in range(0, len(data), batch_size):
            batch = {k: v[start : start + batch_size].to(device) for k, v in data.as_batch().items()}
            size = batch["target"].shape[0]
            total += batch_loss(model, batch, stats, generator).item() * size
            count += size
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
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, stats)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters()):
                    averaged.lerp_(current, 1.0 - decay)
        if epoch % training.get("validate_every", 5) == 0 or epoch == epochs:
            value = validation_loss(ema, splits["val"], stats, batch_size, device)
            print(f"epoch={epoch:03d} val_loss={value:.5f} elapsed={time.time() - started:.0f}s", flush=True)
            if value < best:
                best = value
                torch.save(
                    {"model": ema.state_dict(), "config": config, "stats": stats.to_dict(),
                     "validation_loss": value, "epoch": epoch},
                    run_dir / "best.pt",
                )


if __name__ == "__main__":
    main()

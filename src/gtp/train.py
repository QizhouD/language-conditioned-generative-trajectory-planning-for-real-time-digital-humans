"""Training entry point for digital-human trajectory models."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split

from gtp.dataset import DigitalHumanTrajectoryDataset, collate_trajectories
from gtp.models import ConditionalTrajectoryFlow, DeterministicTrajectoryBaseline


def build_model(config: dict) -> nn.Module:
    model = config["model"]
    arguments = {
        key: model[key]
        for key in (
            "state_dim", "goal_dim", "obstacle_dim", "trajectory_dim", "hidden_dim",
            "layers", "heads", "horizon_steps",
        )
    }
    if model["type"] == "deterministic_trajectory_transformer":
        return DeterministicTrajectoryBaseline(**arguments)
    if model["type"] == "conditional_trajectory_flow":
        return ConditionalTrajectoryFlow(**arguments)
    raise ValueError(f"unsupported model type: {model['type']}")


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def batch_loss(
    model: nn.Module, batch: dict[str, torch.Tensor], weights: dict[str, float]
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["trajectory"]
    supervise_2d = bool(weights.get("supervise_2d", False))
    if isinstance(model, DeterministicTrajectoryBaseline):
        output = model(batch)
        prediction = output["trajectory"]
        if supervise_2d:
            flow_loss = nn.functional.mse_loss(prediction[..., :2], target[..., :2])
        else:
            flow_loss = nn.functional.mse_loss(prediction, target)
    else:
        noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], device=target.device)
        noisy = (1.0 - time[:, None, None]) * noise + time[:, None, None] * target
        velocity = model.velocity(noisy, time, batch)
        flow_loss = nn.functional.mse_loss(velocity, target - noise)
        prediction = noisy + (1.0 - time[:, None, None]) * velocity
        output = {"action_logits": model.action_logits(target, batch)}

    action_loss = nn.functional.cross_entropy(
        output["action_logits"].flatten(end_dim=1), batch["actions"].flatten()
    )
    predicted_end = prediction[..., :2].sum(dim=1)
    target_end = target[..., :2].sum(dim=1)
    goal_loss = nn.functional.mse_loss(predicted_end, target_end)
    if supervise_2d:
        predicted_path = prediction[..., :2].cumsum(dim=1)
        if str(weights.get("smoothness_kind", "first_diff")) == "second_diff":
            # Discrete curvature: penalize left-right jitter, not step length.
            pos = predicted_path
            curve = (pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]).square().mean(dim=(1, 2))
            empty_boost = float(weights.get("smoothness_empty_boost", 1) or 1)
            if empty_boost != 1.0:
                valid = batch["obstacle_valid"]
                center_norm = torch.linalg.vector_norm(batch["obstacles"][..., :2], dim=-1)
                real = valid & (center_norm <= 1e5)
                empty = ~real.any(dim=1)
                w = torch.where(empty, curve.new_full((), empty_boost), curve.new_ones(()))
                smoothness_loss = (curve * w).mean()
            else:
                smoothness_loss = curve.mean()
        else:
            smoothness_loss = (
                (predicted_path[:, 1:] - predicted_path[:, :-1]).square().mean()
            )
    else:
        predicted_velocity = prediction[..., 4:6]
        smoothness_loss = (
            (predicted_velocity[:, 1:] - predicted_velocity[:, :-1]).square().mean()
        )
    total = (
        weights["flow"] * flow_loss
        + weights["action"] * action_loss
        + weights["goal"] * goal_loss
        + weights["smoothness"] * smoothness_loss
    )
    collision_weight = float(weights.get("collision", 0) or 0)
    if collision_weight > 0:
        pos = prediction[..., :2].cumsum(dim=1)
        centers = batch["obstacles"][..., :2]
        clearance = float(weights.get("collision_clearance", 0.3))
        radii = batch["obstacles"][..., 2] + clearance
        valid = batch["obstacle_valid"]
        dist = torch.linalg.vector_norm(
            pos.unsqueeze(2) - centers.unsqueeze(1), dim=-1
        )
        valid_f = valid.to(dtype=dist.dtype)
        if str(weights.get("collision_kind", "overlap")) == "clearance":
            raw_r = batch["obstacles"][..., 2]
            hard = torch.relu(raw_r.unsqueeze(1) - dist)
            band = torch.relu(radii.unsqueeze(1) - dist)
            masked = (hard.square() + 0.5 * band.square()) * valid_f.unsqueeze(1)
        else:
            penetration = torch.relu(radii.unsqueeze(1) - dist)
            masked = penetration.square() * valid_f.unsqueeze(1)
        denom = valid_f.sum() * pos.shape[1]
        collision_loss = masked.sum() / denom.clamp(min=1.0)
        total = total + collision_weight * collision_loss
    metrics = {
        "loss": total.item(),
        "flow": flow_loss.item(),
        "action": action_loss.item(),
        "goal": goal_loss.item(),
        "smoothness": smoothness_loss.item(),
    }
    if collision_weight > 0:
        metrics["collision"] = collision_loss.item()
    path_weight = float(weights.get("path", 0) or 0)
    if path_weight > 0:
        pred_path = prediction[..., :2].cumsum(dim=1)
        gt_path = target[..., :2].cumsum(dim=1)
        per_sample = (pred_path - gt_path).square().mean(dim=(1, 2))
        valid = batch["obstacle_valid"]
        center_norm = torch.linalg.vector_norm(batch["obstacles"][..., :2], dim=-1)
        real = valid & (center_norm <= 1e5)
        empty = ~real.any(dim=1)
        occupied = real.any(dim=1)
        if bool(weights.get("path_empty_only", False)) and not bool(weights.get("path_occupied", 0)):
            path_loss = per_sample[empty].mean() if empty.any() else per_sample.new_zeros(())
        elif float(weights.get("path_occupied", 0) or 0) > 0:
            empty_term = per_sample[empty].mean() if empty.any() else per_sample.new_zeros(())
            occ_term = per_sample[occupied].mean() if occupied.any() else per_sample.new_zeros(())
            occ_w = float(weights.get("path_occupied", 0) or 0)
            path_loss = empty_term + occ_w * occ_term
        else:
            path_loss = per_sample.mean()
        total = total + path_weight * path_loss
        metrics["loss"] = total.item()
        metrics["path"] = path_loss.item()
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = DigitalHumanTrajectoryDataset(config["data_path"])
    validation_size = max(1, round(len(dataset) * config["training"]["validation_fraction"]))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("dataset needs at least two examples")
    train_set, validation_set = random_split(
        dataset, [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    loader_args = {
        "batch_size": config["training"]["batch_size"],
        "collate_fn": collate_trajectories,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    weights = config["loss"]
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch_loss(model, batch, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = sum(
                batch_loss(model, _move(batch, device), weights)[0].item()
                for batch in validation_loader
            ) / len(validation_loader)
        print(f"epoch={epoch:03d} validation_loss={validation:.5f}")
        if validation < best_validation:
            best_validation = validation
            torch.save(
                {"model": model.state_dict(), "config": config,
                 "validation_loss": validation},
                run_dir / "best.pt",
            )


if __name__ == "__main__":
    main()

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
    supervise_2d = weights.get("supervise_2d", False)
    
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
    
    smoothness_kind = weights.get("smoothness_kind", "first_diff")
    smoothness_empty_boost = weights.get("smoothness_empty_boost", 1.0)
    
    if supervise_2d and smoothness_kind == "second_diff":
        cumsum_pred = prediction[..., :2].cumsum(dim=1)
        second_diff = cumsum_pred[:, 2:] - 2 * cumsum_pred[:, 1:-1] + cumsum_pred[:, :-2]
        smoothness_loss = second_diff.square().mean()
        
        obstacle_centers = batch["obstacles"][..., :2]
        center_norm = obstacle_centers.norm(dim=-1)
        is_empty = (center_norm > 1e5).all(dim=-1)
        if is_empty.any() and smoothness_empty_boost != 1.0:
            empty_smooth = second_diff[is_empty].square().mean() if is_empty.sum() > 0 else 0.0
            occupied_smooth = second_diff[~is_empty].square().mean() if (~is_empty).sum() > 0 else 0.0
            empty_count = is_empty.sum().float()
            occupied_count = (~is_empty).sum().float()
            total_count = empty_count + occupied_count
            smoothness_loss = (
                (smoothness_empty_boost * empty_count * empty_smooth + occupied_count * occupied_smooth) 
                / total_count if total_count > 0 else smoothness_loss
            )
    elif supervise_2d:
        cumsum_pred = prediction[..., :2].cumsum(dim=1)
        first_diff = cumsum_pred[:, 1:] - cumsum_pred[:, :-1]
        smoothness_loss = (first_diff[:, 1:] - first_diff[:, :-1]).square().mean()
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
    metrics = {
        "loss": total.item(),
        "flow": flow_loss.item(),
        "action": action_loss.item(),
        "goal": goal_loss.item(),
        "smoothness": smoothness_loss.item(),
    }
    
    if "collision" in weights and weights["collision"] > 0:
        collision_loss = compute_collision_loss(
            prediction, batch, weights
        )
        total += weights["collision"] * collision_loss
        metrics["collision"] = collision_loss.item()
    
    if "path" in weights and weights["path"] > 0:
        path_loss = compute_path_loss(
            prediction, target, batch, weights
        )
        total += weights["path"] * path_loss
        metrics["path"] = path_loss.item()
    
    return total, metrics


def compute_collision_loss(
    prediction: torch.Tensor, batch: dict[str, torch.Tensor], weights: dict[str, float]
) -> torch.Tensor:
    cumsum_pred = prediction[..., :2].cumsum(dim=1)
    
    obstacles = batch["obstacles"]
    obstacle_centers = obstacles[..., :2]
    obstacle_radii = obstacles[..., 2]
    obstacle_valid = batch["obstacle_valid"]
    
    collision_clearance = weights.get("collision_clearance", 0.3)
    collision_kind = weights.get("collision_kind", "overlap")
    
    distances = torch.cdist(cumsum_pred, obstacle_centers)
    effective_radius = obstacle_radii.unsqueeze(1) + collision_clearance
    
    if collision_kind == "clearance":
        hard_penetration = torch.relu(obstacle_radii.unsqueeze(1) - distances)
        band_penetration = torch.relu(effective_radius - distances)
        penetration = hard_penetration.square() + 0.5 * band_penetration.square()
    else:
        penetration = torch.relu(effective_radius - distances).square()
    
    masked_penetration = penetration * obstacle_valid.unsqueeze(1).float()
    collision_loss = masked_penetration.sum() / (obstacle_valid.sum().float() * cumsum_pred.shape[1] + 1e-8)
    
    return collision_loss


def compute_path_loss(
    prediction: torch.Tensor, target: torch.Tensor, 
    batch: dict[str, torch.Tensor], weights: dict[str, float]
) -> torch.Tensor:
    cumsum_pred = prediction[..., :2].cumsum(dim=1)
    cumsum_target = target[..., :2].cumsum(dim=1)
    
    path_empty_only = weights.get("path_empty_only", False)
    path_occupied = weights.get("path_occupied", 0.0)
    
    obstacle_centers = batch["obstacles"][..., :2]
    center_norm = obstacle_centers.norm(dim=-1)
    is_empty = (center_norm > 1e5).all(dim=-1)
    
    path_mse = nn.functional.mse_loss(cumsum_pred, cumsum_target, reduction="none").mean(dim=(1, 2))
    
    if path_empty_only:
        if is_empty.any():
            path_loss = path_mse[is_empty].mean()
        else:
            path_loss = torch.tensor(0.0, device=prediction.device)
    elif path_occupied > 0:
        empty_loss = path_mse[is_empty].mean() if is_empty.any() else torch.tensor(0.0, device=prediction.device)
        occupied_loss = path_mse[~is_empty].mean() if (~is_empty).any() else torch.tensor(0.0, device=prediction.device)
        
        empty_count = is_empty.sum().float()
        occupied_count = (~is_empty).sum().float()
        total_count = empty_count + occupied_count
        
        if total_count > 0:
            path_loss = (empty_count * empty_loss + path_occupied * occupied_count * occupied_loss) / total_count
        else:
            path_loss = path_mse.mean()
    else:
        path_loss = path_mse.mean()
    
    return path_loss


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

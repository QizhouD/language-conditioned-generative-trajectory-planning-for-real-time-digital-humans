"""Evaluate root-motion planners on held-out subjects.

Every method sees the same history, path and gait and is scored on the same
windows. ADE/FDE compare positions at matching timestamps, so they measure the
speed profile as well as the shape. Distribution distances compare pooled
speed, acceleration and turn-rate against the capture.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from gtp.rootmotion import CONTROL_HZ, GAITS
from gtp.rootmotion.baselines import constant_speed_follow, spring_follow
from gtp.rootmotion.constraints import corridor_deviation, kinematic_violation
from gtp.rootmotion.planner import (
    SELECTION_RULES, SOURCE_HOLD, SOURCE_MODEL, PlannerConfig, RootMotionPlanner, initial_velocity,
    load_checkpoint,
)
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.train import load_splits
from gtp.rootmotion.windows import path_polyline, trajectory_positions


def wasserstein(left: torch.Tensor, right: torch.Tensor, points: int = 1_000) -> float:
    """1-D Wasserstein-1 distance from matched quantiles."""
    grid = torch.linspace(0.0, 1.0, points)
    return float((torch.quantile(left.flatten()[:1_000_000], grid)
                  - torch.quantile(right.flatten()[:1_000_000], grid)).abs().mean())


def kinematics(deltas: torch.Tensor) -> dict[str, torch.Tensor]:
    speed = torch.linalg.vector_norm(deltas[..., :2], dim=-1) * CONTROL_HZ
    return {
        "speed": speed,
        "acceleration": (speed[..., 1:] - speed[..., :-1]) * CONTROL_HZ,
        "turn_rate": deltas[..., 2] * CONTROL_HZ,
    }


def displacement(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-step position error: [..., T]."""
    return torch.linalg.vector_norm(
        trajectory_positions(prediction)[..., 1:, :] - trajectory_positions(target)[..., 1:, :], dim=-1
    )


def goal_rows(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Windows whose capture comes to rest on the path's goal inside the horizon."""
    polyline, _ = path_polyline(batch)
    count = batch["path_valid"].sum(dim=1)
    rows = torch.arange(len(count))
    ends = (count > 0) & (batch["path"][rows, (count - 1).clamp_min(0), 3] > 0.5)
    final = trajectory_positions(batch["target"])[:, -1]
    return ends & (torch.linalg.vector_norm(final - polyline[rows, count], dim=-1) < 0.3)


def score(
    name: str, deltas: torch.Tensor, batch: dict[str, torch.Tensor], half_width: float,
) -> dict[str, float]:
    target = batch["target"]
    error = displacement(deltas, target)
    polyline, valid = path_polyline(batch)
    bound = corridor_deviation(deltas, polyline, valid)
    reference, produced = kinematics(target), kinematics(deltas)
    result = {
        "method": name,
        "ade": float(error.mean()),
        "fde": float(error[:, -1].mean()),
        "corridor_violation": float((bound > half_width).float().mean()),
        "kinematic_violation": float(
            kinematic_violation(deltas, initial_velocity=initial_velocity(batch)).float().mean()
        ),
        "w1_speed": wasserstein(produced["speed"], reference["speed"]),
        "w1_acceleration": wasserstein(produced["acceleration"], reference["acceleration"]),
        "w1_turn_rate": wasserstein(produced["turn_rate"], reference["turn_rate"]),
    }
    arrived = goal_rows(batch)
    if arrived.any():
        rows = arrived.nonzero(as_tuple=True)[0]
        count = batch["path_valid"].sum(dim=1)[rows]
        goal = polyline[rows, count]
        final = trajectory_positions(deltas)[rows, -1]
        result["stop_error"] = float(torch.linalg.vector_norm(final - goal, dim=-1).mean())
        result["stop_speed"] = float(produced["speed"][rows, -3:].mean())
    for index, gait in enumerate(GAITS):
        mask = batch["gait"] == index
        if mask.any():
            result[f"ade_{gait}"] = float(error[mask].mean())
    return result


def run_baseline(kind: str, batch: dict[str, torch.Tensor], gait_speed: tuple[float, ...],
                 per_window_speed: bool = False) -> torch.Tensor:
    """Spring or constant-speed follower on every window.

    ``per_window_speed`` gives each window its own target_speed label, the information a
    speed-conditioned model receives; otherwise the gait's median moving speed is used.
    """
    polyline, valid = path_polyline(batch)
    velocity = initial_velocity(batch).numpy()
    count = batch["path_valid"].sum(dim=1)
    rows = []
    for row in range(len(count)):
        vertices = polyline[row][valid[row]].numpy()
        ends = int(count[row]) == 0 or bool(batch["path"][row, int(count[row]) - 1, 3] > 0.5)
        speed = (float(batch["target_speed"][row]) if per_window_speed
                 else gait_speed[int(batch["gait"][row])])
        if kind == "spring":
            rows.append(spring_follow(vertices, ends, velocity[row], speed))
        else:
            rows.append(constant_speed_follow(vertices, ends, float(np.hypot(*velocity[row])), speed))
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


def latency(planner: RootMotionPlanner, batch: dict[str, torch.Tensor], repeats: int) -> dict[str, float]:
    single = {key: value[:1] for key, value in batch.items() if key != "target"}
    timings = []
    for _ in range(repeats + 3):
        started = time.perf_counter()
        planner.plan_batch(single)
        timings.append((time.perf_counter() - started) * 1_000.0)
    timings = sorted(timings[3:])  # drop warm-up calls
    return {
        "latency_ms_p50": timings[len(timings) // 2],
        "latency_ms_p95": timings[min(len(timings) - 1, round(0.95 * (len(timings) - 1)))],
    }


def evaluate_model(
    name: str, checkpoint: Path, batch: dict[str, torch.Tensor], config: PlannerConfig,
    seed: int, repeats: int, chunk: int = 256,
) -> dict[str, float]:
    model, stats, _ = load_checkpoint(checkpoint)
    planner = RootMotionPlanner(model, stats, config)
    generator = torch.Generator().manual_seed(seed)
    parts: dict[str, list[torch.Tensor]] = {"deltas": [], "source": [], "candidates": [], "candidate_invalid": []}
    inputs = {key: value for key, value in batch.items() if key != "target"}
    for start in range(0, len(batch["target"]), chunk):
        piece = {key: value[start : start + chunk] for key, value in inputs.items()}
        result = planner.plan_batch(piece, generator)
        for key in parts:
            parts[key].append(result[key])
    merged = {key: torch.cat(value) for key, value in parts.items()}
    result = score(name, merged["deltas"], batch, config.half_width)
    candidate_error = displacement(merged["candidates"], batch["target"].unsqueeze(1))
    result["min_ade"] = float(candidate_error.mean(-1).amin(1).mean())
    result["candidate_rejection"] = float(merged["candidate_invalid"].float().mean())
    result["fallback_rate"] = float((merged["source"] != SOURCE_MODEL).float().mean())
    result["hold_rate"] = float((merged["source"] == SOURCE_HOLD).float().mean())
    endpoints = trajectory_positions(merged["candidates"])[:, :, -1]
    if endpoints.shape[1] > 1:
        spread = torch.cdist(endpoints, endpoints).sum(dim=(1, 2))
        result["endpoint_diversity"] = float(
            (spread / (endpoints.shape[1] * (endpoints.shape[1] - 1))).mean()
        )
    result.update(latency(planner, batch, repeats))
    return result


COLUMNS = (
    ("method", "method"), ("ade", "ADE m"), ("fde", "FDE m"), ("min_ade", "minADE m"),
    ("w1_speed", "W1 speed"), ("w1_acceleration", "W1 accel"), ("w1_turn_rate", "W1 turn"),
    ("stop_error", "stop err m"), ("corridor_violation", "corridor viol."),
    ("fallback_rate", "fallback"), ("latency_ms_p50", "P50 ms"), ("latency_ms_p95", "P95 ms"),
)


def markdown(rows: list[dict[str, float]]) -> str:
    lines = ["| " + " | ".join(title for _, title in COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]
    for row in rows:
        cells = []
        for key, _ in COLUMNS:
            value = row.get(key)
            if value is None:
                cells.append("–")
            elif isinstance(value, str):
                cells.append(value)
            elif key.startswith("latency"):
                cells.append(f"{value:.0f}")
            elif key in {"corridor_violation", "fallback_rate"}:
                cells.append(f"{100 * value:.1f}%")
            else:
                cells.append(f"{value:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="training config (for the data split)")
    parser.add_argument("--flow", type=Path, help="flow-model checkpoint")
    parser.add_argument("--regressor", type=Path, help="deterministic checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--half-width", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=SELECTION_RULES, default="goal_aware",
                        help="candidate selection; the regressor has one candidate, so only flow changes")
    parser.add_argument("--latency-repetitions", type=int, default=50)
    parser.add_argument("--no-target-speed", action="store_true",
                        help="hide the target speed from the models (a speed-conditioned model then "
                             "uses its 'unspecified' input); baselines are unaffected")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    splits = load_splits(config)
    data = splits[args.split]
    batch = data.as_batch()
    # Baselines use the training set's median speed per gait, like the planner's fallback.
    gait_speed = RootMotionStats.from_batch(splits["train"].as_batch()).gait_speed

    rows = [score("capture (reference)", batch["target"], batch, args.half_width)]
    rows.append(score("constant speed on polyline", run_baseline("constant", batch, gait_speed), batch, args.half_width))
    rows.append(score("spring follower", run_baseline("spring", batch, gait_speed), batch, args.half_width))
    # The same followers told each window's target speed, as a speed-conditioned model is.
    rows.append(score("constant speed, target speed",
                      run_baseline("constant", batch, gait_speed, per_window_speed=True), batch, args.half_width))
    rows.append(score("spring follower, target speed",
                      run_baseline("spring", batch, gait_speed, per_window_speed=True), batch, args.half_width))
    if args.no_target_speed:
        batch = {**batch, "target_speed": torch.full_like(batch["target_speed"], float("nan"))}
    if args.regressor:
        planner_config = PlannerConfig(candidates=1, half_width=args.half_width,
                                       selection=args.selection)
        rows.append(evaluate_model("regressor + filter", args.regressor, batch, planner_config,
                                   args.seed, args.latency_repetitions))
    if args.flow:
        planner_config = PlannerConfig(
            candidates=args.candidates, flow_steps=args.flow_steps, half_width=args.half_width,
            selection=args.selection,
        )
        rows.append(evaluate_model(f"flow K={args.candidates} + filter", args.flow, batch,
                                   planner_config, args.seed, args.latency_repetitions))
    report = {
        "split": args.split, "windows": len(data), "clips": data.clips,
        "gait_counts": {gait: int((batch["gait"] == index).sum()) for index, gait in enumerate(GAITS)},
        "half_width": args.half_width, "seed": args.seed, "selection": args.selection,
        "target_speed_given": not args.no_target_speed,
        "hardware": f"{platform.processor() or platform.machine()}, {torch.get_num_threads()} threads",
        "results": rows,
    }
    print(f"{args.split}: {len(data)} windows from {data.clips}")
    print(markdown(rows))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

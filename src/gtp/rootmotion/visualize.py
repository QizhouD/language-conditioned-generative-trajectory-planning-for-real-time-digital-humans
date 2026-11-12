"""Plot held-out windows: the path, the capture, model samples and the spring baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from gtp.rootmotion import GAITS
from gtp.rootmotion.evaluate import run_baseline
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.train import load_splits
from gtp.rootmotion.windows import path_polyline, trajectory_positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    splits = load_splits(config)
    data = splits[args.split]
    indices = args.indices or torch.linspace(0, len(data) - 1, 8).long().tolist()
    full = data.as_batch()
    batch = {key: value[indices] for key, value in full.items()}
    planner = RootMotionPlanner.from_checkpoint(
        args.checkpoint, PlannerConfig(candidates=args.candidates)
    )
    result = planner.plan_batch(
        {key: value for key, value in batch.items() if key != "target"},
        torch.Generator().manual_seed(args.seed),
    )
    gait_speed = RootMotionStats.from_batch(splits["train"].as_batch()).gait_speed
    spring = run_baseline("spring", batch, gait_speed)
    polyline, valid = path_polyline(batch)

    columns = min(4, len(indices))
    rows = (len(indices) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.2 * rows), squeeze=False)
    for slot, axis in enumerate(axes.flat):
        if slot >= len(indices):
            axis.axis("off")
            continue
        path = polyline[slot][valid[slot]]
        axis.plot(path[:, 0], path[:, 1], color="0.6", linewidth=6, alpha=0.5,
                  solid_capstyle="round", label="path from engine")
        candidates = trajectory_positions(result["candidates"][slot])
        rejected = result["candidate_invalid"][slot]
        for index in range(candidates.shape[0]):
            axis.plot(candidates[index, :, 0], candidates[index, :, 1],
                      color="tab:red" if rejected[index] else "tab:blue",
                      linewidth=0.7, alpha=0.45,
                      label={True: "rejected sample", False: "valid sample"}[bool(rejected[index])])
        chosen = trajectory_positions(result["deltas"][slot])
        axis.plot(chosen[:, 0], chosen[:, 1], color="tab:blue", linewidth=2.0, label="executed plan")
        baseline = trajectory_positions(spring[slot])
        axis.plot(baseline[:, 0], baseline[:, 1], color="tab:orange", linestyle="--",
                  linewidth=1.6, label="spring follower")
        capture = trajectory_positions(batch["target"][slot])
        axis.plot(capture[:, 0], capture[:, 1], color="black", linewidth=1.6, label="capture, 3 s")
        axis.scatter([0], [0], color="black", s=25, zorder=5)
        reach = float(max(capture.abs().max(), chosen.abs().max(), 1.0)) + 0.6
        axis.set_xlim(-reach, reach)
        axis.set_ylim(-reach, reach)
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
        axis.set_title(f"{args.split} window {indices[slot]} · {GAITS[int(batch['gait'][slot])]}", fontsize=10)
        axis.set_xlabel("x, m (right)")
        axis.set_ylabel("z, m (forward)")
    handles, labels = [], []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=130)
    print(args.output)


if __name__ == "__main__":
    main()

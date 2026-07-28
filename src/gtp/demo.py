"""Export one model-generated runtime plan from a checkpoint and JSONL example."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gtp.dataset import DigitalHumanTrajectoryDataset, collate_trajectories
from gtp.integrations.plan_export import build_trajectory_plan, write_plan
from gtp.planning import RollingTrajectoryPlanner
from gtp.train import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--origin-yaw", type=float, default=0.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = DigitalHumanTrajectoryDataset(args.dataset)
    batch = collate_trajectories([dataset[args.index]])
    result = RollingTrajectoryPlanner(
        model, candidates=args.candidates, flow_steps=args.flow_steps
    ).plan(batch)
    plan = build_trajectory_plan(
        result["trajectory"][0].tolist(),
        result["actions"][0].tolist(),
        origin=args.origin,
        origin_yaw=args.origin_yaw,
        score=result["cost"][0].item(),
        fallback=result["fallback"][0].item(),
    )
    write_plan(args.output, plan)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

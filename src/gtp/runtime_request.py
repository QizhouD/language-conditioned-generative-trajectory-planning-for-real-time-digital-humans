"""Plan one JSON runtime request using a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gtp.integrations.plan_export import write_plan
from gtp.runtime import RuntimePlanner
from gtp.schema import MotionState, Obstacle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    character = request["character"]
    history = [MotionState(**item) for item in request["history"]]
    obstacles = [Obstacle(**item) for item in request.get("obstacles", [])]
    planner = RuntimePlanner.from_checkpoint(
        args.checkpoint,
        candidates=args.candidates,
        flow_steps=args.flow_steps,
        device=args.device,
    )
    plan = planner.plan_function_call(
        request["function_call"],
        request.get("scene_objects", {}),
        history,
        obstacles,
        character_position=character["position"],
        character_yaw=character["yaw"],
    )
    write_plan(args.output, plan)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

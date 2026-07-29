"""JSONL dataset and padding for digital-human trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from gtp.schema import MOTION_INTENTS


class DigitalHumanTrajectoryDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                history = torch.tensor(
                    [_state_features(state) for state in row["history"]], dtype=torch.float32
                )
                goal = row["goal"]
                goal_features = torch.tensor(
                    [
                        goal["target_position"][0],
                        goal["target_position"][2],
                        math.sin(goal["target_yaw"]),
                        math.cos(goal["target_yaw"]),
                        goal["stop_distance"],
                    ],
                    dtype=torch.float32,
                )
                obstacles = torch.tensor(
                    [
                        [*item["center"], item["radius"], float(item.get("kind", 0))]
                        for item in row["obstacles"]
                    ],
                    dtype=torch.float32,
                )
                if not len(obstacles):
                    obstacles = torch.empty(0, 4, dtype=torch.float32)
                self.examples.append(
                    {
                        "history": history,
                        "goal": goal_features,
                        "behavior": torch.tensor(
                            MOTION_INTENTS.index(goal["behavior"]), dtype=torch.long
                        ),
                        "obstacles": obstacles,
                        "trajectory": torch.tensor(
                            row["future_trajectory"], dtype=torch.float32
                        ),
                        "actions": torch.tensor(row["future_actions"], dtype=torch.long),
                    }
                )
        if not self.examples:
            raise ValueError(f"no trajectory examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def _state_features(state: dict) -> list[float]:
    action = float(state["action"]) / 4.0
    return [
        *state["position"],
        state["yaw"],
        *state["velocity"],
        state["angular_velocity"],
        action,
        state["action_phase"],
        *state["foot_contacts"],
        1.0,
    ]


def collate_trajectories(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    history_length = max(item["history"].shape[0] for item in batch)
    obstacle_count = max(1, max(item["obstacles"].shape[0] for item in batch))
    horizon = batch[0]["trajectory"].shape[0]
    if any(item["trajectory"].shape[0] != horizon for item in batch):
        raise ValueError("all examples in a batch must share one trajectory horizon")

    histories = torch.zeros(len(batch), history_length, batch[0]["history"].shape[-1])
    history_valid = torch.zeros(len(batch), history_length, dtype=torch.bool)
    obstacles = torch.zeros(len(batch), obstacle_count, 4)
    obstacle_valid = torch.zeros(len(batch), obstacle_count, dtype=torch.bool)
    for index, item in enumerate(batch):
        h_count = item["history"].shape[0]
        o_count = item["obstacles"].shape[0]
        histories[index, :h_count] = item["history"]
        history_valid[index, :h_count] = True
        if o_count:
            obstacles[index, :o_count] = item["obstacles"]
            obstacle_valid[index, :o_count] = True
        else:
            # A distant valid token avoids all-masked attention and phantom collisions.
            obstacles[index, 0, :2] = 1_000_000.0
            obstacle_valid[index, 0] = True

    return {
        "history": histories,
        "history_valid": history_valid,
        "goal": torch.stack([item["goal"] for item in batch]),
        "behavior": torch.stack([item["behavior"] for item in batch]),
        "obstacles": obstacles,
        "obstacle_valid": obstacle_valid,
        "trajectory": torch.stack([item["trajectory"] for item in batch]),
        "actions": torch.stack([item["actions"] for item in batch]),
    }

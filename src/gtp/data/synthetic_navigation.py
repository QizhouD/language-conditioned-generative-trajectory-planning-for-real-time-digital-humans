"""Deterministic procedural navigation trajectories for the digital-human MVP."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from gtp.schema import MotionState, Obstacle, PlanningGoal, TrajectoryExample

CONTROL_HZ = 10
HISTORY_STEPS = 20
HORIZON_STEPS = 30


def _distance_to_segment(point: tuple[float, float], end: tuple[float, float]) -> float:
    ex, ez = end
    denominator = ex * ex + ez * ez
    if denominator == 0:
        return math.hypot(*point)
    projection = max(0.0, min(1.0, (point[0] * ex + point[1] * ez) / denominator))
    return math.hypot(point[0] - projection * ex, point[1] - projection * ez)


def _reference_points(
    goal: tuple[float, float], obstacles: list[Obstacle], steps: int
) -> list[tuple[float, float]]:
    blocking = [
        obstacle
        for obstacle in obstacles
        if _distance_to_segment(tuple(obstacle.center), goal) < obstacle.radius + 0.45
    ]
    waypoints: list[tuple[float, float]] = [(0.0, 0.0)]
    if blocking:
        obstacle = min(blocking, key=lambda item: math.hypot(*item.center))
        gx, gz = goal
        length = max(math.hypot(gx, gz), 1e-6)
        normal = (-gz / length, gx / length)
        side = 1.0 if obstacle.center[0] * normal[0] + obstacle.center[1] * normal[1] >= 0 else -1.0
        clearance = obstacle.radius + 0.65
        waypoints.append(
            (
                obstacle.center[0] + side * normal[0] * clearance,
                obstacle.center[1] + side * normal[1] * clearance,
            )
        )
    waypoints.append(goal)

    lengths = [
        math.dist(waypoints[index], waypoints[index + 1])
        for index in range(len(waypoints) - 1)
    ]
    total = sum(lengths)
    points: list[tuple[float, float]] = []
    for index in range(1, steps + 1):
        distance = total * index / steps
        traversed = 0.0
        for segment, length in enumerate(lengths):
            if distance <= traversed + length or segment == len(lengths) - 1:
                ratio = min(1.0, (distance - traversed) / max(length, 1e-6))
                start, end = waypoints[segment], waypoints[segment + 1]
                points.append(
                    (
                        start[0] + ratio * (end[0] - start[0]),
                        start[1] + ratio * (end[1] - start[1]),
                    )
                )
                break
            traversed += length
    return points


def generate_example(index: int, rng: random.Random) -> TrajectoryExample:
    distance = rng.uniform(1.8, 4.5)
    angle = rng.uniform(-math.pi * 0.75, math.pi * 0.75)
    goal_x, goal_z = distance * math.sin(angle), distance * math.cos(angle)
    obstacles: list[Obstacle] = []
    for _ in range(rng.randint(0, 3)):
        fraction = rng.uniform(0.25, 0.8)
        lateral = rng.uniform(-0.65, 0.65)
        normal = (-math.cos(angle), math.sin(angle))
        obstacles.append(
            Obstacle(
                center=[
                    goal_x * fraction + normal[0] * lateral,
                    goal_z * fraction + normal[1] * lateral,
                ],
                radius=rng.uniform(0.2, 0.45),
            )
        )

    points = _reference_points((goal_x, goal_z), obstacles, HORIZON_STEPS)
    trajectory: list[list[float]] = []
    actions: list[int] = []
    previous = (0.0, 0.0)
    previous_yaw = 0.0
    behavior = "wave" if rng.random() < 0.25 else "walk"
    for step, point in enumerate(points):
        dx, dz = point[0] - previous[0], point[1] - previous[1]
        desired_yaw = math.atan2(dx, dz) if math.hypot(dx, dz) > 1e-6 else previous_yaw
        raw_yaw_delta = math.atan2(
            math.sin(desired_yaw - previous_yaw), math.cos(desired_yaw - previous_yaw)
        )
        yaw_delta = max(-0.3, min(0.3, raw_yaw_delta))
        yaw = previous_yaw + yaw_delta
        speed = math.hypot(dx, dz) * CONTROL_HZ
        trajectory.append(
            [dx, dz, math.sin(yaw), math.cos(yaw), dx * CONTROL_HZ, dz * CONTROL_HZ,
             yaw_delta * CONTROL_HZ, speed]
        )
        if behavior == "wave" and step >= HORIZON_STEPS - 2:
            action = 4
        elif step >= HORIZON_STEPS - 3:
            action = 3
        else:
            action = 2 if abs(yaw_delta) > 0.12 else 1
        actions.append(action)
        previous, previous_yaw = point, yaw

    history = [
        MotionState(
            position=[0.0, 0.0, 0.0],
            yaw=0.0,
            velocity=[0.0, 0.0],
            angular_velocity=0.0,
            action=0,
            action_phase=step / HISTORY_STEPS,
            foot_contacts=[1.0, 1.0],
        )
        for step in range(HISTORY_STEPS)
    ]
    return TrajectoryExample(
        episode_id=f"navigation-{index:06d}",
        history=history,
        goal=PlanningGoal(
            behavior=behavior,
            target_position=[goal_x, 0.0, goal_z],
            target_yaw=angle,
            instruction=(
                "Walk to the target while avoiding nearby obstacles, then wave."
                if behavior == "wave"
                else "Walk to the target while avoiding nearby obstacles."
            ),
        ),
        obstacles=obstacles,
        future_trajectory=trajectory,
        future_actions=actions,
        metadata={"generator": "procedural_navigation_v1", "control_hz": CONTROL_HZ},
    )


def write_dataset(output: Path, episodes: int, seed: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(episodes):
            handle.write(json.dumps(generate_example(index, rng).to_dict()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    write_dataset(args.output, args.episodes, args.seed)


if __name__ == "__main__":
    main()

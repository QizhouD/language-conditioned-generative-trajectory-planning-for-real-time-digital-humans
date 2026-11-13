"""Closed-loop evaluation: replan along a route the way the engine does.

The offline evaluator scores one 3 s plan per capture window. In the engine the
character instead executes part of a plan, records what it did, and asks again.
This module reproduces that loop in Python, through exactly the code path of the
HTTP service (request.build_batch -> planner.plan_batch -> request.build_plan),
so differences between this and the engine point at the integration, and
differences between this and the open-loop numbers point at the planner.

The loop:

* state: world position and facing, plus the last ``HISTORY_STEPS`` root poses
  at 10 Hz (a character starting from rest has stood at its start pose);
* every ``replan_interval`` the request carries the history and the path
  "current position + remaining route vertices", ending at the goal;
* the first ``replan_interval`` steps of the answer are executed and appended
  to the history. With ``latency_steps`` L the plan is computed from the state
  at t, the previous plan keeps playing for L steps, and the new plan is then
  started at index L (the engine's plain latency compensation; the Unity
  follower now aligns the start spatially instead, see AlignStart there);
* a HOLD answer (every candidate and the spring rejected) means standing still
  until the next replan, as the engine does.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import yaml

from gtp.rootmotion import CONTROL_HZ, GAITS, HISTORY_STEPS, HORIZON_STEPS
from gtp.rootmotion.baselines import constant_speed_follow, spring_follow
from gtp.rootmotion.paths import arclength, point_at, project
from gtp.rootmotion.planner import (
    SOURCE_HOLD, SOURCE_MODEL, SOURCE_SPRING, PlannerConfig, RootMotionPlanner, initial_velocity,
)
from gtp.rootmotion.request import WALK_SPEED_THRESHOLD, build_batch, build_plan
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.windows import path_polyline

ARRIVAL_RADIUS = 0.25  # metres; the Unity follower's arrival tolerance
# Arrival also needs this much of the route covered, so a fold-back whose goal is its
# start does not count as arrived at t = 0.
ARRIVAL_PROGRESS_SLACK = 1.0  # metres short of the route length
SETTLE_SPEED = 0.1  # m/s
SETTLE_STEPS = CONTROL_HZ  # one second below SETTLE_SPEED counts as settled
BACKWARD_TOLERANCE = 0.01  # metres of lost progress along the route


# --------------------------------------------------------------------------- routes


@dataclass(frozen=True)
class Route:
    """A world-space polyline from the start (first waypoint) to the goal (last)."""

    name: str
    waypoints: tuple[tuple[float, float], ...]
    start_yaw: float
    gait: str

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.waypoints, dtype=np.float64)

    @property
    def length(self) -> float:
        return float(arclength(self.points)[-1])


def heading(start: Sequence[float], end: Sequence[float]) -> float:
    """Yaw facing from start to end; yaw 0 faces +z and pi/2 faces +x (Unity's convention)."""
    return math.atan2(end[0] - start[0], end[1] - start[1])


SHAPES: dict[str, tuple[tuple[float, float], ...]] = {
    "straight_3": ((0.0, 0.0), (0.0, 3.0)),
    "straight_6": ((0.0, 0.0), (0.0, 6.0)),
    "straight_12": ((0.0, 0.0), (0.0, 12.0)),
    "corner_4_4": ((0.0, 0.0), (0.0, 4.0), (4.0, 4.0)),
    "zigzag_4": ((0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 8.0)),
    "uturn_4_4": ((0.0, 0.0), (0.0, 4.0), (0.0, 0.0)),
}

# The route of first-run checklist steps 5 and 6 in the LLM-NPC demo scene.
UNITY_ROUTE = ((-4.0, 0.0), (-1.04, -1.28), (1.04, -1.28), (6.5, 0.0))


def standard_routes(gaits: Sequence[str] = ("walk", "run"),
                    shapes: Sequence[str] | None = None) -> list[Route]:
    """Every shape, facing the first leg and turned 90 degrees from it, per gait."""
    routes = []
    for gait in gaits:
        for shape in shapes or SHAPES:
            waypoints = SHAPES[shape]
            aligned = heading(waypoints[0], waypoints[1])
            for start, yaw in (("aligned", aligned), ("turned90", aligned + math.pi / 2)):
                routes.append(Route(f"{shape}/{start}/{gait}", waypoints, yaw, gait))
    return routes


def unity_route(gait: str = "walk") -> Route:
    """Vexa's route around the obstacle, starting at rest facing +x."""
    return Route(f"unity_demo/aligned/{gait}", UNITY_ROUTE, math.pi / 2, gait)


# --------------------------------------------------------------------------- planners

# A plan function takes a /plan request and returns (deltas [T, 3], source, details).
PlanFunction = Callable[[dict], tuple[np.ndarray, int, dict]]


def model_plan_function(planner: RootMotionPlanner, generator: torch.Generator | None) -> PlanFunction:
    """The service's code path: build_batch -> plan_batch, plus selection diagnostics."""

    def plan(request: dict) -> tuple[np.ndarray, int, dict]:
        batch, _ = build_batch(request)
        result = planner.plan_batch(batch, generator)
        candidates = result["candidates"][0]  # [K, T, 3]
        invalid = result["candidate_invalid"][0]
        speeds = torch.linalg.vector_norm(candidates[..., :2], dim=-1).mean(dim=-1) * CONTROL_HZ
        source = int(result["source"][0])
        details = {
            "candidate_speed": float(speeds.mean()),
            "valid_candidate_speed": float(speeds[~invalid].mean()) if bool((~invalid).any()) else None,
            "rejected": float(invalid.float().mean()),
        }
        return result["deltas"][0].cpu().numpy(), source, details

    return plan


def baseline_plan_function(kind: str, gait_speed: Sequence[float]) -> PlanFunction:
    """Spring or constant-speed follower on the same local frame as the model."""

    def plan(request: dict) -> tuple[np.ndarray, int, dict]:
        batch, _ = build_batch(request)
        polyline, valid = path_polyline(batch)
        vertices = polyline[0][valid[0]].numpy().astype(np.float64)
        count = int(batch["path_valid"][0].sum())
        ends = count == 0 or bool(batch["path"][0, count - 1, 3] > 0.5)
        velocity = initial_velocity(batch)[0].numpy().astype(np.float64)
        speed = gait_speed[int(batch["gait"][0])]
        if kind == "spring":
            deltas = spring_follow(vertices, ends, velocity, speed)
        elif kind == "constant":
            deltas = constant_speed_follow(vertices, ends, float(np.hypot(*velocity)), speed)
        else:
            raise ValueError(f"unknown baseline: {kind}")
        return deltas, SOURCE_MODEL, {}

    return plan


# --------------------------------------------------------------------------- simulation


@dataclass
class Rollout:
    route: Route
    poses: np.ndarray  # [N + 1, 3] world (x, z, yaw), start pose first
    seams: np.ndarray  # [N + 1] True on the first step executed from a newly adopted plan
    sources: list[int]
    details: list[dict]
    time_limit: float


def _request(history: list[tuple[float, float, float]], path: np.ndarray, gait: str) -> dict:
    return {
        "history": [{"position": [x, 0.0, z], "yaw": yaw} for x, z, yaw in history[-HISTORY_STEPS:]],
        "path": [[float(x), 0.0, float(z)] for x, z in path],
        "path_ends_at_goal": True,
        "gait": gait,
    }


def remaining_path(route: np.ndarray, cumulative: np.ndarray, position: np.ndarray,
                   progress: float) -> np.ndarray:
    """Current position followed by the route vertices still ahead; always ends at the goal."""
    ahead = [route[index] for index in range(1, len(route)) if cumulative[index] > progress + 1e-6]
    if not ahead:
        ahead = [route[-1]]
    return np.vstack([position[None], np.asarray(ahead)])


def simulate(
    route: Route, plan: PlanFunction, *, replan_interval: float = 1.0, latency_steps: int = 0,
    time_limit: float | None = None, gait_speed: float | None = None, settle_early_stop: bool = True,
) -> Rollout:
    """Roll a planner out along ``route`` from rest. See the module docstring."""
    every = max(1, round(replan_interval * CONTROL_HZ))
    if time_limit is None:
        if gait_speed is None:
            raise ValueError("give time_limit or gait_speed")
        time_limit = 3.0 * route.length / gait_speed
    total = math.ceil(time_limit * CONTROL_HZ)
    points = route.points
    cumulative = arclength(points)
    goal = points[-1]

    start = (float(points[0, 0]), float(points[0, 1]), float(route.start_yaw))
    history = [start] * HISTORY_STEPS
    poses, seams = [start], [False]
    sources: list[int] = []
    details: list[dict] = []
    progress = 0.0
    active: np.ndarray | None = None  # [T + 1, 3] world poses of the plan being executed
    index = 0
    pending: tuple[np.ndarray, int] | None = None  # (plan poses, step at which it is adopted)
    adopted_once = False
    still = 0

    for step in range(total):
        position = np.array(history[-1][:2])
        if step % every == 0:
            path = remaining_path(points, cumulative, position, progress)
            request = _request(history, path, route.gait)
            deltas, source, detail = plan(request)
            _, frame = build_batch(request)
            world = build_plan(np.asarray(deltas).tolist(), frame, fallback=source == SOURCE_HOLD)
            plan_poses = np.array([[p["position"][0], p["position"][2], p["yaw"]] for p in world.trajectory])
            if source == SOURCE_HOLD:
                plan_poses[:] = plan_poses[0]
            sources.append(source)
            details.append(detail)
            pending = (plan_poses, step + latency_steps)
        seam = False
        if pending is not None and pending[1] == step:
            active, index = pending[0], latency_steps
            seam, pending = adopted_once, None
            adopted_once = True
        # Execute one control step.
        index += 1
        if active is None or index >= len(active):
            pose = history[-1]  # nothing to play: stand still
        else:
            pose = (float(active[index, 0]), float(active[index, 1]), float(active[index, 2]))
        history.append(pose)
        poses.append(pose)
        seams.append(seam)
        progress = max(progress, project(points, cumulative, np.array(pose[:2]), progress, window=3.0))

        moved = math.hypot(pose[0] - poses[-2][0], pose[1] - poses[-2][1]) * CONTROL_HZ
        still = still + 1 if moved < SETTLE_SPEED else 0
        arrived = (math.hypot(pose[0] - goal[0], pose[1] - goal[1]) <= ARRIVAL_RADIUS
                   and progress >= route.length - ARRIVAL_PROGRESS_SLACK)
        if settle_early_stop and arrived and still >= 2 * SETTLE_STEPS:
            break

    return Rollout(route, np.asarray(poses), np.asarray(seams), sources, details, time_limit)


# --------------------------------------------------------------------------- metrics


def _distance_to_route(positions: np.ndarray, route: np.ndarray) -> np.ndarray:
    start, end = route[:-1], route[1:]
    segment = end - start
    length_sq = np.maximum((segment ** 2).sum(-1), 1e-12)
    offset = positions[:, None, :] - start[None]
    ratio = np.clip((offset * segment[None]).sum(-1) / length_sq[None], 0.0, 1.0)
    closest = start[None] + ratio[..., None] * segment[None]
    return np.linalg.norm(positions[:, None, :] - closest, axis=-1).min(axis=1)


def rollout_deltas(rollout: Rollout) -> np.ndarray:
    """Executed motion as (dx, dz, dyaw) per step in the world frame."""
    poses = rollout.poses
    turn = np.arctan2(np.sin(np.diff(poses[:, 2])), np.cos(np.diff(poses[:, 2])))
    return np.concatenate([np.diff(poses[:, :2], axis=0), turn[:, None]], axis=1)


def rollout_metrics(rollout: Rollout) -> dict:
    """Per-route closed-loop metrics; see the module docstring for the definitions."""
    route, poses = rollout.route, rollout.poses
    points = route.points
    cumulative = arclength(points)
    goal = points[-1]
    positions = poses[:, :2]
    velocity = np.diff(positions, axis=0) * CONTROL_HZ  # step k -> velocity of step k+1
    speed = np.linalg.norm(velocity, axis=1)
    distance = np.linalg.norm(positions - goal, axis=1)

    monotonic, best = [], 0.0
    for position in positions:
        best = max(best, project(points, cumulative, position, best, window=3.0))
        monotonic.append(best)
    covered = np.asarray(monotonic) >= route.length - ARRIVAL_PROGRESS_SLACK
    inside = np.flatnonzero((distance <= ARRIVAL_RADIUS) & covered)
    inside = inside[inside > 0]
    arrival_step = int(inside[0]) if len(inside) else None
    arrival_time = arrival_step / CONTROL_HZ if arrival_step is not None else None

    moving = speed > WALK_SPEED_THRESHOLD
    first_move = int(np.flatnonzero(moving)[0]) if moving.any() else None
    # The Unity analysis: mean speed over every step from the first plan until arrival.
    until = arrival_step if arrival_step is not None else len(speed)
    follow_mean = float(speed[:until].mean()) if until else 0.0

    # Final rest: the character settles after its last step at or above SETTLE_SPEED.
    fast = np.flatnonzero(speed >= SETTLE_SPEED)
    settle_from = int(fast[-1]) + 1 if len(fast) else 0
    settled = len(speed) - settle_from >= SETTLE_STEPS
    stop_error = float(distance[settle_from]) if settled else None
    residual = float(speed[settle_from:settle_from + SETTLE_STEPS].mean()) if settled else None

    # A planner that never settles (a flow model fidgeting at the goal) still has an
    # end-of-run error: the mean distance and speed over the final second.
    final_distance = float(distance[-SETTLE_STEPS:].mean())
    final_speed = float(speed[-SETTLE_STEPS:].mean())

    # Progress along the route: raw (may drop) and the monotonic version used for the path.
    # Backward steps are counted before arrival only: shuffling on the goal is not "backward".
    raw, best, backward = [], 0.0, []
    last_counted = arrival_step if arrival_step is not None else len(positions)
    for step, position in enumerate(positions):
        value = project(points, cumulative, position, max(0.0, best - 0.5), window=1.5)
        if raw and value < raw[-1] - BACKWARD_TOLERANCE and step <= last_counted:
            backward.append(raw[-1] - value)
        raw.append(value)
        best = max(best, value)

    # Vector acceleration |dv|/dt; seams are steps executed first from a new plan.
    accel = np.linalg.norm(np.diff(velocity, axis=0), axis=1) * CONTROL_HZ  # at steps 2..N
    seam_steps = rollout.seams[2:]
    active_end = min(len(accel), (arrival_step + SETTLE_STEPS) if arrival_step else len(accel))
    window = np.arange(len(accel)) < active_end
    seam_accel = accel[seam_steps & window]
    other_accel = accel[~seam_steps & window]

    sources = np.asarray(rollout.sources)
    candidate = [d for d in rollout.details if d]
    return {
        "route": route.name,
        "length": route.length,
        "time_limit": rollout.time_limit,
        "moving_mean_speed": float(speed[moving].mean()) if moving.any() else 0.0,
        "follow_mean_speed": follow_mean,
        "start_delay": first_move / CONTROL_HZ if first_move is not None else None,
        "arrival_time": arrival_time,
        "arrived_in_time": arrival_time is not None and arrival_time <= rollout.time_limit,
        "settled": bool(settled),
        "stop_error": stop_error,
        "residual_speed": residual,
        "final_distance": final_distance,
        "final_speed": final_speed,
        "max_deviation": float(_distance_to_route(positions, points).max()),
        "fallback_rate": float((sources == SOURCE_SPRING).mean()) if len(sources) else 0.0,
        "hold_rate": float((sources == SOURCE_HOLD).mean()) if len(sources) else 0.0,
        "replans": int(len(sources)),
        "seam_accel_mean": float(seam_accel.mean()) if len(seam_accel) else None,
        "seam_accel_max": float(seam_accel.max()) if len(seam_accel) else None,
        "other_accel_mean": float(other_accel.mean()) if len(other_accel) else None,
        "other_accel_max": float(other_accel.max()) if len(other_accel) else None,
        "backward_steps": len(backward),
        "backward_max": float(max(backward)) if backward else 0.0,
        "candidate_speed": _mean([d["candidate_speed"] for d in candidate]),
        "valid_candidate_speed": _mean([d["valid_candidate_speed"] for d in candidate]),
        "rejected_fraction": _mean([d["rejected"] for d in candidate]),
        "active_steps": int(active_end),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(statistics.fmean(present)) if present else None


def active_kinematics(rollout: Rollout, active_steps: int) -> dict[str, np.ndarray]:
    """Speed, scalar acceleration and turn rate from the start to arrival + 1 s, as in evaluate."""
    deltas = rollout_deltas(rollout)[: active_steps + 1]
    speed = np.linalg.norm(deltas[:, :2], axis=1) * CONTROL_HZ
    return {
        "speed": speed,
        "acceleration": np.diff(speed) * CONTROL_HZ,
        "turn_rate": deltas[:, 2] * CONTROL_HZ,
    }


def wasserstein_np(left: np.ndarray, right: np.ndarray, points: int = 1_000) -> float:
    grid = np.linspace(0.0, 1.0, points)
    return float(np.abs(np.quantile(left, grid) - np.quantile(right, grid)).mean())


def reference_kinematics(config: dict) -> dict[str, dict[str, np.ndarray]]:
    """Capture kinematics of the test windows, per gait."""
    from gtp.rootmotion.train import load_splits

    test = load_splits(config)["test"].as_batch()
    target, gaits = test["target"], test["gait"]
    out = {}
    for index, gait in enumerate(GAITS):
        rows = target[gaits == index]
        if not len(rows):
            continue
        speed = torch.linalg.vector_norm(rows[..., :2], dim=-1) * CONTROL_HZ
        out[gait] = {
            "speed": speed.flatten().numpy(),
            "acceleration": ((speed[:, 1:] - speed[:, :-1]) * CONTROL_HZ).flatten().numpy(),
            "turn_rate": (rows[..., 2] * CONTROL_HZ).flatten().numpy(),
        }
    return out


# --------------------------------------------------------------------------- experiments


def route_seed(route: Route) -> int:
    """Sampling seed fixed per route, so differences across checkpoints come from the model."""
    return zlib.crc32(route.name.encode("utf-8"))


def run_method(
    method: str, routes: Sequence[Route], stats: RootMotionStats, *,
    planner: RootMotionPlanner | None = None, replan_interval: float = 1.0, latency_steps: int = 0,
    reference: dict | None = None,
) -> dict:
    """All routes for one method (one checkpoint for a model). Returns per-route rows and W1."""
    rows, pooled = [], {}
    for route in routes:
        gait_speed = stats.gait_speed[GAITS.index(route.gait)]
        if planner is None:
            plan = baseline_plan_function(method, stats.gait_speed)
        else:
            plan = model_plan_function(planner, torch.Generator().manual_seed(route_seed(route)))
        rollout = simulate(route, plan, replan_interval=replan_interval,
                           latency_steps=latency_steps, gait_speed=gait_speed)
        metrics = rollout_metrics(rollout)
        rows.append(metrics)
        kin = active_kinematics(rollout, metrics["active_steps"])
        for key, values in kin.items():
            pooled.setdefault(route.gait, {}).setdefault(key, []).append(values)
    w1 = {}
    if reference:
        for gait, kinds in pooled.items():
            if gait in reference:
                w1[gait] = {
                    f"w1_{key}": wasserstein_np(np.concatenate(values), reference[gait][key])
                    for key, values in kinds.items()
                }
    return {"routes": rows, "w1": w1}


def open_loop(route: Route, plan: PlanFunction) -> dict:
    """One plan from rest, executed for its whole 3 s horizon: no replanning."""
    rollout = simulate(route, plan, replan_interval=HORIZON_STEPS / CONTROL_HZ + 0.1,
                       time_limit=HORIZON_STEPS / CONTROL_HZ, settle_early_stop=False)
    speed = np.linalg.norm(np.diff(rollout.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
    moving = speed > WALK_SPEED_THRESHOLD
    return {"mean_speed_3s": float(speed.mean()),
            "moving_mean_speed_3s": float(speed[moving].mean()) if moving.any() else 0.0}


def first_seconds_speed(route: Route, plan: PlanFunction, replan_interval: float, seconds: float = 3.0) -> float:
    rollout = simulate(route, plan, replan_interval=replan_interval, time_limit=seconds,
                       settle_early_stop=False)
    return float((np.linalg.norm(np.diff(rollout.poses[:, :2], axis=0), axis=1) * CONTROL_HZ).mean())


def training_path_statistics(config: dict) -> dict:
    """How long the straight stretches the model trains on are, and how fast mocap moves on them."""
    from gtp.rootmotion.train import load_splits
    from gtp.rootmotion.windows import POSITION_SCALE

    train = load_splits(config)["train"].as_batch()
    count = train["path_valid"].sum(dim=1)
    rows = count > 0
    first = torch.linalg.vector_norm(train["path"][:, 0, :2], dim=-1) * POSITION_SCALE
    total = train["path"][torch.arange(len(count)), (count - 1).clamp_min(0), 2] * POSITION_SCALE
    single = rows & (count == 1)
    speed = torch.linalg.vector_norm(train["target"][..., :2], dim=-1).mean(dim=-1) * CONTROL_HZ
    out = {"windows": int(len(count)), "windows_with_path": int(rows.sum())}
    for gait_index, gait in enumerate(GAITS):
        g = rows & (train["gait"] == gait_index)
        if not g.any():
            continue
        entry = {"windows": int(g.sum())}
        for threshold in (3.0, 5.0):
            entry[f"first_segment_ge_{threshold:g}m"] = float((first[g] >= threshold).float().mean())
            entry[f"single_segment_path_ge_{threshold:g}m"] = float(
                ((single & (total >= threshold))[g]).float().mean())
        bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 99.0)]
        entry["mocap_window_speed_by_path_length"] = {
            f"{low:g}-{high:g}m": (float(speed[g & (total >= low) & (total < high)].mean()), int((g & (total >= low) & (total < high)).sum()))
            for low, high in bins if bool((g & (total >= low) & (total < high)).any())
        }
        straight = g & single
        entry["mocap_window_speed_single_segment_by_length"] = {
            f"{low:g}-{high:g}m": (float(speed[straight & (total >= low) & (total < high)].mean()), int((straight & (total >= low) & (total < high)).sum()))
            for low, high in bins if bool((straight & (total >= low) & (total < high)).any())
        }
        out[gait] = entry
    return out


def aggregate(per_seed: list[dict]) -> dict:
    """Mean and sample std across checkpoints, per route and for W1 per gait."""
    out = {"routes": {}, "w1": {}}
    names = [row["route"] for row in per_seed[0]["routes"]]
    for position, name in enumerate(names):
        rows = [seed["routes"][position] for seed in per_seed]
        entry = {}
        for key, value in rows[0].items():
            if isinstance(value, bool) or key == "arrived_in_time":
                entry[key] = float(statistics.fmean(float(bool(r[key])) for r in rows))
            elif isinstance(value, (int, float)) or value is None:
                values = [r[key] for r in rows if r[key] is not None]
                if values:
                    entry[key] = {"mean": statistics.fmean(values),
                                  "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                                  "n": len(values)}
        out["routes"][name] = entry
    for gait in per_seed[0]["w1"]:
        out["w1"][gait] = {
            key: {"mean": statistics.fmean(s["w1"][gait][key] for s in per_seed),
                  "std": statistics.stdev([s["w1"][gait][key] for s in per_seed]) if len(per_seed) > 1 else 0.0}
            for key in per_seed[0]["w1"][gait]
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="training config (data split, reference)")
    parser.add_argument("--flow", type=Path, nargs="*", default=[], help="flow checkpoints, one per seed")
    parser.add_argument("--regressor", type=Path, nargs="*", default=[], help="regressor checkpoints")
    parser.add_argument("--replan-interval", type=float, default=1.0,
                        help="seconds; the LLM-NPC scene's TrajectoryPlanFollower uses 1 s")
    parser.add_argument("--latency-steps", type=int, default=0)
    parser.add_argument("--selection", default="goal_aware",
                        help="first_valid, closest or goal_aware; the regressor has one candidate")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--half-width", type=float, default=0.5)
    parser.add_argument("--baselines", nargs="*", default=["spring", "constant"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference = reference_kinematics(config)
    routes = standard_routes() + [unity_route()]
    checkpoints = {"flow": args.flow, "regressor": args.regressor}
    report: dict = {"replan_interval": args.replan_interval, "latency_steps": args.latency_steps,
                    "selection": args.selection, "methods": {}}
    stats = None
    for method, paths in checkpoints.items():
        per_seed = []
        for path in paths:
            candidates = 1 if method == "regressor" else args.candidates
            planner = RootMotionPlanner.from_checkpoint(
                path, PlannerConfig(candidates=candidates, half_width=args.half_width,
                                    selection=args.selection))
            stats = planner.stats
            per_seed.append(run_method(method, routes, planner.stats, planner=planner,
                                       replan_interval=args.replan_interval,
                                       latency_steps=args.latency_steps, reference=reference))
            print(f"{method} {path}: done", flush=True)
        if per_seed:
            report["methods"][method] = {"per_seed": per_seed, "summary": aggregate(per_seed)}
    if stats is None:
        raise SystemExit("give at least one checkpoint (the baselines use its gait speeds)")
    for method in args.baselines:
        result = run_method(method, routes, stats, replan_interval=args.replan_interval,
                            latency_steps=args.latency_steps, reference=reference)
        report["methods"][method] = {"per_seed": [result], "summary": aggregate([result])}
        print(f"{method}: done", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

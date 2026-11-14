"""Closed-loop experiments of phase e (walk only, standard split).

  e_experiments.py segments   rest-to-rest capture walks -> results/phase_e/capture_segments.json
  e_experiments.py phase1     6 m, 12 m and Unity routes at 0.9 m/s, every method -> results/phase_e/phase1.json
  e_experiments.py phase2     closed loop on the test actor's 10 captured walks -> results/phase_e/phase2.json

Methods: flow goal-aware / consistent at K = 16 and 64, regressor, untuned spring, constant speed.
Feeds: T8 (selection rule) through e_analyze1.py, e_analyze2.py and e_checks.py.
Checkpoints: runs/speedcond/rootmotion_{flow,regressor}_speed_cpu_s{7,11,13,17,19}/best.pt
  (train_speedcond.py).
Run from the repository root with the project venv:
  python scripts/rootmotion/e_experiments.py segments
  python scripts/rootmotion/e_experiments.py phase1
  python scripts/rootmotion/e_experiments.py phase2
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from gtp.rootmotion import CONTROL_HZ
from gtp.rootmotion.closed_loop import (
    SHAPES, Route, active_kinematics, baseline_plan_function, capture_segments, heading,
    model_plan_function, phase_breakdown, reference_kinematics, rollout_metrics, route_seed,
    simulate, speed_by_progress, unity_route,
)
from gtp.rootmotion.extract import parse_clip_name
from gtp.rootmotion.paths import arclength
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner

SEEDS = [7, 11, 13, 17, 19]
OUT = Path("results/phase_e")
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = yaml.safe_load(Path("configs/rootmotion_flow_speed_cpu.yaml").read_text(encoding="utf-8"))
ASKED = 0.9


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ckpt(model, seed):
    return f"runs/speedcond/rootmotion_{model}_speed_cpu_s{seed}/best.pt"


def all_segments(subjects=None):
    tracks = dict(np.load(CONFIG["data_path"]))
    out = []
    for name in sorted(tracks):
        parsed = parse_clip_name(name)
        if parsed is None or parsed[0] != "walk" or (subjects and parsed[1] not in subjects):
            continue
        for segment in capture_segments(tracks[name], name):
            out.append((parsed[1], segment))
    return out


def cycle_stats(speed, every=10):
    """Autocorrelation of the mean-removed speed, and the mean speed by step within a replan cycle."""
    x = np.asarray(speed, dtype=np.float64)
    if len(x) < 2 * every + 2 or x.std() < 1e-9:
        return None
    x = x - x.mean()
    acf = [float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x)) for lag in range(1, 2 * every + 1)]
    return {"acf": acf}


def phase_profile(speed, start, every=10):
    """Mean speed by phase within the replan cycle, over speed[start:] (phase 0 = first step of a plan)."""
    s = np.asarray(speed, dtype=np.float64)
    idx = np.arange(len(s))
    keep = idx >= start
    return [float(s[keep & (idx % every == k)].mean()) if (keep & (idx % every == k)).any() else None
            for k in range(every)]


# ---------------------------------------------------------------------------- methods

def methods():
    """(label, planner factory or None, baseline kind)."""
    out = []
    for selection in ("goal_aware", "consistent"):
        for k in (16, 64):
            out.append((f"flow/{selection}/K{k}", "flow", selection, k))
    out.append(("regressor", "regressor", "goal_aware", 1))
    out.append(("spring", None, None, None))
    out.append(("constant", None, None, None))
    return out


def plan_functions(model, selection, k, route, gait_speed):
    if model is None:
        return None
    return [model_plan_function(RootMotionPlanner.from_checkpoint(ckpt(model, s),
                                                                  PlannerConfig(candidates=k, selection=selection)),
                                torch.Generator().manual_seed(route_seed(route))) for s in SEEDS]


def run_record(rollout, asked):
    metrics = rollout_metrics(rollout)
    speed = np.linalg.norm(np.diff(rollout.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
    arrival = int(round(metrics["arrival_time"] * CONTROL_HZ)) if metrics["arrival_time"] is not None else len(speed)
    phases = phase_breakdown(speed[:arrival], asked)
    record = {k: metrics[k] for k in ("follow_mean_speed", "arrival_time", "final_distance", "max_deviation",
                                     "fallback_rate", "seam_accel_mean", "other_accel_mean", "cruise_p90",
                                     "first_second_speed", "arrived_in_time", "active_steps")}
    record["phases"] = phases
    record["speed"] = speed[:arrival].tolist()
    if phases["reached"]:
        reach = np.flatnonzero(speed[:arrival] >= 0.9 * asked)
        cruise = speed[reach[0]:reach[-1] + 1]
        record["cycle"] = cycle_stats(cruise)
        record["phase_profile"] = phase_profile(speed[:arrival], int(reach[0]))
    kin = active_kinematics(rollout, metrics["active_steps"])
    record["kin"] = {k: v.tolist() for k, v in kin.items()}
    return record


# ---------------------------------------------------------------------------- experiments

def segments_cmd():
    rows = []
    for subject, seg in all_segments():
        speed = np.linalg.norm(np.diff(seg.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
        dist = np.linalg.norm(seg.poses[:, :2] - seg.poses[-1, :2], axis=1)
        inside = np.flatnonzero(dist <= 0.25)
        arrival = int(inside[0]) if len(inside) else len(speed)
        phases = phase_breakdown(speed[:arrival], seg.target_speed)
        reach = np.flatnonzero(speed[:arrival] >= 0.9 * seg.target_speed)
        cycle = None
        if len(reach) > 1:
            cycle = cycle_stats(speed[reach[0]:reach[-1] + 1])
        rows.append({"subject": subject, "clip": seg.clip, "start": seg.start, "end": seg.end,
                     "length": float(arclength(seg.poses[:, :2])[-1]), "vertices": len(seg.route.waypoints),
                     "duration": (seg.end - seg.start) / CONTROL_HZ, "label": seg.target_speed,
                     "mean_speed": float(speed[:arrival].mean()), "arrival_s": arrival / CONTROL_HZ,
                     "phases": phases, "cycle": cycle})
    (OUT / "capture_segments.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    log(f"{len(rows)} segments; test actor: {sum(r['subject'] == 5 for r in rows)}")


def phase1_cmd():
    records = {}
    routes = [Route("straight_6/aligned/walk", SHAPES["straight_6"], 0.0, "walk"),
              Route("straight_12/aligned/walk", SHAPES["straight_12"], 0.0, "walk"), unity_route()]
    stats = RootMotionPlanner.from_checkpoint(ckpt("flow", 7)).stats
    for route in routes:
        for label, model, selection, k in methods():
            if model is None:
                fns = [baseline_plan_function(label, stats.gait_speed)]
            else:
                fns = plan_functions(model, selection, k, route, ASKED)
            runs = []
            for fn in fns:
                rollout = simulate(route, fn, replan_interval=1.0, gait_speed=ASKED,
                                   request_extras={"target_speed": ASKED})
                runs.append(run_record(rollout, ASKED))
            records[f"{route.name}|{label}"] = runs
            log(f"phase1 {route.name} {label}")
    (OUT / "phase1.json").write_text(json.dumps(records), encoding="utf-8")


def phase2_cmd():
    segments = [seg for subject, seg in all_segments({5})]
    stats = RootMotionPlanner.from_checkpoint(ckpt("flow", 7)).stats
    planners = {}
    for label, model, selection, k in methods():
        if model is not None:
            planners[label] = [RootMotionPlanner.from_checkpoint(ckpt(model, s),
                                                                 PlannerConfig(candidates=k, selection=selection))
                               for s in SEEDS]
    records = {}
    for seg in segments:
        duration = (seg.end - seg.start) / CONTROL_HZ
        capture_speed = np.linalg.norm(np.diff(seg.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
        dist = np.linalg.norm(seg.poses[:, :2] - seg.poses[-1, :2], axis=1)
        inside = np.flatnonzero(dist <= 0.25)
        capture_arrival = int(inside[0]) if len(inside) else len(capture_speed)
        entry = {"capture": {"arrival_time": capture_arrival / CONTROL_HZ,
                             "follow_mean_speed": float(capture_speed[:capture_arrival].mean()),
                             "profile": speed_by_progress(seg.poses[:, :2], seg.route.points).tolist(),
                             "phases": phase_breakdown(capture_speed[:capture_arrival], seg.target_speed),
                             "label": seg.target_speed, "length": float(arclength(seg.poses[:, :2])[-1]),
                             "duration": duration}}
        for label, model, selection, k in methods():
            if model is None:
                fns = [baseline_plan_function(label, stats.gait_speed)]
            else:
                fns = [model_plan_function(p, torch.Generator().manual_seed(route_seed(seg.route)))
                       for p in planners[label]]
            runs = []
            for fn in fns:
                rollout = simulate(seg.route, fn, replan_interval=1.0, time_limit=3.0 * duration,
                                   request_extras={"target_speed": seg.target_speed},
                                   initial_history=seg.history)
                record = run_record(rollout, seg.target_speed)
                arrival = rollout_metrics(rollout)["arrival_time"]
                end = int(round(arrival * CONTROL_HZ)) if arrival is not None else len(rollout.poses) - 1
                record["profile"] = speed_by_progress(rollout.poses[: end + 1, :2], seg.route.points).tolist()
                runs.append(record)
            entry[label] = runs
        records[seg.route.name] = entry
        log(f"phase2 {seg.route.name} ({entry['capture']['length']:.1f} m)")
    (OUT / "phase2.json").write_text(json.dumps(records), encoding="utf-8")


if __name__ == "__main__":
    {"segments": segments_cmd, "phase1": phase1_cmd, "phase2": phase2_cmd}[sys.argv[1]]()

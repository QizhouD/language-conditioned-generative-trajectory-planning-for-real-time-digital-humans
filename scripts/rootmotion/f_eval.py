"""Evaluate one variant on the standard split (phase f, walk) -> results/phase_f/eval_NAME.json.

  f_eval.py NAME ...     NAME in base, A1.5, A2, A3, B1, B2, C5, C2, C2+A2 or spring

Per variant: closed loop on the test actor's 10 captured walks (consistent, K = 16, 5 seeds),
12 m and Unity routes at 0.9 m/s, and open loop through `python -m gtp.rootmotion.evaluate`
(-> results/phase_f/open_NAME_sSEED.json; with no previous plan the open loop is goal-aware).
Feeds: T7 (open loop, NAME = base) and T9, through f_table.py. Its plan_ms field is not used
(measured while the GPU was shared).
Checkpoints: runs/speedcond/rootmotion_flow_speed_cpu_s{7,11,13,17,19}/best.pt (base, A*);
  runs/phase_f/{B1,B2,C5,C2}_s*/best.pt (f_train.py screen).
Run from the repository root with the project venv:
  python scripts/rootmotion/f_eval.py base spring
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from gtp.rootmotion import CONTROL_HZ
from gtp.rootmotion.closed_loop import (
    SHAPES, Route, active_kinematics, baseline_plan_function, capture_segments, model_plan_function,
    phase_breakdown, rollout_metrics, route_seed, simulate, speed_by_progress, unity_route,
)
from gtp.rootmotion.extract import parse_clip_name
from gtp.rootmotion.paths import arclength, project
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.request import build_batch

SEEDS = [7, 11, 13, 17, 19]
BASE = "runs/speedcond/rootmotion_flow_speed_cpu_s{}/best.pt"
VARIANTS = {
    "base": (BASE, 1.0), "A1.5": (BASE, 1.5), "A2": (BASE, 2.0), "A3": (BASE, 3.0),
    "B1": ("runs/phase_f/B1_s{}/best.pt", 1.0), "B2": ("runs/phase_f/B2_s{}/best.pt", 1.0),
    "C5": ("runs/phase_f/C5_s{}/best.pt", 1.0), "C2": ("runs/phase_f/C2_s{}/best.pt", 1.0),
}
OUT = Path("results/phase_f")
CONFIG = yaml.safe_load(Path("configs/rootmotion_flow_speed_cpu.yaml").read_text(encoding="utf-8"))


def register(name):
    """Combinations like 'C5+A2' use C5's checkpoints with A2's guidance."""
    if name in VARIANTS or name == "spring":
        return
    parts = name.split("+")
    pattern = next((VARIANTS[p][0] for p in parts if VARIANTS[p][0] != BASE), BASE)
    if pattern == BASE and not any(p.startswith("A") for p in parts):
        raise SystemExit(f"unknown variant {name}")
    trained = [p for p in parts if not p.startswith("A")]
    if len(trained) > 1 or (trained and trained[0] not in VARIANTS):
        pattern = f"runs/phase_f/{'+'.join(trained)}_s{{}}/best.pt"
    guidance = next((VARIANTS[p][1] for p in parts if p.startswith("A")), 1.0)
    VARIANTS[name] = (pattern, guidance)


def gated_arrival(poses, route_points):
    cumulative = arclength(route_points)
    best = 0.0
    for k, p in enumerate(poses[:, :2]):
        best = max(best, project(route_points, cumulative, p, best, window=3.0))
        if k > 0 and np.hypot(*(p - route_points[-1])) <= 0.25 and best >= cumulative[-1] - 1.0:
            return k
    return len(poses) - 1


def time_to(speed, threshold):
    hit = np.flatnonzero(np.asarray(speed) >= threshold)
    return hit[0] / CONTROL_HZ if len(hit) else None


def record(rollout, asked):
    metrics = rollout_metrics(rollout)
    speed = np.linalg.norm(np.diff(rollout.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
    arrival = int(round(metrics["arrival_time"] * CONTROL_HZ)) if metrics["arrival_time"] is not None else len(speed)
    s = speed[:arrival]
    a, b = time_to(s, 0.45), time_to(s, 0.81)
    stall = None
    if a is not None:
        window = s[int(a * CONTROL_HZ):int((b if b is not None else len(s) / CONTROL_HZ) * CONTROL_HZ)]
        stall = float((window < 0.45).mean()) if len(window) else 0.0
    kin = active_kinematics(rollout, metrics["active_steps"])
    out = {k: metrics[k] for k in ("follow_mean_speed", "arrival_time", "final_distance", "max_deviation",
                                  "fallback_rate", "seam_accel_mean", "other_accel_mean", "cruise_p90")}
    out.update(to_045=a, to_081=b, stall=stall, phases=phase_breakdown(s, asked),
               kin={k: v.tolist() for k, v in kin.items()})
    return out, rollout


def main(name):
    register(name)
    tracks = dict(np.load(CONFIG["data_path"]))
    walks = []
    for clip in sorted(tracks):
        parsed = parse_clip_name(clip)
        if parsed and parsed[0] == "walk" and parsed[1] == 5:
            walks += capture_segments(tracks[clip], clip)
    stats = RootMotionPlanner.from_checkpoint(BASE.format(7)).stats
    if name == "spring":
        fns = lambda route: [baseline_plan_function("spring", stats.gait_speed)]
        guidance = 1.0
    else:
        pattern, guidance = VARIANTS[name]
        planners = [RootMotionPlanner.from_checkpoint(pattern.format(s), PlannerConfig(candidates=16, speed_guidance=guidance))
                    for s in SEEDS]
        fns = lambda route: [model_plan_function(p, torch.Generator().manual_seed(route_seed(route))) for p in planners]
    result = {"name": name, "guidance": guidance, "walks": {}, "synthetic": {}}
    for seg in walks:
        capture_arrival = gated_arrival(seg.poses, seg.route.points)
        cap_speed = np.linalg.norm(np.diff(seg.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
        end = min(len(seg.poses) - 1, capture_arrival + CONTROL_HZ)
        d = np.diff(seg.poses[: end + 1], axis=0)
        sp = np.linalg.norm(d[:, :2], axis=1) * CONTROL_HZ
        entry = {"capture": {"arrival_time": capture_arrival / CONTROL_HZ, "mean": float(cap_speed[:capture_arrival].mean()),
                             "length": float(arclength(seg.poses[:, :2])[-1]), "label": seg.target_speed,
                             "profile": speed_by_progress(seg.poses[: capture_arrival + 1, :2], seg.route.points).tolist(),
                             "kin": {"speed": sp.tolist(), "acceleration": (np.diff(sp) * CONTROL_HZ).tolist(),
                                     "turn_rate": (np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2])) * CONTROL_HZ).tolist()}},
                 "runs": []}
        duration = (seg.end - seg.start) / CONTROL_HZ
        for fn in fns(seg.route):
            rollout = simulate(seg.route, fn, time_limit=3.0 * duration, initial_history=seg.history,
                               request_extras={"target_speed": seg.target_speed})
            rec, rollout = record(rollout, seg.target_speed)
            stop = int(round(rec["arrival_time"] * CONTROL_HZ)) if rec["arrival_time"] is not None else len(rollout.poses) - 1
            rec["profile"] = speed_by_progress(rollout.poses[: stop + 1, :2], seg.route.points).tolist()
            entry["runs"].append(rec)
        result["walks"][seg.route.name] = entry
        print(f"[{time.strftime('%H:%M:%S')}] {name} {seg.route.name}", flush=True)
    for route in (Route("straight_12/aligned/walk", SHAPES["straight_12"], 0.0, "walk"), unity_route()):
        runs = []
        for fn in fns(route):
            rec, _ = record(simulate(route, fn, gait_speed=0.9, request_extras={"target_speed": 0.9}), 0.9)
            runs.append(rec)
        result["synthetic"][route.name] = runs
    # Planning time per request (median of 30 after warm-up).
    if name != "spring":
        batch, _ = build_batch({"history": [{"position": [0, 0, 0], "yaw": 0}], "path": [[0, 0, 0], [0, 0, 12]],
                                "gait": "walk", "target_speed": 0.9})
        times = []
        for i in range(33):
            started = time.perf_counter()
            planners[0].plan_batch(batch, torch.Generator().manual_seed(i))
            times.append((time.perf_counter() - started) * 1000)
        result["plan_ms"] = float(np.median(times[3:]))
        # Open loop: the standard evaluate on each seed (flow only; consistent = goal-aware without history).
        result["open"] = []
        for s in SEEDS:
            out = OUT / f"open_{name}_s{s}.json"
            subprocess.run([sys.executable, "-m", "gtp.rootmotion.evaluate", "--config",
                            "configs/rootmotion_flow_speed_cpu.yaml", "--flow", pattern.format(s),
                            "--speed-guidance", str(guidance), "--output", str(out)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
            rows = {r["method"]: r for r in json.loads(out.read_text())["results"]}
            result["open"].append(rows["flow K=16 + filter"])
    (OUT / f"eval_{name}.json").write_text(json.dumps(result), encoding="utf-8")
    print(f"done {name}", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)

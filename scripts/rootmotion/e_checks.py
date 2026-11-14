"""Phase-e checks (reads results/phase_e/phase1.json and phase2.json).

  1. Captured walks with the progress-gated arrival, short (<= 35 m) and long: T8 rows
     "short captured walks: arrival / capture" and "mean speed / capture".
  2. Human reference for starts from rest: length-matched rest-to-rest walks (8 to 15 m,
     label 0.8 to 1.1, n = 6) -> time to 0.45 / 0.81 m/s and mean / label: the human row of T6.
  3. plan_batch time at K = 16 / 64 (median of 30, no previous plan passed): superseded,
     see T10 and latency.py.
Checkpoints: runs/speedcond/rootmotion_flow_speed_cpu_s7/best.pt (section 3 only); needs data/lafan1_root.npz.
Run from the repository root with the project venv:
  python scripts/rootmotion/e_checks.py
"""
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from gtp.rootmotion import CONTROL_HZ
from gtp.rootmotion.closed_loop import capture_segments, wasserstein_np
from gtp.rootmotion.extract import parse_clip_name
from gtp.rootmotion.paths import arclength, project
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.request import build_batch

OUT = Path("results/phase_e")
PM = " ± "
config = yaml.safe_load(Path("configs/rootmotion_flow_speed_cpu.yaml").read_text(encoding="utf-8"))
tracks = dict(np.load(config["data_path"]))


def ms(values, d=2):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not values:
        return "-"
    return f"{st.fmean(values):.{d}f}" + (PM + f"{st.stdev(values):.{d}f}" if len(values) > 1 else "")


def gated_arrival(poses, route_points):
    """Same rule as the rollouts: within 0.25 m of the end AND route progress within 1 m of its length."""
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


# ------------------------------------------------------------------ 1. capture arrival with the progress gate
segments = {}
all_walks = []
for name in sorted(tracks):
    parsed = parse_clip_name(name)
    if parsed and parsed[0] == "walk":
        for seg in capture_segments(tracks[name], name):
            all_walks.append((parsed[1], seg))
            if parsed[1] == 5:
                segments[seg.route.name] = seg

phase2 = json.loads((OUT / "phase2.json").read_text())
METHODS = ["flow/goal_aware/K16", "flow/consistent/K16", "flow/goal_aware/K64", "flow/consistent/K64",
           "regressor", "spring", "constant"]
fixed = {}
print("## 1. Capture arrival with the progress gate\n")
print("| walk | length | old capture arrival s | gated arrival s | walk duration s | capture mean speed (gated) |")
print("|---|---|---|---|---|---|")
for key, entry in phase2.items():
    seg = segments[key]
    arrival = gated_arrival(seg.poses, seg.route.points)
    speed = np.linalg.norm(np.diff(seg.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
    end = min(len(seg.poses) - 1, arrival + CONTROL_HZ)
    d = np.diff(seg.poses[: end + 1], axis=0)
    sp = np.linalg.norm(d[:, :2], axis=1) * CONTROL_HZ
    fixed[key] = {"arrival": arrival / CONTROL_HZ, "mean": float(speed[:arrival].mean()),
                  "length": entry["capture"]["length"],
                  "kin": {"speed": sp, "acceleration": np.diff(sp) * CONTROL_HZ,
                          "turn_rate": np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2])) * CONTROL_HZ}}
    print(f"| {key.split('/', 1)[1]} | {entry['capture']['length']:.1f} | {entry['capture']['arrival_time']:.1f} | "
          f"{arrival / CONTROL_HZ:.1f} | {(seg.end - seg.start) / CONTROL_HZ:.1f} | {fixed[key]['mean']:.2f} |")

for group, pick in (("short (<= 35 m, n=8)", lambda L: L <= 35), ("long (> 35 m, n=2)", lambda L: L > 35)):
    keys = [k for k in phase2 if pick(fixed[k]["length"])]
    print(f"\n### Real walks, {group}: per seed median over walks, then mean{PM}std over seeds\n")
    print("| method | arrival / capture | mean speed / capture | W1 speed | W1 accel | W1 turn (median of per-walk W1) | cruise s | cruise CV (cruise >= 2 s) |")
    print("|---|---|---|---|---|---|---|---|")
    for m in METHODS:
        n = len(phase2[keys[0]][m])
        t, v, w1s, w1a, w1t, cs, cv = [], [], [], [], [], [], []
        for i in range(n):
            runs = [(k, phase2[k][m][i]) for k in keys]
            t.append(np.median([r["arrival_time"] / fixed[k]["arrival"] for k, r in runs if r["arrival_time"] is not None]))
            v.append(np.median([r["follow_mean_speed"] / fixed[k]["mean"] for k, r in runs]))
            per = {"s": [], "a": [], "t": []}
            for k, r in runs:
                per["s"].append(wasserstein_np(np.asarray(r["kin"]["speed"]), fixed[k]["kin"]["speed"]))
                per["a"].append(wasserstein_np(np.asarray(r["kin"]["acceleration"]), fixed[k]["kin"]["acceleration"]))
                per["t"].append(wasserstein_np(np.asarray(r["kin"]["turn_rate"]), fixed[k]["kin"]["turn_rate"]))
            w1s.append(np.median(per["s"]))
            w1a.append(np.median(per["a"]))
            w1t.append(np.median(per["t"]))
            cs.append(np.median([r["phases"]["cruise_s"] for _, r in runs]))
            cvs = [r["phases"]["cruise_cv"] for _, r in runs if r["phases"]["cruise_cv"] is not None and r["phases"]["cruise_s"] >= 2.0]
            cv.append(np.median(cvs) if cvs else np.nan)
        print(f"| {m} | {ms(t)} | {ms(v)} | {ms(w1s, 3)} | {ms(w1a, 3)} | {ms(w1t, 3)} | {ms(cs, 1)} | {ms(cv, 3)} |")

# ------------------------------------------------------------------ 2. length-matched human starts, absolute thresholds
print("\n## 2. Starts at absolute thresholds\n")
human = {"to_0.45": [], "to_0.81": [], "mean": [], "ratio": [], "arrival_speed": []}
matched = []
for subject, seg in all_walks:
    L = float(arclength(seg.poses[:, :2])[-1])
    if 8.0 <= L <= 15.0 and 0.8 <= seg.target_speed <= 1.1:
        speed = np.linalg.norm(np.diff(seg.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
        arrival = gated_arrival(seg.poses, seg.route.points)
        matched.append(seg)
        human["to_0.45"].append(time_to(speed[:arrival], 0.45))
        human["to_0.81"].append(time_to(speed[:arrival], 0.81))
        human["mean"].append(float(speed[:arrival].mean()))
        human["ratio"].append(float(speed[:arrival].mean()) / seg.target_speed)
        human["arrival_speed"].append(float(speed[arrival - 1]) if arrival > 0 else None)


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"median {np.median(xs):.2f} (p25 {np.percentile(xs, 25):.2f}, p75 {np.percentile(xs, 75):.2f}, n={len(xs)}, never {len([1 for x in xs if x is None])})"


never81 = sum(1 for x in human["to_0.81"] if x is None)
print(f"- humans, rest-to-rest walks 8-15 m with label 0.8-1.1 (n={len(matched)}): time to 0.45 m/s "
      f"{med(human['to_0.45'])}; time to 0.81 m/s {med(human['to_0.81'])} (never reached: {never81}); "
      f"mean speed {np.median(human['mean']):.2f}; mean/label {np.median(human['ratio']):.2f}; "
      f"speed on the step entering 0.25 m of the end {med(human['arrival_speed'])}")

phase1 = json.loads((OUT / "phase1.json").read_text())
for route in ("straight_6/aligned/walk", "straight_12/aligned/walk", "unity_demo/aligned/walk"):
    for m in ("flow/goal_aware/K16", "flow/consistent/K16", "flow/goal_aware/K64", "flow/consistent/K64", "regressor", "spring"):
        runs = phase1[f"{route}|{m}"]
        to45 = [time_to(r["speed"], 0.45) for r in runs]
        to81 = [time_to(r["speed"], 0.81) for r in runs]
        # Share of time between first reaching 0.45 and first reaching 0.81 spent below 0.45 again ("stalls").
        stall = []
        for r in runs:
            s = np.asarray(r["speed"])
            a, b = time_to(s, 0.45), time_to(s, 0.81)
            if a is not None:
                hi = int((b if b is not None else len(s) / CONTROL_HZ) * CONTROL_HZ)
                window = s[int(a * CONTROL_HZ):hi]
                stall.append(float((window < 0.45).mean()) if len(window) else 0.0)
        arrival_speed = [r["speed"][-1] if r["speed"] else None for r in runs]
        print(f"- {route.split('/')[0]:12s} {m:22s} to 0.45: {ms(to45, 1)} s; to 0.81: {ms(to81, 1)} s; "
              f"time below 0.45 between them: {ms(stall)}; speed entering 0.25 m: {ms(arrival_speed)}")

# ------------------------------------------------------------------ 3. planning time per request
print("\n## 3. plan_batch time, K = 16 and 64 (CPU, one request, median of 30)\n")
request = {"history": [{"position": [0.0, 0.0, 0.0], "yaw": 0.0}], "path": [[0.0, 0.0, 0.0], [0.0, 0.0, 12.0]],
           "gait": "walk", "target_speed": 0.9}
batch, _ = build_batch(request)
previous = torch.zeros(1, 20, 2)
previous[0, :, 1] = torch.arange(1, 21) * 0.09
for k in (16, 64):
    for selection in ("goal_aware", "consistent"):
        planner = RootMotionPlanner.from_checkpoint("runs/speedcond/rootmotion_flow_speed_cpu_s7/best.pt",
                                                    PlannerConfig(candidates=k, selection=selection))
        times = []
        for i in range(33):
            started = time.perf_counter()
            planner.plan_batch(batch, torch.Generator().manual_seed(i), previous=previous)
            times.append((time.perf_counter() - started) * 1000)
        print(f"- K={k} {selection}: {np.median(times[3:]):.1f} ms")

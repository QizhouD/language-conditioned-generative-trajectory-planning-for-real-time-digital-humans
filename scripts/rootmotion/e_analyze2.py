"""Phase-e captured-walk tables (reads results/phase_e/phase2.json).

Produces: per-method table against the test actor's 10 captured walks, on stdout.
  Source of T8 row "seam / other accel, captured walks". (Its arrival ratios use the
  ungated capture arrival; e_checks.py section 1 replaces them.)
Checkpoints: none (reads the JSON written by e_experiments.py phase2); needs data/lafan1_root.npz.
Run from the repository root with the project venv:
  python scripts/rootmotion/e_analyze2.py
"""
import json
import statistics as st
from pathlib import Path

import numpy as np
import yaml

from gtp.rootmotion import CONTROL_HZ
from gtp.rootmotion.closed_loop import capture_segments, wasserstein_np
from gtp.rootmotion.extract import parse_clip_name

OUT = Path("results/phase_e")
PM = " ± "
data = json.loads((OUT / "phase2.json").read_text())
config = yaml.safe_load(Path("configs/rootmotion_flow_speed_cpu.yaml").read_text(encoding="utf-8"))
tracks = dict(np.load(config["data_path"]))
METHODS = ["flow/goal_aware/K16", "flow/consistent/K16", "flow/goal_aware/K64", "flow/consistent/K64",
           "regressor", "spring", "constant"]

# Captured kinematics over the same window as the rollouts (start to arrival + 1 s).
segments = {}
for name in sorted(tracks):
    parsed = parse_clip_name(name)
    if parsed and parsed[0] == "walk" and parsed[1] == 5:
        for seg in capture_segments(tracks[name], name):
            segments[seg.route.name] = seg
capture_kin = {"speed": [], "acceleration": [], "turn_rate": []}
for key, entry in data.items():
    seg = segments[key]
    poses = seg.poses
    end = min(len(poses) - 1, int(round(entry["capture"]["arrival_time"] * CONTROL_HZ)) + CONTROL_HZ)
    d = np.diff(poses[: end + 1], axis=0)
    speed = np.linalg.norm(d[:, :2], axis=1) * CONTROL_HZ
    capture_kin["speed"].append(speed)
    capture_kin["acceleration"].append(np.diff(speed) * CONTROL_HZ)
    capture_kin["turn_rate"].append(np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2])) * CONTROL_HZ)
capture_kin = {k: np.concatenate(v) for k, v in capture_kin.items()}

print(f"## {len(data)} captured walks (test actor, subject 5)\n")
print("| walk | length m | RDP vertices | label m/s | capture arrival s | capture mean speed | capture start s |")
print("|---|---|---|---|---|---|---|")
for key, entry in data.items():
    c = entry["capture"]
    print(f"| {key.split('/', 1)[1]} | {c['length']:.1f} | {len(segments[key].route.waypoints)} | {c['label']:.2f} | "
          f"{c['arrival_time']:.1f} | {c['follow_mean_speed']:.2f} | {c['phases']['start_s']:.1f} |")


def ms(values, d=2):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not values:
        return "-"
    return f"{st.fmean(values):.{d}f}" + (PM + f"{st.stdev(values):.{d}f}" if len(values) > 1 else "")


print("\n## Against the capture (per seed: median over walks of the per-walk ratio; then mean" + PM + "std over seeds)\n")
print("| method | arrived | arrival time / capture | mean speed / capture | speed-by-progress RMSE m/s | start s | cruise CV | "
      "seam accel | other accel | final dist | max dev | fallback | W1 speed | W1 accel | W1 turn |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
keys = list(data)
for m in METHODS:
    nseeds = len(data[keys[0]][m])
    cols = {k: [] for k in ("arrived", "t", "v", "rmse", "start", "cv", "seam", "other", "final", "dev", "fb", "w1s", "w1a", "w1t")}
    for i in range(nseeds):
        runs = [(data[k]["capture"], data[k][m][i]) for k in keys]
        arrived = [r["arrival_time"] is not None for _, r in runs]
        cols["arrived"].append(np.mean(arrived))
        cols["t"].append(np.median([r["arrival_time"] / c["arrival_time"] for c, r in runs if r["arrival_time"] is not None]))
        cols["v"].append(np.median([r["follow_mean_speed"] / c["follow_mean_speed"] for c, r in runs]))
        rmses = []
        for c, r in runs:
            a, b = np.asarray(c["profile"], dtype=float), np.asarray(r["profile"], dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.any():
                rmses.append(float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))))
        cols["rmse"].append(np.median(rmses))
        cols["start"].append(np.median([r["phases"]["start_s"] for _, r in runs]))
        cvs = [r["phases"]["cruise_cv"] for _, r in runs if r["phases"]["cruise_cv"] is not None]
        cols["cv"].append(np.median(cvs) if cvs else np.nan)
        cols["seam"].append(np.mean([r["seam_accel_mean"] for _, r in runs if r["seam_accel_mean"] is not None]))
        cols["other"].append(np.mean([r["other_accel_mean"] for _, r in runs if r["other_accel_mean"] is not None]))
        cols["final"].append(np.median([r["final_distance"] for _, r in runs]))
        cols["dev"].append(np.median([r["max_deviation"] for _, r in runs]))
        cols["fb"].append(100 * np.mean([r["fallback_rate"] for _, r in runs]))
        kin = {k: np.concatenate([np.asarray(r["kin"][k]) for _, r in runs]) for k in capture_kin}
        cols["w1s"].append(wasserstein_np(kin["speed"], capture_kin["speed"]))
        cols["w1a"].append(wasserstein_np(kin["acceleration"], capture_kin["acceleration"]))
        cols["w1t"].append(wasserstein_np(kin["turn_rate"], capture_kin["turn_rate"]))
    print(f"| {m} | {ms(cols['arrived'])} | {ms(cols['t'])} | {ms(cols['v'])} | {ms(cols['rmse'], 3)} | {ms(cols['start'], 1)} | "
          f"{ms(cols['cv'], 3)} | {ms(cols['seam'])} | {ms(cols['other'])} | {ms(cols['final'], 3)} | {ms(cols['dev'], 3)} | "
          f"{ms(cols['fb'], 1)}% | {ms(cols['w1s'], 3)} | {ms(cols['w1a'], 3)} | {ms(cols['w1t'], 3)} |")

cap_start = [data[k]["capture"]["phases"]["start_s"] for k in keys]
cap_cv = [data[k]["capture"]["phases"]["cruise_cv"] for k in keys if data[k]["capture"]["phases"]["cruise_cv"] is not None]
print(f"\ncapture itself: start s median {np.median(cap_start):.1f}, cruise CV median {np.median(cap_cv):.3f}")

print("\n## Per walk: arrival time / capture (flow goal_aware K16 and consistent K16, mean over seeds; spring)\n")
for k in keys:
    c = data[k]["capture"]
    cells = []
    for m in ("flow/goal_aware/K16", "flow/consistent/K16", "regressor", "spring"):
        ts = [r["arrival_time"] / c["arrival_time"] for r in data[k][m] if r["arrival_time"] is not None]
        cells.append(f"{m.split('/')[0] if m in ('regressor', 'spring') else m.split('/')[1]} "
                     f"{np.mean(ts):.2f}" + ("" if len(ts) == len(data[k][m]) else f" ({len(ts)}/{len(data[k][m])} arrived)"))
    print(f"- {k.split('/', 1)[1]} ({c['length']:.1f} m): " + "; ".join(cells))

"""Phase-e synthetic-route tables and speed-over-time plots (reads results/phase_e/phase1.json).

Produces: phase / cycle / seam tables on stdout, and results/phase_e/speed_time_{straight_12,unity_demo}.png.
  Source of T8 row "seam / other accel, synthetic routes".
Checkpoints: none (reads the JSON written by e_experiments.py phase1).
Run from the repository root with the project venv:
  python scripts/rootmotion/e_analyze1.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from gtp.rootmotion.closed_loop import reference_kinematics, wasserstein_np

OUT = Path("results/phase_e")
PM = " ± "
ASKED = 0.9
data = json.loads((OUT / "phase1.json").read_text())
config = yaml.safe_load(Path("configs/rootmotion_flow_speed_cpu.yaml").read_text(encoding="utf-8"))
reference = reference_kinematics(config)["walk"]
ROUTES = ["straight_6/aligned/walk", "straight_12/aligned/walk", "unity_demo/aligned/walk"]
METHODS = ["flow/goal_aware/K16", "flow/consistent/K16", "flow/goal_aware/K64", "flow/consistent/K64",
           "regressor", "spring", "constant"]


def ms(values, d=2):
    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return "-"
    return f"{st.fmean(values):.{d}f}" + (PM + f"{st.stdev(values):.{d}f}" if len(values) > 1 else "")


def lost(run, part):
    """Seconds lost in a phase against covering the same distance at the asked speed."""
    speed = np.asarray(run["speed"])
    p = run["phases"]
    if not p["reached"]:
        return {"start": len(speed) / 10 - speed.sum() / 10 / ASKED, "cruise": 0.0, "decel": 0.0}[part]
    reach = np.flatnonzero(speed >= 0.9 * ASKED)
    a, b = reach[0], reach[-1]
    seg = {"start": speed[:a], "cruise": speed[a:b + 1], "decel": speed[b + 1:]}[part]
    return len(seg) / 10 - seg.sum() / 10 / ASKED


print("## Phases at 0.9 m/s (mean" + PM + "std over seeds; one run for spring/constant)\n")
print("| route | method | reached 90% | start s | start speed | cruise s | cruise speed | cruise CV | decel s | decel speed | "
      "lost s: start / cruise / decel | mean speed | arrival s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for route in ROUTES:
    for m in METHODS:
        runs = data[f"{route}|{m}"]
        ph = [r["phases"] for r in runs]
        g = lambda k: [p[k] for p in ph]
        lost_cells = " / ".join(ms([lost(r, part) for r in runs], 1) for part in ("start", "cruise", "decel"))
        print(f"| {route.split('/')[0]} | {m} | {sum(p['reached'] for p in ph)}/{len(ph)} | {ms(g('start_s'), 1)} | "
              f"{ms(g('start_speed'))} | {ms(g('cruise_s'), 1)} | {ms(g('cruise_speed'))} | {ms(g('cruise_cv'), 3)} | "
              f"{ms(g('decel_s'), 1)} | {ms(g('decel_speed'))} | {lost_cells} | "
              f"{ms([r['follow_mean_speed'] for r in runs])} | {ms([r['arrival_time'] for r in runs], 1)} |")

print("\n## Replan-cycle structure in the cruise phase (replan every 10 steps)\n")
print("| route | method | ACF lag 5 | ACF lag 10 | ACF lag 20 | in-cycle speed: step 0..9 (mean over runs) | amplitude / cruise mean |")
print("|---|---|---|---|---|---|---|")
for route in ROUTES:
    for m in METHODS:
        runs = [r for r in data[f"{route}|{m}"] if r.get("cycle")]
        if not runs:
            print(f"| {route.split('/')[0]} | {m} | - | - | - | - | - |")
            continue
        acf = np.array([r["cycle"]["acf"] for r in runs])
        prof = np.array([[np.nan if v is None else v for v in r["phase_profile"]] for r in runs])
        mean_prof = np.nanmean(prof, axis=0)
        amp = [(np.nanmax(p) - np.nanmin(p)) / np.nanmean(p) for p in prof]
        print(f"| {route.split('/')[0]} | {m} | {ms(acf[:, 4])} | {ms(acf[:, 9])} | {ms(acf[:, 19])} | "
              f"{' '.join(f'{v:.2f}' for v in mean_prof)} | {ms(amp)} |")

print("\n## Seams, accuracy and realism (pooled over the three routes, per seed)\n")
print("| method | seam accel | other accel | final dist | max dev | fallback | W1 speed | W1 accel | W1 turn |")
print("|---|---|---|---|---|---|---|---|---|")
for m in METHODS:
    nseeds = len(data[f"{ROUTES[0]}|{m}"])
    per = {k: [] for k in ("seam", "other", "final", "dev", "fb", "w1s", "w1a", "w1t")}
    for i in range(nseeds):
        runs = [data[f"{route}|{m}"][i] for route in ROUTES]
        per["seam"].append(st.fmean(r["seam_accel_mean"] for r in runs if r["seam_accel_mean"] is not None))
        per["other"].append(st.fmean(r["other_accel_mean"] for r in runs))
        per["final"].append(st.fmean(r["final_distance"] for r in runs))
        per["dev"].append(st.fmean(r["max_deviation"] for r in runs))
        per["fb"].append(st.fmean(r["fallback_rate"] for r in runs))
        kin = {k: np.concatenate([np.asarray(r["kin"][k]) for r in runs]) for k in ("speed", "acceleration", "turn_rate")}
        per["w1s"].append(wasserstein_np(kin["speed"], reference["speed"]))
        per["w1a"].append(wasserstein_np(kin["acceleration"], reference["acceleration"]))
        per["w1t"].append(wasserstein_np(kin["turn_rate"], reference["turn_rate"]))
    print(f"| {m} | {ms(per['seam'])} | {ms(per['other'])} | {ms(per['final'], 3)} | {ms(per['dev'], 3)} | "
          f"{ms([100 * v for v in per['fb']], 1)}% | {ms(per['w1s'], 3)} | {ms(per['w1a'], 3)} | {ms(per['w1t'], 3)} |")

# Plots: seed 7 (first run) of each method, speed against time with replan marks.
for route in ("straight_12/aligned/walk", "unity_demo/aligned/walk"):
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for ax, m in zip(axes, ["flow/goal_aware/K16", "flow/consistent/K16", "flow/consistent/K64", "regressor", "spring"]):
        run = data[f"{route}|{m}"][0]
        speed = np.asarray(run["speed"])
        t = np.arange(1, len(speed) + 1) / 10
        ax.plot(t, speed, lw=1.4)
        for k in range(0, len(speed), 10):
            ax.axvline(k / 10, color="0.8", lw=0.6, zorder=0)
        ax.axhline(ASKED, color="tab:red", lw=0.8, ls="--")
        ax.axhline(0.9 * ASKED, color="tab:orange", lw=0.6, ls=":")
        ax.set_ylabel("m/s")
        ax.set_title(f"{m} (seed 7)" if m.startswith(("flow", "regressor")) else m, fontsize=9, loc="left")
        ax.set_ylim(0, 1.6)
    axes[-1].set_xlabel("time (s); grey lines = replans every 1 s; red = asked 0.9, orange = 90%")
    fig.suptitle(f"Speed over time, {route.split('/')[0]} at 0.9 m/s", fontsize=11)
    fig.tight_layout()
    name = route.split("/")[0]
    fig.savefig(OUT / f"speed_time_{name}.png", dpi=110)
    plt.close(fig)
print("\nplots written")

"""Time to 0.81 m/s on the 12 m straight and the Unity route at 0.9 m/s, 20 runs per variant
(5 checkpoints x 4 sampling seeds), consistent selection, K = 16.

Produces: results/phase_f/resample_NAMES.json and stdout. Source of the flow row of T6
(NAME = base -> resample_base_C2.json) and the start-time columns of T9.
Checkpoints: as in f_eval.py for each NAME.
Run from the repository root with the project venv:
  python scripts/rootmotion/f_resample.py base C2
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

from gtp.rootmotion import CONTROL_HZ
from gtp.rootmotion.closed_loop import SHAPES, Route, model_plan_function, rollout_metrics, route_seed, simulate, unity_route
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner

SEEDS = [7, 11, 13, 17, 19]
OFFSETS = [0, 1, 2, 3]
VARIANTS = {"base": ("runs/speedcond/rootmotion_flow_speed_cpu_s{}/best.pt", 1.0),
            "C2": ("runs/phase_f/C2_s{}/best.pt", 1.0),
            "A2": ("runs/speedcond/rootmotion_flow_speed_cpu_s{}/best.pt", 2.0),
            "C2+A2": ("runs/phase_f/C2_s{}/best.pt", 2.0),
            "A1.5": ("runs/speedcond/rootmotion_flow_speed_cpu_s{}/best.pt", 1.5),
            "A3": ("runs/speedcond/rootmotion_flow_speed_cpu_s{}/best.pt", 3.0),
            "B1": ("runs/phase_f/B1_s{}/best.pt", 1.0), "B2": ("runs/phase_f/B2_s{}/best.pt", 1.0),
            "C5": ("runs/phase_f/C5_s{}/best.pt", 1.0)}
out = {}
for name in sys.argv[1:]:
    pattern, w = VARIANTS[name]
    out[name] = {}
    for route in (Route("straight_12/aligned/walk", SHAPES["straight_12"], 0.0, "walk"), unity_route()):
        rows = []
        for s in SEEDS:
            planner = RootMotionPlanner.from_checkpoint(pattern.format(s), PlannerConfig(candidates=16, speed_guidance=w))
            for o in OFFSETS:
                fn = model_plan_function(planner, torch.Generator().manual_seed(route_seed(route) + o))
                r = simulate(route, fn, gait_speed=0.9, request_extras={"target_speed": 0.9})
                m = rollout_metrics(r)
                speed = np.linalg.norm(np.diff(r.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
                end = int(round(m["arrival_time"] * CONTROL_HZ)) if m["arrival_time"] is not None else len(speed)
                hit = np.flatnonzero(speed[:end] >= 0.81)
                rows.append({"seed": s, "offset": o, "to_081": hit[0] / CONTROL_HZ if len(hit) else None,
                             "mean": m["follow_mean_speed"], "arrival": m["arrival_time"]})
        t = [r["to_081"] for r in rows]
        ok = [x for x in t if x is not None]
        per_ckpt = [st.median([r["to_081"] if r["to_081"] is not None else 99 for r in rows if r["seed"] == s]) for s in SEEDS]
        out[name][route.name] = rows
        print(f"{name} {route.name}: median {st.median([x if x is not None else 99 for x in t]):.1f} s, "
              f"mean {st.fmean(ok):.1f}, in band {sum(1.35 <= x <= 4.8 for x in ok)}/{len(t)}, below 1.35 "
              f"{sum(x < 1.35 for x in ok)}, never {len(t) - len(ok)}; per-ckpt medians {per_ckpt}; "
              f"mean speed {st.fmean(r['mean'] for r in rows):.2f}", flush=True)
Path(f"results/phase_f/resample_{'_'.join(sys.argv[1:])}.json").write_text(json.dumps(out), encoding="utf-8")

"""Closed-loop replans binned by history end speed: plan speed in its 1st / 2nd / 3rd second.

Only replans with more than 5 m of path left; 12 m and Unity routes, 5 checkpoints x 4 sampling seeds.
Produces: results/phase_f/replan_bins_NAME.json (compared with f_openloop_ask09.py in the T9 notes).
Checkpoints: base -> runs/speedcond/rootmotion_flow_speed_cpu_s*/best.pt; C2 -> runs/phase_f/C2_s*/best.pt.
Run from the repository root with the project venv:
  python scripts/rootmotion/f_replan_bins.py base
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

from gtp.rootmotion import CONTROL_HZ
from gtp.rootmotion.closed_loop import SHAPES, Route, model_plan_function, route_seed, simulate, unity_route
from gtp.rootmotion.paths import arclength
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.request import build_batch
from gtp.rootmotion.windows import POSITION_SCALE

VARIANTS = {"base": "runs/speedcond/rootmotion_flow_speed_cpu_s{}/best.pt", "C2": "runs/phase_f/C2_s{}/best.pt"}
SEEDS, OFFSETS = [7, 11, 13, 17, 19], [0, 1, 2, 3]
BINS = {"< 0.3": (0, 0.3), "0.3-0.6": (0.3, 0.6), "> 0.6": (0.6, 99)}


def recording(fn, log):
    def plan(request, previous=None):
        deltas, source, details = fn(request, previous)
        batch, _ = build_batch(request)
        end = float(torch.linalg.vector_norm(batch["history"][0, -1, 4:6]) * POSITION_SCALE)
        path = np.array([[p[0], p[2]] for p in request["path"]])
        speed = np.linalg.norm(deltas[:, :2], axis=1) * CONTROL_HZ
        if arclength(path)[-1] > 5.0:
            log.append((end, [float(speed[10 * k:10 * (k + 1)].mean()) for k in range(3)]))
        return deltas, source, details
    return plan


name = sys.argv[1]
out = {}
for route in (Route("straight_12/aligned/walk", SHAPES["straight_12"], 0.0, "walk"), unity_route()):
    log = []
    for s in SEEDS:
        planner = RootMotionPlanner.from_checkpoint(VARIANTS[name].format(s), PlannerConfig(candidates=16))
        for o in OFFSETS:
            fn = model_plan_function(planner, torch.Generator().manual_seed(route_seed(route) + o))
            simulate(route, recording(fn, log), gait_speed=0.9, request_extras={"target_speed": 0.9})
    out[route.name] = {}
    for label, (lo, hi) in BINS.items():
        rows = [v for e, v in log if lo <= e < hi]
        if rows:
            means = [st.fmean(r[k] for r in rows) for k in range(3)]
            out[route.name][label] = {"n": len(rows), "seconds": means}
            print(f"{name} {route.name} {label}: n={len(rows)} " + " / ".join(f"{m:.2f}" for m in means), flush=True)
Path(f"results/phase_f/replan_bins_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

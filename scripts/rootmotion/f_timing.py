"""Per-request planning time for each phase-f variant (100 timed requests after 10 warm-up).

Produces: results/phase_f/timing.json (P50 / P95). Superseded for the paper: it runs on the CPU
(from_checkpoint's default device) and passes no previous plan, so consistent selection is
not exercised. See T10 and latency.py.
Checkpoints: runs/speedcond/rootmotion_flow_speed_cpu_s7/best.pt, runs/phase_f/*_s7/best.pt.
Run from the repository root with the project venv:
  python scripts/rootmotion/f_timing.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.request import build_batch

VARIANTS = {"base": ("runs/speedcond/rootmotion_flow_speed_cpu_s7/best.pt", 1.0),
            "A1.5": ("runs/speedcond/rootmotion_flow_speed_cpu_s7/best.pt", 1.5),
            "A2": ("runs/speedcond/rootmotion_flow_speed_cpu_s7/best.pt", 2.0),
            "A3": ("runs/speedcond/rootmotion_flow_speed_cpu_s7/best.pt", 3.0),
            "B1": ("runs/phase_f/B1_s7/best.pt", 1.0), "B2": ("runs/phase_f/B2_s7/best.pt", 1.0),
            "C5": ("runs/phase_f/C5_s7/best.pt", 1.0), "C2": ("runs/phase_f/C2_s7/best.pt", 1.0),
            "C2+A2": ("runs/phase_f/C2_s7/best.pt", 2.0)}
batch, _ = build_batch({"history": [{"position": [0, 0, 0], "yaw": 0}], "path": [[0, 0, 0], [0, 0, 12]],
                        "gait": "walk", "target_speed": 0.9})
out = {}
for name, (path, w) in VARIANTS.items():
    planner = RootMotionPlanner.from_checkpoint(path, PlannerConfig(candidates=16, speed_guidance=w))
    times = []
    for i in range(110):
        started = time.perf_counter()
        planner.plan_batch(batch, torch.Generator().manual_seed(i))
        times.append((time.perf_counter() - started) * 1000)
    t = np.array(times[10:])
    out[name] = {"p50": float(np.median(t)), "p95": float(np.percentile(t, 95))}
    print(f"{name}: p50 {out[name]['p50']:.1f} ms, p95 {out[name]['p95']:.1f} ms", flush=True)
Path("results/phase_f/timing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

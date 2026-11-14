"""Single-request planning latency for the paper's method and the regressor (T10).

Times one /plan request at a time through the service's own code path, in process (no HTTP):
PlanService.plan = build_batch -> previous_plan -> plan_batch -> build_plan. The request is a
walk at 0.9 m/s, mid-route, with a 20-pose moving history, a 12 m route with one corner and the
unexecuted 2 s of the previous plan, so flow's consistent selection is exercised (the script
checks that previous_plan_used is true on every flow request).

  flow       speed-conditioned flow, consistent selection, K = 16
  regressor  speed-conditioned regressor, one candidate

Each method: 20 warm-up requests, then N timed requests (default 200), a new sampling seed each.
Reports P50 / P95 / mean / max of the wall-clock time around PlanService.plan and of the
service's own latency_ms (plan_batch + build_plan), with the hardware and software versions.
Run only on an otherwise idle machine (no Unity, no training).

Produces: results/latency/latency_DEVICE.json and a Markdown table on stdout -> T10.
Checkpoints: runs/speedcond/rootmotion_flow_speed_cpu_s7/best.pt and
  runs/speedcond/rootmotion_regressor_speed_cpu_s7/best.pt (train_speedcond.py).
Run from the repository root with the project venv:
  python scripts/rootmotion/latency.py                 # CPU, 200 requests each
  python scripts/rootmotion/latency.py --device cuda   # optional, if a GPU service is planned
"""
import argparse
import json
import os
import platform
import statistics as st
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.service import PlanService

CHECKPOINTS = {
    "flow": ("runs/speedcond/rootmotion_flow_speed_cpu_s7/best.pt", PlannerConfig(candidates=16, selection="consistent")),
    "regressor": ("runs/speedcond/rootmotion_regressor_speed_cpu_s7/best.pt",
                  PlannerConfig(candidates=1, selection="goal_aware")),
}
WARMUP = 20


def request(previous=None, seed=0):
    """Mid-walk at 0.9 m/s heading +z, 12 m route with a right turn after 6 m."""
    history = [{"position": [0.0, 0.0, 0.09 * (k - 19)], "yaw": 0.0} for k in range(20)]
    out = {"history": history, "path": [[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [6.0, 0.0, 6.0]],
           "path_ends_at_goal": True, "gait": "walk", "target_speed": 0.9, "seed": seed}
    if previous is not None:
        out["previous_plan"] = previous
    return out


def cpu_name():
    if platform.system() == "Windows":
        try:
            text = subprocess.run(["powershell", "-NoProfile", "-Command",
                                   "(Get-CimInstance Win32_Processor).Name"],
                                  capture_output=True, text=True, timeout=20).stdout.strip()
            if text:
                return text
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def hardware(device):
    info = {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "device": str(device)}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def summary(values):
    a = np.asarray(values)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean()), "max": float(a.max()), "n": int(len(a))}


def measure(name, device, n, warmup):
    path, config = CHECKPOINTS[name]
    service = PlanService(RootMotionPlanner.from_checkpoint(path, config, device))
    first = service.plan(request(seed=0))
    # The character has executed 1 s (10 steps) of the first plan; send the remaining 2 s.
    previous = [p["position"] for p in first["trajectory"][11:]]
    wall, served, used = [], [], []
    for i in range(warmup + n):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        plan = service.plan(request(previous, seed=i + 1))
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        if i >= warmup:
            wall.append(elapsed)
            served.append(plan["latency_ms"])
            used.append(bool(plan.get("previous_plan_used")))
    if name == "flow" and not all(used):
        raise SystemExit("flow: previous_plan was not used on every request; consistent selection not timed")
    return {"checkpoint": path, "candidates": config.candidates, "selection": config.selection,
            "previous_plan_used": st.fmean(used), "wall_ms": summary(wall), "service_latency_ms": summary(served)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--output", type=Path, default=None, help="default results/latency/latency_DEVICE.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    result = {"date": time.strftime("%Y-%m-%d %H:%M"), "hardware": hardware(device), "warmup": args.warmup,
              "methods": {name: measure(name, device, args.requests, args.warmup) for name in CHECKPOINTS}}
    out = args.output or Path("results/latency") / f"latency_{device.type}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["hardware"], indent=1))
    print("\n| Method | Requests | Wall P50 ms | Wall P95 ms | Service latency_ms P50 / P95 |")
    print("|---|---|---|---|---|")
    for name, m in result["methods"].items():
        w, s = m["wall_ms"], m["service_latency_ms"]
        print(f"| {name} (K = {m['candidates']}, {m['selection']}) | {w['n']} | {w['p50']:.1f} | {w['p95']:.1f} | "
              f"{s['p50']:.1f} / {s['p95']:.1f} |")


if __name__ == "__main__":
    main()

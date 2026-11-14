"""Open-loop speed by history end speed (test windows, walk, label >= 0.8), each window's own label.

Produces: results/phase_f/openloop_check.json and a table on stdout. Caution: these windows
carry labels of 1.2 to 1.6 m/s, so this table is not comparable with closed loop at 0.9 m/s
(see "Superseded claims" in docs/ROOT_MOTION_RESULTS.md); f_openloop_ask09.py is the comparable check.
Checkpoints: runs/speedcond/rootmotion_{flow,regressor}_speed_cpu_s{7,11,13,17,19}/best.pt.
Run from the repository root with the project venv:
  python scripts/rootmotion/f_openloop_check.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from gtp.rootmotion import CONTROL_HZ, GAITS
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.train import load_splits
from gtp.rootmotion.windows import POSITION_SCALE

SEEDS = [7, 11, 13, 17, 19]
PM = " ± "
config = yaml.safe_load(Path("configs/rootmotion_flow_speed_cpu.yaml").read_text(encoding="utf-8"))
test = load_splits(config)["test"].as_batch()
walk = (test["gait"] == GAITS.index("walk")) & (test["target_speed"] >= 0.8)
end_speed = torch.linalg.vector_norm(test["history"][:, -1, 4:6], dim=-1) * POSITION_SCALE
bins = {"< 0.3": end_speed < 0.3, "0.3-0.6": (end_speed >= 0.3) & (end_speed < 0.6), "> 0.6": end_speed >= 0.6}
capture_speed = torch.linalg.vector_norm(test["target"][..., :2], dim=-1) * CONTROL_HZ  # [N, 30]


def per_second(speed):
    return [float(speed[..., 10 * k:10 * (k + 1)].mean()) for k in range(3)]


def ms(v):
    return f"{st.fmean(v):.2f}" + (PM + f"{st.stdev(v):.2f}" if len(v) > 1 else "")


variants = {"flow (selected plan)": ("runs/speedcond/rootmotion_flow_speed_cpu_s{}/best.pt", 16, "selected"),
            "flow (mean of 16 candidates)": ("runs/speedcond/rootmotion_flow_speed_cpu_s{}/best.pt", 16, "all"),
            "regressor": ("runs/speedcond/rootmotion_regressor_speed_cpu_s{}/best.pt", 1, "selected")}
if len(sys.argv) > 1:  # extra variants: name=path_pattern
    for arg in sys.argv[1:]:
        name, pattern = arg.split("=", 1)
        variants[f"{name} (selected plan)"] = (pattern, 16, "selected")

print("| history end speed | n | capture 1st / 2nd / 3rd s | " + " | ".join(variants) + " |")
print("|---|---|---|" + "---|" * len(variants))
rows = {}
for label, mask in bins.items():
    m = walk & mask
    idx = m.nonzero(as_tuple=True)[0]
    capture = per_second(capture_speed[idx])
    cells = []
    for name, (pattern, k, mode) in variants.items():
        seconds = [[], [], []]
        for seed in SEEDS:
            planner = RootMotionPlanner.from_checkpoint(pattern.format(seed), PlannerConfig(candidates=k))
            batch = {key: value[idx] for key, value in test.items() if key != "target"}
            result = planner.plan_batch(batch, torch.Generator().manual_seed(0))
            if mode == "all":
                speed = torch.linalg.vector_norm(result["candidates"][..., :2], dim=-1) * CONTROL_HZ
            else:
                speed = torch.linalg.vector_norm(result["deltas"][..., :2], dim=-1) * CONTROL_HZ
            for k2, value in enumerate(per_second(speed)):
                seconds[k2].append(value)
        cells.append(" / ".join(ms(s) for s in seconds))
        rows.setdefault(name, {})[label] = seconds
    print(f"| {label} | {len(idx)} | {' / '.join(f'{c:.2f}' for c in capture)} | " + " | ".join(cells) + " |")
    rows.setdefault("capture", {})[label] = capture
Path("results/phase_f").mkdir(parents=True, exist_ok=True)
Path("results/phase_f/openloop_check.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

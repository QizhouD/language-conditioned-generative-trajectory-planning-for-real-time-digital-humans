"""Open-loop check with every window's target_speed set to 0.9 (as in the synthetic closed loop).

Produces: results/phase_f/openloop_ask09.json (mean plan speed per second, by history end speed).
  Source of the "near fixed point" evidence in the plateau discussion (T9 notes).
Checkpoints: runs/speedcond/rootmotion_flow_speed_cpu_s{7,11,13,17,19}/best.pt.
Run from the repository root with the project venv:
  python scripts/rootmotion/f_openloop_ask09.py
"""
import json, statistics as st
from pathlib import Path
import torch, yaml
from gtp.rootmotion import CONTROL_HZ, GAITS
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner
from gtp.rootmotion.train import load_splits
from gtp.rootmotion.windows import POSITION_SCALE
config = yaml.safe_load(Path("configs/rootmotion_flow_speed_cpu.yaml").read_text(encoding="utf-8"))
test = load_splits(config)["test"].as_batch()
walk = (test["gait"] == GAITS.index("walk")) & (test["target_speed"] >= 0.8)
end = torch.linalg.vector_norm(test["history"][:, -1, 4:6], dim=-1) * POSITION_SCALE
print("label mean per bin:")
out = {}
for label, (lo, hi) in {"< 0.3": (0, .3), "0.3-0.6": (.3, .6), "> 0.6": (.6, 99)}.items():
    idx = (walk & (end >= lo) & (end < hi)).nonzero(as_tuple=True)[0]
    secs = [[], [], []]
    for seed in [7, 11, 13, 17, 19]:
        p = RootMotionPlanner.from_checkpoint(f"runs/speedcond/rootmotion_flow_speed_cpu_s{seed}/best.pt", PlannerConfig(candidates=16))
        b = {k: v[idx] for k, v in test.items() if k != "target"}
        b["target_speed"] = torch.full_like(b["target_speed"], 0.9)
        sp = torch.linalg.vector_norm(p.plan_batch(b, torch.Generator().manual_seed(0))["deltas"][..., :2], dim=-1) * CONTROL_HZ
        for k in range(3):
            secs[k].append(float(sp[:, 10 * k:10 * (k + 1)].mean()))
    out[label] = {"n": len(idx), "label_mean": float(test["target_speed"][idx].mean()), "seconds_ask_0.9": [st.fmean(s) for s in secs]}
    print(label, out[label])
Path("results/phase_f/openloop_ask09.json").write_text(json.dumps(out, indent=1))

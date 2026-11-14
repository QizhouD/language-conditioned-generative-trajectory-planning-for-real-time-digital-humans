"""(1) Does the 0.4 m look-ahead floor bind? Each fold's selected setting with min_lookahead 0.4
(default) vs 0.2 vs 0.1, on that fold's training walks. (2) Scale-only spring (default
constants, scale 0.8) on all walks.
Produces: results/phase_g/checks.json (T1 notes: floor check and scale-only spring).
Checkpoints: none. Needs results/phase_g/selection.json (g_eval.py).
Run from the repository root with the project venv:
  python scripts/rootmotion/g_checks.py
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g_common import capture_entry, objective_terms, run_spring, walks
from f_table import SHORT, w1

def job(args):
    params, idx = args
    w = walks(); out = []
    for i in idx:
        seg = w[i][1]; cap = capture_entry(seg)
        rec = run_spring(seg, params)
        rmse, dev = objective_terms(rec, cap)
        out.append((i, rmse / cap["mean"], dev, rec["arrival_time"] / cap["arrival_time"], rec["kin"], cap["kin"], cap["length"]))
    return out

if __name__ == "__main__":
    sel = json.loads(Path("results/phase_g/selection.json").read_text())
    subjects = [s for s, _ in walks()]
    jobs, keys = [], []
    for h in range(1, 6):
        train = [i for i, s in enumerate(subjects) if s != int(h)]
        for floor in (0.4, 0.2, 0.1):
            p = dict(sel[str(h)]["params"]); p["min_lookahead"] = floor
            jobs.append((p, train)); keys.append((h, floor))
    jobs.append(({"speed_scale": 0.8}, list(range(len(subjects))))); keys.append(("scale0.8", None))
    with Pool(14) as pool:
        res = pool.map(job, jobs)
    out = {}
    for (h, floor), rows in zip(keys, res):
        if h == "scale0.8":
            short = [r for r in rows if r[6] <= SHORT]
            arr = float(np.mean([r[3] for r in short]))
            human = {k: np.concatenate([r[5][k] for r in short]) for k in ("speed", "acceleration", "turn_rate")}
            model = {k: np.concatenate([r[4][k] for r in short]) for k in ("speed", "acceleration", "turn_rate")}
            out["scale0.8_short"] = {"arrival_ratio": arr, **{f"w1_{k}": w1(model[k], human[k]) for k in human}}
            print("scale-only 0.8, short walks pooled:", out["scale0.8_short"])
        else:
            obj = float(np.mean([r[1] for r in rows]) + np.mean([r[2] for r in rows]))
            out[f"fold{h}_floor{floor}"] = obj
            print(f"fold {h} min_lookahead {floor}: train objective {obj:.4f}")
    Path("results/phase_g/checks.json").write_text(json.dumps(out, indent=1))

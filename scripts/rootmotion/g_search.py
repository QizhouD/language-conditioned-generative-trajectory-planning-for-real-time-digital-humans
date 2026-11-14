"""Random search for the spring's parameters; selection per LOAO fold on training actors only.

Stage 1: 400 random configurations + the default, each run on all 73 walks (scores stored per walk).
Stage 2: per fold, 80 multiplicative perturbations around that fold's stage-1 best.
A fold's choice uses only the walks of its four training actors (made in g_eval.py).
Objective J = mean over walks [ profile RMSE / capture mean speed + |arrival / capture - 1| ].
Produces: results/phase_g/search.json -> T1 (through g_eval.py).
Checkpoints: none (the spring has no learned parameters). 14 worker processes.
Run from the repository root with the project venv:
  python scripts/rootmotion/g_search.py
"""
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g_common import capture_entry, objective_terms, run_spring, walks  # noqa: E402

RANGES = {  # name: (low, high, log scale)
    "speed_scale": (0.5, 1.1, False),
    "halflife": (0.1, 2.0, True),
    "deceleration": (0.2, 4.0, True),
    "arrival_seconds": (0.3, 2.0, False),
    "lookahead_seconds": (0.05, 1.5, False),
}
DEFAULT = {"speed_scale": 1.0, "halflife": 0.27, "deceleration": 3.0, "arrival_seconds": 0.6, "lookahead_seconds": 0.5}
OUT = Path("results/phase_g")
_CAPS = None


def sample(rng):
    out = {}
    for k, (lo, hi, log) in RANGES.items():
        out[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if log else float(rng.uniform(lo, hi))
    return out


def perturb(base, rng):
    out = {}
    for k, (lo, hi, _) in RANGES.items():
        out[k] = float(np.clip(base[k] * math.exp(rng.normal(0.0, 0.15)), lo, hi))
    return out


def score(params):
    global _CAPS
    w = walks()
    if _CAPS is None:
        _CAPS = [capture_entry(seg) for _, seg in w]
    rows = []
    for (subject, seg), cap in zip(w, _CAPS):
        rmse, dev = objective_terms(run_spring(seg, params, keep_kin=False), cap)
        rows.append([rmse / cap["mean"], dev])
    return rows


def fold_objective(rows, subjects, hold):
    keep = [i for i, s in enumerate(subjects) if s != hold]
    r = np.asarray(rows)[keep]
    return float(r[:, 0].mean() + r[:, 1].mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = [s for s, _ in walks()]
    rng = np.random.default_rng(0)
    configs = [DEFAULT] + [sample(rng) for _ in range(400)]
    started = time.time()
    with Pool(14) as pool:
        scores = pool.map(score, configs, chunksize=4)
        print(f"stage 1: {len(configs)} configs in {time.time() - started:.0f}s", flush=True)
        stage2 = []
        for hold in range(1, 6):
            best = min(range(len(configs)), key=lambda i: fold_objective(scores[i], subjects, hold))
            local = np.random.default_rng(100 + hold)
            stage2 += [perturb(configs[best], local) for _ in range(80)]
        scores += pool.map(score, stage2, chunksize=4)
        configs += stage2
    print(f"stage 2 done in {time.time() - started:.0f}s", flush=True)
    folds = {}
    for hold in range(1, 6):
        objectives = [fold_objective(s, subjects, hold) for s in scores]
        best = int(np.argmin(objectives))
        rows = np.asarray(scores[best])
        train = [i for i, s in enumerate(subjects) if s != hold]
        folds[hold] = {"params": configs[best], "index": best, "objective": objectives[best],
                       "default_objective": objectives[0],
                       "train_rmse_rel": float(rows[train, 0].mean()), "train_arrival_dev": float(rows[train, 1].mean())}
        print(hold, json.dumps(folds[hold]), flush=True)
    (OUT / "search.json").write_text(json.dumps({"ranges": RANGES, "default": DEFAULT, "configs": configs,
                                                "scores": scores, "subjects": subjects, "folds": folds}),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()

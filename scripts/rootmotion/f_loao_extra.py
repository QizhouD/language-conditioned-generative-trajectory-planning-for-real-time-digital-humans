"""LOAO: per-fold seed std of the short-walk arrival ratio, and ratios pooled over all walks.

Produces: stdout only (pooled short-walk arrival / capture: flow 1.09, 2-step 1.05, spring 0.73).
Checkpoints: none (reads results/phase_f/loao/*.json).
Run from the repository root with the project venv:
  python scripts/rootmotion/f_loao_extra.py
"""
import json, statistics as st, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from f_table import SHORT, walk_block
D = Path("results/phase_f/loao")
for label, keep in (("short", lambda L: L <= SHORT), ("long", lambda L: L > SHORT)):
    print(f"## {label}")
    for n in ("spring", "base", "C2"):
        pooled = {"walks": {}}
        per = []
        for h in range(1, 6):
            d = json.loads((D / f"{n}_hold{h}.json").read_text())
            pooled["walks"].update(d["walks"])
            b = walk_block(d, keep)
            per.append(f"{h}: {np.nanmean(b['arrival/human']):.2f}±{np.nanstd(b['arrival/human'], ddof=1) if len(b['arrival/human'])>1 else 0:.2f}")
        b = walk_block(pooled, keep)
        print(n, "pooled n", b["n"], "arrival/human", f"{np.nanmean(b['arrival/human']):.2f} (seed std {np.nanstd(b['arrival/human'], ddof=1) if len(b['arrival/human'])>1 else 0:.2f})",
              "mean/human", f"{np.nanmean(b['mean/human']):.2f}", "|", "; ".join(per))

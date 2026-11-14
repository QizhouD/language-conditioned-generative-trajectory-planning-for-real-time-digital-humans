"""LOAO tables: per held-out actor and mean ± std across folds, short / long separate.

Produces: the phase-f LOAO tables (base, C2, untuned spring) on stdout and
results/phase_f/loao/summary.json. For the paper, g_stats.py supersedes it (T2 to T4).
Checkpoints: none (reads results/phase_f/loao/*.json).
Run from the repository root with the project venv:
  python scripts/rootmotion/f_loao_table.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from f_table import SHORT, walk_block  # noqa: E402

DIR = Path("results/phase_f/loao")
NAMES = ["spring", "base", "C2"]
COLS = ["arrival/human", "mean/human", "profile RMSE", "W1 speed", "W1 accel", "W1 turn", "seam accel", "other accel",
        "final dist", "max dev", "fallback", "arrived"]
PM = " ± "


def fmt(v, d=2):
    v = [x for x in v if np.isfinite(x)]
    return "–" if not v else f"{st.fmean(v):.{d}f}" + (PM + f"{st.stdev(v):.{d}f}" if len(v) > 1 else "")


summary = {}
for label, keep in (("short", lambda L: L <= SHORT), ("long", lambda L: L > SHORT)):
    print(f"\n### LOAO, {label} walks (≤ {SHORT:.0f} m)" if label == "short" else f"\n### LOAO, {label} walks (> {SHORT:.0f} m)")
    print("\nPer held-out actor (mean over that fold's seeds):\n")
    print("| actor | n | " + " | ".join(f"{n} arrival/human" for n in NAMES) + " | "
          + " | ".join(f"{n} mean/human" for n in NAMES) + " |")
    print("|---|---|" + "---|" * (2 * len(NAMES)))
    folds = {n: {} for n in NAMES}
    for hold in range(1, 6):
        blocks = {}
        for n in NAMES:
            path = DIR / f"{n}_hold{hold}.json"
            if path.exists():
                b = walk_block(json.loads(path.read_text()), keep)
                if b:
                    blocks[n] = b
                    folds[n][hold] = {c: float(np.nanmean(b[c])) for c in COLS}
        if not blocks:
            continue
        count = next(iter(blocks.values()))["n"]
        cell = lambda n, c: f"{np.nanmean(blocks[n][c]):.2f}" if n in blocks else "–"
        print(f"| {hold} | {count} | " + " | ".join(cell(n, "arrival/human") for n in NAMES) + " | "
              + " | ".join(cell(n, "mean/human") for n in NAMES) + " |")
    print("\nAcross folds (mean ± std of the per-fold means):\n")
    print("| method | folds | " + " | ".join(COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for n in NAMES:
        if not folds[n]:
            continue
        cells = [fmt([f[c] for f in folds[n].values()], 3 if c.startswith("W1") or c == "fallback" else 2) for c in COLS]
        print(f"| {n} | {len(folds[n])} | " + " | ".join(cells) + " |")
    summary[label] = folds
(DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

# Root-motion evaluation scripts

These scripts produce every table in [`docs/ROOT_MOTION_RESULTS.md`](../../docs/ROOT_MOTION_RESULTS.md)
(T1 to T10). Each file's docstring says which table it feeds, which checkpoints it reads and
how to run it.

General notes:

- Run everything from the repository root with the project venv.
- Paths such as `runs/...`, `results/...` and `data/...` are relative to the repository root.
- Data (`data/lafan1_root.npz`, LAFAN1, CC BY-NC-ND), checkpoints (`runs/`) and outputs
  (`results/`) stay local and are not committed.
- On a Windows console with a non-UTF-8 code page, set `PYTHONIOENCODING=utf-8`, because the
  tables print `±`, `−` and `✓`.

The prefixes match the result folders: `e_*` writes `results/phase_e`, `f_*` writes
`results/phase_f`, and `g_*` writes `results/phase_g`.

## Order

```bash
# 0. data (once): see docs/ROOT_MOTION_LAYER.md, "Data"
# 1. checkpoints
python scripts/rootmotion/train_speedcond.py          # standard split, 5 seeds   (T6 flow, T7, T8, T9)
python scripts/rootmotion/f_train.py loao base        # leave-one-actor-out, 3 seeds x 5 folds (T2-T5)
python scripts/rootmotion/f_train.py screen           # plateau variants (T9)
# 2. leave-one-actor-out closed loop, flow and untuned spring
for h in 1 2 3 4 5; do
  python scripts/rootmotion/f_loao_eval.py base $h
  python scripts/rootmotion/f_loao_eval.py spring $h
done
# 3. tuned spring
python scripts/rootmotion/g_search.py                 # T1 (search)
python scripts/rootmotion/g_eval.py                   # T1 (selection), tuned rows of T2-T6
python scripts/rootmotion/g_checks.py                 # T1 notes
python scripts/rootmotion/g_stats.py                  # T2, T3, T4, T5
# 4. standard split
python scripts/rootmotion/e_experiments.py segments
python scripts/rootmotion/e_experiments.py phase1
python scripts/rootmotion/e_experiments.py phase2
python scripts/rootmotion/e_analyze1.py               # T8 (synthetic seams)
python scripts/rootmotion/e_analyze2.py               # T8 (captured seams)
python scripts/rootmotion/e_checks.py                 # T8 (short walks), human reference of T6
python scripts/rootmotion/f_eval.py base spring A1.5 A2 A3 B1 B2 C5 C2 C2+A2
python scripts/rootmotion/f_table.py base A1.5 A2 A3 B1 B2 C5 C2 C2+A2 spring   # T7, T9
python scripts/rootmotion/f_resample.py base C2 A2 C2+A2 A1.5 A3 B1 B2 C5       # T6 flow, T9
python scripts/rootmotion/f_openloop_ask09.py         # T9 notes
python scripts/rootmotion/f_replan_bins.py base       # T9 notes
# 5. latency, on an idle machine only (no Unity, no training)
python scripts/rootmotion/latency.py                  # T10
```

## Index

| Script | Table(s) | Reads | Writes |
|---|---|---|---|
| `train_speedcond.py` | checkpoints for T6–T10 | data | `runs/speedcond/` |
| `f_train.py` | checkpoints for T2–T5, T9 | data | `runs/loao/`, `runs/phase_f/` |
| `f_loao_eval.py` | T2–T5 (flow, untuned spring) | `runs/loao/` | `results/phase_f/loao/*_hold*.json` |
| `g_common.py` | helper for `g_*` | data | – |
| `g_search.py` | T1 | data | `results/phase_g/search.json` |
| `g_eval.py` | T1, T2–T6 (tuned spring) | `search.json` | `selection.json`, `springT_hold*.json`, `springD_hold*.json`, `synthetic.json` |
| `g_checks.py` | T1 notes | `selection.json` | `checks.json` |
| `g_stats.py` | T2, T3, T4, T5 | LOAO and tuned-spring JSON | `results/phase_g/tables.md`, `paired.json` |
| `e_experiments.py` | T8 inputs | `runs/speedcond/` | `results/phase_e/*.json` |
| `e_analyze1.py` | T8 (synthetic seams) | `phase1.json` | stdout, `speed_time_*.png` |
| `e_analyze2.py` | T8 (captured seams) | `phase2.json` | stdout |
| `e_checks.py` | T8 (short walks), T6 human reference | `phase1.json`, `phase2.json`, data | stdout |
| `f_eval.py` | T7, T9 inputs | `runs/speedcond/`, `runs/phase_f/` | `results/phase_f/eval_*.json`, `open_*_s*.json` |
| `f_table.py` | T7, T9; helpers for `f_loao_*`, `g_*` | `eval_*.json` | stdout, `summary.json` |
| `f_resample.py` | T6 (flow), T9 (start times) | checkpoints | `results/phase_f/resample_*.json` |
| `f_openloop_ask09.py` | T9 notes | `runs/speedcond/` | `openloop_ask09.json` |
| `f_replan_bins.py` | T9 notes | checkpoints | `replan_bins_*.json` |
| `f_openloop_check.py` | superseded claim 3 (own-label check; not comparable) | `runs/speedcond/` | `openloop_check.json` |
| `f_loao_table.py`, `f_loao_extra.py` | earlier LOAO tables (superseded by `g_stats.py`) | `results/phase_f/loao/` | stdout, `loao/summary.json` |
| `f_timing.py` | earlier latency (superseded by `latency.py`) | checkpoints | `timing.json` |
| `latency.py` | T10 | `runs/speedcond/*_s7` | `results/latency/latency_DEVICE.json` |

All seeds are fixed in the scripts:

- Training seeds 7, 11, 13, 17, 19 (LOAO: 7, 11, 13).
- Sampling generators are seeded by route (`route_seed`).
- The search uses `default_rng(0)` and `default_rng(100 + fold)`.
- The bootstrap uses `default_rng(0)`.

Rerunning the JSON-to-table scripts (`g_stats.py`, `f_loao_table.py`, `f_table.py`,
`e_analyze1.py`, `e_analyze2.py`) on the existing result files reproduces the published tables
exactly.

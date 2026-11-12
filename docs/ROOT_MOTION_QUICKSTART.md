# Root-motion layer: quickstart

From a fresh checkout of `feat/root-motion-layer` to a running planner service.
Commands are PowerShell; on macOS or Linux replace the backtick line continuation
with a backslash. Background and results are in
[ROOT_MOTION_LAYER.md](ROOT_MOTION_LAYER.md).

## 0. Get the branch

If the branch is on GitHub:

```powershell
git fetch origin
git checkout feat/root-motion-layer
```

If you only have the patch file:

```powershell
git checkout main
git checkout -b feat/root-motion-layer
git am --3way path\to\gtp-feat-root-motion-layer.patch
```

## 1. Install

Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,plots]"
python -m pytest -q
```

Expected: `35 passed`. The tests need no dataset.

`pip` installs a CPU build of PyTorch on Windows. For GPU training install the CUDA
build first, following the selector on pytorch.org, then run the `pip install -e`
line. Training picks the GPU up automatically.

## 2. Get the data

```powershell
python -m gtp.rootmotion.fetch_lafan1 --output data/lafan1
python -m gtp.rootmotion.extract --bvh-dir data/lafan1 --output data/lafan1_root.npz
```

Expected: a 144 MB download, `77 BVH files in data\lafan1`, then
`data\lafan1_root.npz: 18 clips, 73.5 minutes`.

LAFAN1 is CC BY-NC-ND 4.0: research use only. `data/lafan1/` and `data/*.npz` are
git-ignored; keep them that way.

## 3. Check the pipeline end to end (under a minute)

```powershell
python -m gtp.rootmotion.train --config configs/rootmotion_flow_smoke.yaml
python -m gtp.rootmotion.evaluate --config configs/rootmotion_flow_smoke.yaml `
  --flow runs/rootmotion_flow_smoke/best.pt --latency-repetitions 5
```

This trains for one epoch, so the model is useless: expect a high fallback rate. The
point is that every stage runs and the table prints.

## 4. Train

| Config | Hardware | Time | Purpose |
|---|---|---|---|
| `rootmotion_flow_cpu.yaml` | 2 CPU cores | about 20 min | Reproduces the preliminary table |
| `rootmotion_regressor_cpu.yaml` | 2 CPU cores | about 20 min | Deterministic baseline for that table |
| `rootmotion_flow.yaml` | GPU | not yet measured | Full-size model for the paper |
| `rootmotion_regressor.yaml` | GPU | not yet measured | Full-size deterministic baseline |

```powershell
python -m gtp.rootmotion.train --config configs/rootmotion_flow_cpu.yaml
python -m gtp.rootmotion.train --config configs/rootmotion_regressor_cpu.yaml
```

Each run prints validation loss every few epochs and keeps the best checkpoint at
`runs/<experiment>/best.pt`. For another seed, copy the config and change `seed`,
`experiment` and `run_dir`.

## 5. Evaluate on the held-out actor

```powershell
python -m gtp.rootmotion.evaluate --config configs/rootmotion_flow_cpu.yaml `
  --flow runs/rootmotion_flow_cpu/best.pt `
  --regressor runs/rootmotion_regressor_cpu/best.pt `
  --output results/rootmotion_cpu_test.json
```

Prints a table and writes the full metrics as JSON. With the CPU configs and seed 7
the flow row should land near ADE 0.45 m, minADE 0.23 m, fallback under 1%. Exact
values vary slightly across PyTorch versions and hardware.

Useful flags: `--split val`, `--candidates 32`, `--flow-steps 4`, `--half-width 0.4`.

To look at samples:

```powershell
python -m gtp.rootmotion.visualize --config configs/rootmotion_flow_cpu.yaml `
  --checkpoint runs/rootmotion_flow_cpu/best.pt --output results/windows.png
```

## 6. Run the planner service

```powershell
python -m gtp.rootmotion.service --checkpoint runs/rootmotion_flow_cpu/best.pt --port 8787
```

In a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/health

$body = @{
  history = @(@{ position = @(10.0, 0.0, 20.0); yaw = 0.0 })
  path = @(@(10.0, 0.0, 20.0), @(10.0, 0.0, 23.0), @(13.0, 0.0, 23.0))
  path_ends_at_goal = $true
  gait = "walk"
} | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8787/plan -Method Post -ContentType "application/json" -Body $body |
  Select-Object source, rejected_candidates, candidates, latency_ms
```

Expected: `source` is `model` for a trained checkpoint (`spring_fallback` means every
sample was rejected). Leave the service running for Unity.

## 7. Connect Unity

In the LLM-NPC repo, check out its `feat/root-motion-layer` branch and follow
`docs/root-motion-layer.md` there: first-run checklist, scene wiring, and the logging
protocol.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named gtp` | The virtual environment is not active, or `pip install -e` was skipped. |
| `checksum mismatch` from `fetch_lafan1` | Interrupted download. Delete `data\lafan1\lafan1.zip` and rerun. |
| Download blocked by a proxy | Download `lafan1/lafan1.zip` from the Ubisoft repository by hand into `data\lafan1\` and rerun; the script skips the download when the file exists. |
| `no clips for gaits` from `extract` | `--bvh-dir` does not point at the folder holding the `.bvh` files. |
| Training is very slow | CPU build of PyTorch with a GPU config. Use a `_cpu` config or install the CUDA build. |
| Service answers `spring_fallback` every time | The checkpoint is undertrained (for example the smoke run). |
| Port already in use | Pass another `--port` and set the same URL on `HttpTrajectoryPlanSource`. |

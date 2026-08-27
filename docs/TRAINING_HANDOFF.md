# Training Handoff: 2D Deterministic Trajectory Baseline

**Date:** 2026-08-22  
**Status:** HOLD (no new training unless explicitly requested by Jim / Qizhou Deng)  
**Contact:** Jim / Qizhou Deng

## Executive Summary

This document describes the experimental 2D trajectory loss functions tested on the deterministic transformer baseline. All runs used CPU-only training on `synthetic_navigation.jsonl` (1000 episodes, seed 7) with a small transformer (128 hidden dim, 3 layers, 4 heads, 30-step horizon, 8-D trajectory tokens).

**Two faculty-approved figures** are preserved for documentation:
1. **Collision avoidance** (`trajectory-baseline-2d-collide-pred-vs-gt.png`): Demonstrates obstacle detours.
2. **Empty-scene path** (`trajectory-baseline-2d-path-pred-vs-gt.png`): Demonstrates empty-field trajectory learning.

**Training is currently FROZEN.** Do not start new runs or retune `path_occupied` weights without explicit approval.

## Faculty Assets

The following two figures are the official results for documentation:

### 1. Collision Avoidance (digital_human_baseline_2d_collide)
- **File:** `docs/training-figures/trajectory-baseline-2d-collide-pred-vs-gt.png`
- **Config:** `configs/digital_human_baseline_2d_collide.yaml`
- **Loss:** 2D MSE + second-difference smoothness + circular overlap collision (weight 2.0, clearance 0.3)
- **Metrics:** ADE 0.766, FDE 0.122, collision rate 0.321
- **Caption:** Obstacles are avoided and endpoints reached on the shown detours. Episodes 17 and 128 demonstrate wide arcs around obstacles. Empty-field paths can still wobble.

### 2. Empty-Scene Path (digital_human_baseline_2d_path)
- **File:** `docs/training-figures/trajectory-baseline-2d-path-pred-vs-gt.png`
- **Config:** `configs/digital_human_baseline_2d_path.yaml`
- **Loss:** Collision avoidance + cumulative path MSE on empty scenes only (dummy-obstacle aware)
- **Caption:** Empty scenes (episodes 0 and 42) show straighter paths after adding empty-scene cumulative path supervision.

**Note:** The other two figures (`trajectory-baseline-2d-combo-pred-vs-gt.png` and `trajectory-baseline-2d-collide-empty-pred-vs-gt.png`) are contrast-only and demonstrate failed experiments where path supervision on occupied scenes pulled predictions through obstacles.

## Chronological Training Results

All runs: seed 7, `synthetic_navigation.jsonl`, 40 epochs (except where noted), batch size 64, learning rate 0.0003, AdamW optimizer, gradient clipping 1.0, validation fraction 0.15.

| Run | Config | Loss Components | Best Val Loss | ADE | FDE | Collision Rate | Notes |
|-----|--------|----------------|---------------|-----|-----|----------------|-------|
| 1 | CFM 40 epochs | Flow matching baseline | 6.92 → 1.33 | - | - | - | Zigzag 2D roots, not faculty |
| 2 | CFM 80 epochs | Flow matching (80 epochs) | 1.07 (epoch 75) | 2.70 | 0.64 | - | Still zigzag, stopped |
| 3 | Deterministic 8-D | 8-D MSE + action + goal + smoothness | - | - | - | - | Scribble paths, figure saved |
| 4 | 2D-only | 2D MSE + first-order smoothness | - | - | - | - | Still scribble |
| 5 | 2D smooth | 2D MSE + second-diff smoothness | - | - | - | - | Zigzag partly damped |
| 6 | **2D collide (FACULTY)** | 2D + second-diff + collision (2.0, clearance 0.3) | 0.11933 | **0.766** | **0.122** | **0.321** | First avoidance figure, eps 17/128 detour |
| 7 | **2D path (FACULTY)** | Collide + empty path MSE | - | - | - | - | Empty scenes 0/42 straighter |
| 8 | 2D combo | Path on occupied + clearance + empty boost | 0.47751 | 0.283 | 0.092 | 0.684 | Ep 17 cuts through disk (FAILED) |
| 9 | 2D combo avoid | Collision 5.0, path_occupied 0.4 | 0.28793 (epoch 37) | 0.358 | 0.140 | 0.687 | Ep 17/128 still clip (FAILED) |
| 10 | 2D empty path | path_occupied=0, clearance kind | 0.12562 | 0.743 | 0.055 | 0.381 | Ep 17 still through disk |
| 11 | 2D collide-empty | Overlap + empty-only path | 0.12399 | 0.758 | 0.084 | 0.362 | Wide arcs but official mask still True |

**Diagnostic episodes:**
- **Episode 17:** Large circular obstacle between start and goal
- **Episode 128:** Cluster of multiple obstacles
- **Episodes 0 and 42:** Empty scenes (no real obstacles, only dummy padding)

## Dummy-Obstacle Behavior

The `collate_trajectories` function in `dataset.py` pads empty scenes with a dummy obstacle at `(1e6, 1e6)` to maintain uniform batch shapes. This causes `obstacle_valid` to be `True` even when the scene has no real obstacles.

**CRITICAL:** Empty-scene detection MUST treat a scene as empty when all obstacle centers have `norm > 1e5`. The path loss and smoothness boost logic correctly ignores the dummy obstacle using this threshold.

Do not modify this behavior in `dataset.py` or the loss computation without updating both locations consistently.

## Loss Function Reference

The experimental 2D loss flags have been ported to `src/gtp/train.py`. The original 8-D CFM training path remains functional.

### New Flags for DeterministicTrajectoryBaseline

All flags are optional and can be added to the `loss:` section of any config file:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `supervise_2d` | bool | false | MSE on `prediction[..., :2]` vs `target[..., :2]` instead of full 8-D |
| `smoothness_kind` | str | "first_diff" | "second_diff" = MSE of `(p[t+1] - 2*p[t] + p[t-1])` on cumsum 2D path. Else first-diff on cumsum when `supervise_2d` |
| `smoothness_empty_boost` | float | 1.0 | Multiply second-diff loss on empty scenes (dummy-aware) |
| `collision` | float | 0 | Weight for circular overlap collision loss (disabled if 0) |
| `collision_clearance` | float | 0.3 | Safety margin added to obstacle radius |
| `collision_kind` | str | "overlap" | "overlap" = `relu(r+clearance - dist)^2`. "clearance" = hard penetration + 0.5 * soft band |
| `path` | float | 0 | Weight for cumulative path MSE (disabled if 0) |
| `path_empty_only` | bool | false | Apply path MSE only to empty scenes |
| `path_occupied` | float | 0 | Weight for occupied scenes. If >0, loss = `(empty_term + path_occupied * occupied_term) / total_scenes` |

### Example Configurations

See `configs/digital_human_baseline_2d_*.yaml` for complete examples. Key combinations:

- **Basic 2D:** `supervise_2d: true`
- **Smooth 2D:** Add `smoothness_kind: second_diff`
- **Collision avoidance:** Add `collision: 2.0, collision_clearance: 0.3`
- **Empty-scene path:** Add `path: 2.0, path_empty_only: true, path_occupied: 0`

## How to Continue Training

### Prerequisites
1. Generate dataset (if not already present):
   ```bash
   python -m gtp.data.synthetic_navigation \
     --output data/synthetic_navigation.jsonl \
     --episodes 1000 \
     --seed 7
   ```

### Training
```bash
python -m gtp.train --config configs/digital_human_baseline_2d_collide.yaml
```

Change the config file to test different loss combinations. Checkpoints are saved to `runs/<experiment_name>/best.pt`.

### Evaluation
```bash
python -m gtp.evaluate \
  --checkpoint runs/digital_human_baseline_2d_collide/best.pt \
  --dataset data/synthetic_navigation.jsonl
```

The evaluator reports:
- **ADE/FDE:** Average/final displacement error
- **Collision rate:** Fraction of episodes with minimum clearance < threshold
- **Action accuracy:** Predicted vs. ground-truth action classification
- **Latency:** P50/P95 inference time

### Visualization
The evaluator saves trajectory figures to the run directory. **Always use a NEW filename** when saving figures. Never overwrite the two faculty-approved PNGs.

## What NOT to Do

1. **Do not start new training runs** without explicit approval from Jim / Qizhou Deng.
2. **Do not retune `path_occupied` weights.** Two additional weight searches have already failed.
3. **Do not add text-to-motion (T2M) integration.** Full-body motion is deferred.
4. **Do not claim the planner is "solved."** The official collision mask at 0.3 clearance remains True even on visual detours.
5. **Do not modify `dataset.py` dummy-obstacle behavior** without updating the loss computation accordingly.
6. **Do not overwrite the faculty figures.** Use new filenames for any new results.

## Open Issues and Hypotheses

These are **hypotheses**, not prescriptions. Do not act on them without approval.

### 1. Soft Collision Penalties
The collision loss is a soft penalty, so the official 0.3-clearance collision mask can remain True even when the trajectory visually deviates around obstacles. Episodes 17 and 128 show wide arcs but still trigger the official collision flag.

### 2. Path MSE on Occupied Scenes
Adding cumulative path MSE to occupied scenes (with ground-truth paths that already navigate around obstacles) appears to fight the collision avoidance objective. The combo experiments show increased ADE and collision rates when `path_occupied > 0`.

### 3. Limited Model Capacity
All experiments used a small transformer (128 dim, 3 layers) trained on 1000 synthetic episodes. Scaling to more data, diverse obstacle layouts, or a larger model may improve performance.

### 4. Runtime Hard Constraints
The architecture documentation already specifies that safety constraints must remain outside the neural model. A hard reject-and-replan mechanism at runtime (rather than soft loss penalties during training) may be necessary for deployment.

### 5. Empty-Scene Jitter
Even with empty-scene path supervision, episodes 0 and 42 retain minor wobbles. This may be due to the model still learning the multi-objective balance or insufficient training epochs.

### 6. CPU-Only Smoke Runs
All experiments were CPU-only for rapid iteration. GPU training may be required for larger models or datasets.

## File Attachments

The following figures were attached to the original task and are preserved under `docs/training-figures/`:

1. `trajectory-baseline-2d-collide-pred-vs-gt.png` (Faculty collision avoidance)
2. `trajectory-baseline-2d-path-pred-vs-gt.png` (Faculty empty-scene path)
3. `trajectory-baseline-2d-collide-empty-pred-vs-gt.png` (Contrast: collide-empty experiment)
4. `trajectory-baseline-2d-combo-pred-vs-gt.png` (Contrast: combo experiment, path-on-obstacles failed)

**Note:** If the PNG files could not be persisted as binary artifacts, they were originally attached to the task that created this handoff document.

## Questions?

Contact Jim / Qizhou Deng before starting new experiments or modifying the training freeze.

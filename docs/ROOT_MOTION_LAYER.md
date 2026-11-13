# Root-motion layer

Branch `feat/root-motion-layer`. Paired with the branch of the same name in
[LLM-NPC](https://github.com/QizhouD/LLM-NPC).

## What changed and why

The navigation MVP tried to learn pathfinding. That is a solved problem in games
(NavMesh), and the MVP could not beat a straight line on its own benchmark: its
expert labels collided with obstacles in 48.9% of episodes, the flow model's
samples were rejected 100% of the time, and nothing was trained on real motion.

The new split:

| Question | Answered by |
|---|---|
| Where to go | The engine's NavMesh. It sends path corners. |
| How to move along that route | This package: speed profile, corner rounding, facing, coming to rest. |
| Is the plan walkable | The engine again, by ray-casting each plan segment on the NavMesh. |

The research question becomes: *does a learned root-motion layer between NavMesh
and animation move an NPC more like a person than classical path following, within
a real-time budget?*

## Data

[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) (Ubisoft La
Forge), locomotion clips only: 18 clips, 73.5 minutes, five actors, gaits `walk`,
`run`, `sprint`.

**Licence: CC BY-NC-ND 4.0.** Research use only. Never commit the archive, the BVH
files or the extracted tracks; `.gitignore` covers `data/lafan1/` and `data/*.npz`.

```powershell
python -m gtp.rootmotion.fetch_lafan1 --output data/lafan1
python -m gtp.rootmotion.extract --bvh-dir data/lafan1 --output data/lafan1_root.npz
```

`extract` reads only the root joint, projects the hips to the ground, smooths with a
0.5 s cubic Savitzky-Golay filter, and resamples to 10 Hz in Unity's convention
(x right, z forward, yaw = atan2(x, z)).

### How a training window is built

Each window has 2 s of history, a 3 s future, a gait label and a **path**. The path
is what a NavMesh would give: a coarse polyline with no timing. It is recovered from
the capture by simplifying the trace the actor went on to walk (Ramer-Douglas-Peucker,
tolerance 0.1 to 0.4 m in training, 0.25 m in evaluation). The path ends with a goal
flag where the actor stops for at least one second; otherwise it is cut at 25 m or
16 vertices and flagged as continuing.

So the model must supply exactly what a polyline lacks. By construction the capture
stays within the simplification tolerance of its own path, which means **the
reference passes the runtime constraints 100% of the time**. The old benchmark's
labels failed its own filter half the time.

Split by actor: subjects 1 to 3 train (24,180 windows), subject 4 validation (1,737),
subject 5 test (1,441). Training mirrors windows left to right at random.

## Model

`RootMotionModel` generates 30 steps of `(dx, dz, dyaw)`. Three deliberate changes
from the MVP's flow model:

1. Targets are normalised (step std is about 0.095 m against unit noise before).
2. The output is 30 time tokens through a Transformer decoder, not one flattened
   vector through an MLP.
3. History (20 tokens) and path (up to 16 tokens) are read by cross-attention
   instead of being mean-pooled.

Velocity, speed and angular velocity are derived from the three generated channels,
so an exported plan cannot be kinematically inconsistent. The same network with
`deterministic: true` is the regression baseline.

## Planner and constraints

`RootMotionPlanner` draws K samples, rejects the invalid, executes the first valid
one (rejection sampling, which keeps the model's distribution), and otherwise falls
back to the spring follower, then to standing still.

- **Corridor (swept, with a bound).** Each 0.1 s segment is sub-sampled four times.
  Distance-to-path is 1-Lipschitz, so if every sample is within
  `half_width - spacing / 2`, the whole continuous trajectory is within
  `half_width`. The check includes the current position. This is a guarantee about
  deviation from the supplied path; walkability is verified in the engine.
- **Kinematics.** Speed 8 m/s, acceleration 20 m/s², turn rate 10 rad/s, including
  the jump from the current velocity to the first step.

## Results

Held-out subject 5: 1,441 windows (976 walk, 465 run, no sprint). The working model
is the CPU-sized one (587k parameters, 24 epochs for flow, best epoch 12 for the
regressor), trained on a GPU in about a minute per run.

### Which model

The full-size configuration (`rootmotion_flow.yaml`, `rootmotion_regressor.yaml`,
300 epochs on a GPU) overfits: validation loss is lowest at **epoch 60** for both
(flow 0.533, regressor 0.537) and then rises steadily; flow's reaches 1.088 by epoch
300. Evaluated at epoch 60 it brings no clear gain over the small model: flow ADE
0.411 against 0.442, but W1 acceleration 0.402 against 0.322 and a P50 latency of
121 ms against 23 ms on the CPU, and the full regressor is worse on ADE (0.462) and
fallback (17.8%). So the small model is the working model; the large one would need
regularisation or early stopping before it is worth its cost.

### Open loop, five seeds

Seeds 7, 11, 13, 17, 19; mean ± sample std. Reproduce one seed with:

```powershell
python -m gtp.rootmotion.train --config configs/rootmotion_flow_cpu.yaml
python -m gtp.rootmotion.train --config configs/rootmotion_regressor_cpu.yaml
python -m gtp.rootmotion.evaluate --config configs/rootmotion_flow_cpu.yaml `
  --flow runs/rootmotion_flow_cpu/best.pt `
  --regressor runs/rootmotion_regressor_cpu/best.pt --output results/rootmotion_cpu_test.json
```

| Method | ADE m | FDE m | minADE@16 m | W1 speed | W1 accel | W1 turn rate | Stop error m | Fallback | P50 ms |
|---|---|---|---|---|---|---|---|---|---|
| Constant speed on the polyline | 0.563 | 0.903 | – | 0.400 | 1.006 | 0.550 | 0.126 | – | – |
| Spring follower | 0.577 | 1.016 | – | 0.304 | 0.774 | **0.198** | **0.111** | – | – |
| Regressor + filter | **0.400 ± 0.006** | **0.737 ± 0.015** | 0.390 ± 0.006 | 0.221 ± 0.015 | 0.628 ± 0.017 | 0.565 ± 0.013 | 0.316 ± 0.020 | 9.5% ± 0.7 | 2 |
| Flow, K = 16 + filter (goal-aware) | 0.439 ± 0.006 | 0.856 ± 0.008 | **0.227 ± 0.004** | **0.197 ± 0.006** | **0.322 ± 0.006** | 0.288 ± 0.009 | 0.226 ± 0.007 | **0.4% ± 0.1** | 23 |

ADE compares positions at matching timestamps over 3 s, so it scores the speed
profile as well as the shape. W1 is the Wasserstein-1 distance between pooled
per-step distributions and the capture (m/s, m/s², rad/s). Stop error is the
distance from the final position to the goal on windows where the capture comes to
rest; the capture itself scores 0.104. Latency is one request on the CPU.

The flow row uses goal-aware selection, the default since the ablation below; with
the previous first-valid rule its stop error is 0.366 ± 0.009 and ADE 0.442. The
regressor draws one candidate, so the rule does not affect it.

What this shows:

- The pipeline works end to end, and every executed plan satisfies both constraints.
- **The regressor has the better ADE and FDE.** The flow model's advantage is
  distributional realism (speed and acceleration W1, minADE of its 16 samples) and a
  far lower fallback rate (0.4% against 9.5%), not average accuracy.
- Both lose to the spring follower on turn-rate realism and on stopping at the goal.
- The seed spread is small next to the differences between methods, but this is
  still one actor's three test clips and no sprint.

### Closed loop

`python -m gtp.rootmotion.closed_loop` replans along routes the way the engine does,
through the service's own code path (`build_batch → plan_batch → build_plan`):
every 1 s (the scene's `TrajectoryPlanFollower.replanInterval`) it sends the last
20 poses and "current position + remaining route", executes 1 s of the answer and
asks again. Routes: straight 3, 6 and 12 m, a 4+4 m right angle, a Z, a 4+4 m
fold-back, each facing the first leg or turned 90 degrees, walk and run, plus the
Unity demo route. W1 is taken over the active window only (start to arrival + 1 s).

Mean over the 12 walk routes, then over the five seeds:

| Method | Mean speed while following | Final distance to goal m | Final speed m/s | W1 speed | W1 accel | W1 turn |
|---|---|---|---|---|---|---|
| Regressor | 0.561 ± 0.039 | 0.140 ± 0.025 | 0.031 | 0.525 | 0.808 | 1.057 |
| Flow (goal-aware) | 0.596 ± 0.046 | 0.108 ± 0.014 | 0.122 | 0.501 | **0.578** | **0.811** |
| Spring | 0.821 | **0.015** | 0.009 | **0.345** | 0.783 | 0.992 |
| Constant speed | 0.880 | 0.000 | 0.000 | 0.389 | 0.895 | 0.995 |

The diagnosis below was run with first-valid selection (flow 0.555 m/s, final
distance 0.157 m on these routes); goal-aware selection does not change it.

On the Unity demo route the flow model averages 0.49 m/s against 0.85 for the spring,
the same gap the engine showed (0.57 against 0.86), so the slowness is in the
planner, not the integration. Diagnosis:

- **Closed-loop feedback is the main cause for flow.** From rest a plan accelerates
  late (0.34 m/s in its first second, 1.06 in its third on a 12 m straight), and the
  next plan follows the history speed (from a 0.4 m/s history its first second is
  0.46). Executing 1 s and replanning keeps playing only the slow head of each plan:
  on the 12 m straight the first 3 s average 0.80 m/s open loop and 0.42 closed loop,
  and replanning every 3 s (each plan played in full) gives 1.03 m/s on the Unity
  route against 0.49 at 1 s. The dependence on the interval is not monotonic
  (0.2 s is fast on the straight but not on the Unity route).
- **Data distribution is secondary.** Single straight paths of 5 m or more are under
  1% of walk windows, and capture walking is slow on short paths (0.22 to 0.64 m/s
  below 10 m), which explains the short routes. It does not explain the 12 m routes:
  the capture walks 0.92 m/s on 5 to 10 m straight paths, the model reaches 0.80
  open loop, and hiding the goal behind a 20 m continuation makes it slower, not
  faster.
- **The regressor is slow already open loop** (0.38 to 0.60 m/s within one plan from
  rest): one deterministic output averages over when to start.
- In closed loop the flow model does not settle: it keeps shuffling on the goal at
  about 0.14 m/s. Its plan seams carry about 1.7 times the acceleration of other
  steps (the regressor's 3.5 times); the spring has none.

So the open-loop advantage on the speed distribution does not survive closed loop;
the advantages on acceleration and turn-rate distributions do.

### Stop error: two fixes as an ablation

- **A, goal-aware selection** (`--selection goal_aware`, no retraining): when the
  path ends at the goal within gait speed × 3 s, take the valid candidate minimising
  endpoint distance + 0.5 s × final speed.
- **B, position-space loss** (`training.position_loss_weight`, weights 0.5 and 2,
  five seeds each).

| Flow, mean ± std over 5 seeds | Open stop error | ADE | minADE | W1 speed | W1 accel | W1 turn | Fallback | Closed final distance (walk / run) |
|---|---|---|---|---|---|---|---|---|
| First-valid (previous default) | 0.366 ± 0.009 | 0.442 | 0.227 | 0.197 | 0.322 | 0.286 | 0.4% | 0.152 / 0.233 |
| A | **0.226 ± 0.007** | 0.439 | 0.227 | 0.197 | 0.322 | 0.288 | 0.4% | **0.108** / **0.138** |
| B, weight 0.5 | 0.365 ± 0.012 | 0.440 | 0.227 | 0.198 | 0.318 | 0.286 | 0.4% | 0.143 / 0.217 |
| B, weight 2 | 0.361 ± 0.007 | 0.440 | 0.226 | 0.198 | 0.321 | 0.287 | 0.2% | 0.150 / 0.215 |
| A + B, weight 0.5 | 0.223 ± 0.011 | 0.437 | 0.227 | 0.198 | 0.319 | 0.288 | 0.4% | 0.106 / 0.138 |
| A + B, weight 2 | 0.222 ± 0.009 | 0.437 | 0.226 | 0.197 | 0.322 | 0.289 | 0.2% | 0.112 / 0.134 |

A cuts the flow model's open-loop stop error by 38% and its closed-loop final
distance by a third, at no cost in any W1. The regressor draws one candidate, so A
does not apply to it. B makes no measurable difference at these weights for either
model (regressor open stop error 0.316 → 0.325 / 0.319): the position term is about
1% of the flow loss and 3 to 5% of the regressor's, so much larger weights would be
needed. Even with A, the flow model's stop error (0.226 m) remains twice the
spring's (0.111) and the capture's (0.104). A is now the default selection rule of
the planner and the service; the position loss stays off.

![Held-out windows](rootmotion-figures/test-windows-cpu.png)

The figure also shows a data limitation: LAFAN1 was captured in a roughly 7 by 12 m
volume, so long paths fold back over themselves (windows 520, 700 and 900). A
self-overlapping path weakens the corridor check and makes the task ambiguous.

## Service

```powershell
python -m gtp.rootmotion.service --checkpoint runs/rootmotion_flow_cpu/best.pt --port 8787
```

`POST /plan`, world space, Unity convention:

```json
{
  "history": [{"position": [10.0, 0.0, 20.0], "yaw": 0.0}],
  "path": [[10.0, 0.0, 20.0], [10.0, 0.0, 23.0], [13.0, 0.0, 23.0]],
  "path_ends_at_goal": true,
  "gait": "walk",
  "corridor_half_width": 0.5,
  "candidates": 16
}
```

`history` is up to 20 poses at 10 Hz, oldest first; fewer are padded as standing.
The response is the existing `TrajectoryPlan` (31 points, the first being the
current pose at `t = 0`) plus `source` (`model`, `spring_fallback` or `hold`),
`rejected_candidates`, `candidates` and `latency_ms`. `fallback` is true only for
`hold`.

## Next steps

1. Closed-loop speed: the planner should not fall into a slow equilibrium when it
   replans on its own history. Candidates: training on rolled-out (closed-loop)
   windows, conditioning on a target speed, or replanning at the end of each plan.
2. Arrival: goal-aware selection halves the gap; the rest needs the model itself to
   stop, for example classifier-free guidance on the goal flag or a much larger
   position-loss weight, and the closed-loop shuffling on the goal must stop.
3. Facing: generate facing relative to the direction of travel, or smooth it.
4. The full-size model needs regularisation or early stopping before it pays off.
5. More data with open floor space (AMASS locomotion, 100STYLE) so paths stop
   folding; add HumanML3D text for real language conditioning. These need your own
   licence agreement and download.
6. Closed-loop evaluation in Unity with `RootMotionMetricsLogger`, against the real
   NavMeshAgent, including a moving target.
7. Body layer: drive a full-body model from the planned root, then measure foot
   sliding and run a small perceptual study.

## Legacy

The navigation MVP (`gtp.train`, `gtp.evaluate`, `gtp.planning`,
`gtp.data.synthetic_navigation`) is unchanged and its tests still pass. It is kept
for the record and is not part of the new direction.

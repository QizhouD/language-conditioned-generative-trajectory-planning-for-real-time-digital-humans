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
- **Without a speed input the regressor has the better ADE and FDE.** The flow
  model's advantage is distributional realism (speed and acceleration W1, minADE of
  its 16 samples) and a far lower fallback rate (0.4% against 9.5%), not average
  accuracy. Given a target speed (below), flow's ADE drops to 0.413 ± 0.003, level
  with the regressor.
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
  about 0.14 m/s. That is all post-arrival motion that the engine never shows:
  with `--lock-on-arrival` (stop at 0.25 m, as `TrajectoryPlanFollower` does) its
  final speed is 0, and every method's final distance then sits at the tolerance
  (flow 0.227, regressor 0.234, spring 0.190 m on walk), so stop error no longer
  separates them. Its plan seams carry about 1.7 times the acceleration of other
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

### Target-speed conditioning

A `walk` label covers 0.2 to 1.3 m/s and more, so without being told a speed the
model can only infer it from the history, and in closed loop a slow history keeps
it slow; the spring follower is told its speed. With `model.speed_conditioning`
(`configs/rootmotion_*_speed_cpu.yaml`) the gait token also carries a target speed.

- **Label.** Per window, the 90th percentile of the moving step speeds (> 0.2 m/s)
  from now to where the path ends (up to 25 m ahead). Training labels, p5 / median /
  p95: walk 0.40 / 0.93 / 2.08, run 1.36 / 3.12 / 4.21 m/s; correlation with the
  mean speed of the next 3 s is 0.77 for walk and 0.57 for run. It is a route-level
  high speed: humans average about two thirds of it.
- **Model.** The speed (÷ POSITION_SCALE) goes through a small MLP and is added to
  the gait token; a learned "unspecified" vector replaces it when no speed is given,
  and training drops the speed 10% of the time. Off by default; old checkpoints load
  and reproduce exactly.
- **Runtime.** `/plan` takes an optional `target_speed`; it is clamped to the
  training labels' 1st to 99th percentile for the gait, and the spring fallback and
  goal-aware selection use it.

Open loop, five seeds (the followers told each window's own label for a fair
comparison):

| Method | ADE | FDE | minADE | W1 speed | W1 accel | W1 turn | Stop error | Corridor viol. | Fallback |
|---|---|---|---|---|---|---|---|---|---|
| Spring, gait median | 0.577 | 1.016 | – | 0.304 | 0.774 | 0.198 | 0.111 | 0.9% | – |
| Spring, target speed | 0.521 | 0.996 | – | 0.259 | 0.515 | **0.183** | **0.059** | 6.2% | – |
| Constant speed, target speed | 0.798 | 1.380 | – | 0.503 | 1.026 | 0.654 | 0.091 | 0.0% | – |
| Regressor, no speed input | 0.400 ± 0.006 | 0.737 | 0.390 | 0.221 | 0.628 | 0.565 | 0.316 | 0.0% | 9.5% |
| Regressor, speed given | 0.415 ± 0.019 | 0.764 | 0.369 | 0.210 | 0.624 | 0.578 | 0.243 | 0.0% | 7.7% |
| Flow, no speed input | 0.439 ± 0.006 | 0.856 | 0.227 | 0.197 | 0.322 | 0.288 | 0.226 | 0.0% | 0.4% |
| **Flow, speed given** | **0.413 ± 0.003** | 0.804 | **0.214** | **0.183** | **0.316** | 0.284 | 0.168 | 0.0% | 0.3% |
| Flow, speed hidden | 0.437 ± 0.006 | 0.854 | 0.228 | 0.195 | 0.315 | 0.308 | 0.233 | 0.0% | 0.3% |

With the speed given, flow improves on every metric except turn rate, and its stop
error falls from 0.226 to 0.168. With the speed hidden it matches the unconditioned
model, so the "unspecified" input works. The spring told the speed is a much
stronger baseline: it wins on stop error and turn rate, but 6.2% of its plans leave
the 0.5 m corridor, where every executed flow plan stays inside.

Closed loop at a requested speed (replan 1 s, 12 routes per gait, five seeds;
cruise = 90th percentile of the moving speeds before arrival, the label's own
definition):

| Gait @ asked | Method | Mean abs(cruise − asked) | Mean speed while following | First second | Final distance | W1 speed | W1 accel | W1 turn |
|---|---|---|---|---|---|---|---|---|
| walk @ 0.9 | flow | 0.063 ± 0.014 | 0.575 | 0.255 | 0.119 | 0.505 | **0.579** | **0.798** |
| walk @ 0.9 | regressor | 0.176 ± 0.040 | 0.551 | 0.285 | 0.121 | 0.520 | 0.809 | 1.054 |
| walk @ 0.9 | spring | 0.000 | 0.838 | 0.557 | 0.015 | **0.338** | 0.781 | 0.994 |
| walk @ 1.3 | flow | 0.124 ± 0.019 | 0.827 | 0.377 | 0.137 | 0.297 | **0.423** | **0.692** |
| walk @ 1.3 | spring | 0.001 | 1.176 | 0.804 | 0.025 | **0.208** | 0.571 | 0.935 |
| run @ 2.0 | flow | 0.405 ± 0.065 | 1.293 | 0.570 | 0.131 | 1.039 | **0.875** | **0.422** |
| run @ 2.0 | spring | 0.014 | 1.703 | 1.237 | 0.033 | **0.641** | 0.937 | 0.681 |

- **Walk: the cruise speed follows the ask** (flow within 0.04 to 0.12 m/s of 0.6,
  0.9 and 1.3; the regressor undershoots by twice that). **Run does not**: flow
  overshoots on the 12 m straight (2.73 for 2.0) and undershoots on short routes,
  0.41 to 0.78 m/s off on average.
- **The mean speed and arrival time did not catch up.** On the Unity demo route at
  0.9 m/s the flow model averages 0.510 m/s and arrives after 22.3 s (unconditioned:
  0.488 m/s, 22.9 s; spring 0.87 m/s, 12.3 s). Its ratio of mean speed to cruise is
  0.65, close to the capture's own 0.67 for the same quantity (walk), against 0.94
  for the spring. The label is a high speed and humans average two thirds of it, so
  asking the model for "0.9" asks for a human whose peak is 0.9.
- **Starts are slow, as in the capture.** Flow's first second averages 0.26 m/s at
  0.9; capture starts from rest with a similar label average 0.14 m/s, the spring
  0.56. Flow also arrives at a human-like speed (0.52 m/s when it enters the 0.25 m
  tolerance, capture median 0.58), where the spring arrives at 0.83 and the engine's
  lock stops it dead.
- **Replanning interval.** With the speed given, the walk cruise speed no longer
  depends much on it (0.79 to 0.88 on the 12 m straight across 0.2 to 3 s); on the
  Unity route 3 s is still faster (0.63 against 0.51 m/s mean speed) and run still
  depends on it (1.40 at 0.2 s, 2.04 at 1 s).

Against the spring told the same speed, flow keeps its advantage on acceleration
and turn-rate realism and on staying in the corridor, and loses on speed W1, path
deviation, stop error, seams (0.96 against 0.03 m/s² at walk 0.9) and run speed
tracking.

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
  "candidates": 16,
  "target_speed": 0.9
}
```

`history` is up to 20 poses at 10 Hz, oldest first; fewer are padded as standing.
`target_speed` (m/s) is optional. The response is the existing `TrajectoryPlan`
(31 points, the first being the current pose at `t = 0`) plus `source` (`model`,
`spring_fallback` or `hold`), `rejected_candidates`, `candidates`, `latency_ms` and
`target_speed`: `{"requested", "used", "clipped"}` with a `note` when the speed was
clamped into the gait's training range, left unspecified, or ignored because the
checkpoint is not speed-conditioned. `fallback` is true only for `hold`. Candidate
selection defaults to goal-aware.

Closed-loop evaluation:

```powershell
python -m gtp.rootmotion.closed_loop --config configs/rootmotion_flow_speed_cpu.yaml `
  --flow runs/rootmotion_flow_speed_cpu/best.pt `
  --target-speed walk=0.9,run=2.0 --lock-on-arrival --output results/closed.json
```

`--target-speed` asks every request of a gait for that speed (and runs only those
gaits); `--lock-on-arrival` stops at 0.25 m like the Unity follower.

## Next steps

1. Closed-loop speed: speed conditioning makes the walking cruise speed follow the
   ask, but mean speed and arrival time stay human-like, well below the spring.
   Whether that is wanted is a design question; if not, a mean-speed label or asking
   above the desired mean would be the next things to try. Run speed tracking does
   not work yet.
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

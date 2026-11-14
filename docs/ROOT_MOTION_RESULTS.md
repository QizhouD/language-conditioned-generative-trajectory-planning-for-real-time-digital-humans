# Root-motion results (walk)

This document is the only source of numbers for the paper. If another document
(`ROOT_MOTION_LAYER.md`, the dated reports, `results/**/tables.md`) disagrees with it,
this one wins. Each table names its protocol, sample size, seeds and the script that
produces it; the scripts are in [`scripts/rootmotion/`](../scripts/rootmotion/README.md).
Data, checkpoints and result files are not in the repository.

Status 2026-11-14: tables T1 to T9 are final. T10 (latency) is waiting to be measured
again on an idle machine.

## Method under test

- **Flow:** speed-conditioned flow model (`configs/rootmotion_flow_speed_cpu.yaml`,
  597k parameters). Every request carries the target speed.
- **Selection:** consistent, K = 16. Among the valid candidates it takes the one
  closest to the unexecuted part of the previous plan; with no previous plan, or near
  the goal, it falls back to goal-aware.
- **Runtime:** replans every 1 s and executes 1 s of each plan. Plans must stay in a
  0.5 m corridor. Spring fallback, then hold.
- **Scope:** walk only. Run is not used (see Limitations).

Baselines:

- **Spring (untuned):** the spring follower told the walk's p90 label as its cruise
  speed, with hand-set constants.
- **Spring (tuned):** the same follower with `speed_scale`, `halflife`, `deceleration`,
  `arrival_seconds` and `lookahead_seconds` tuned per fold on training actors only (T1).
- **Regressor:** the same network, deterministic. It appears only in T8 and T10.

## Protocols

| Code | Protocol | Samples | Seeds / checkpoints |
|---|---|---|---|
| **LOAO** | Leave one actor out. Train on four LAFAN1 actors, test closed loop on every rest-to-rest walk of the fifth. Each walk becomes a route (RDP, 0.25 m), starts from the captured history, asks for its own p90 label and has 3 × the capture's duration. The capture is the reference. Its arrival is gated: within 0.25 m of the end and within 1 m of the route's length. | 73 walks: 61 short (≤ 35 m), 12 long (> 35 m). Per actor 1–5: short 25 / 10 / 16 / 2 / 8, long 3 / 3 / 1 / 3 / 2 | 5 folds × 3 seeds (7, 11, 13), last-epoch checkpoints (the held-out actor never picks one); `runs/loao/base_hold{1..5}_s{7,11,13}` |
| **STD** | Standard split: actors 1–3 train, 4 validation, 5 test; best-validation checkpoint | Open loop: 1,441 test windows (976 walk, 465 run). Closed loop: actor 5's 10 rest-to-rest walks (8 short, 2 long) | 5 seeds (7, 11, 13, 17, 19); `runs/speedcond/rootmotion_*_speed_cpu_s*` |
| **SYN** | Synthetic routes from rest at 0.9 m/s: 12 m straight and the Unity demo route; 6 m straight in T8 | 1 route each | Flow: STD checkpoints × 4 sampling seeds = 20 runs. Tuned spring: each fold's setting. Deterministic baselines: 1 run |
| **HUM** | Human reference for SYN: rest-to-rest walks of 8 to 15 m with label 0.8 to 1.1, all actors | n = 6 | – |

Metrics:

- **Arrival / capture:** arrival time over the capture's arrival time.
- **Mean speed / capture:** mean speed while following, over the capture's.
- **Profile RMSE (m/s):** speed by route progress against the capture.
- **W1 speed / accel / turn:** Wasserstein-1 distance between pooled per-step
  distributions and the capture (m/s, m/s², rad/s), taken over the active window (start
  to arrival + 1 s).
- **Seam / other accel (m/s²):** mean |acceleration| on the first step of a new plan,
  and on all other steps.
- **Final dist:** distance to the goal at the end (m).
- **Max dev:** maximum distance from the route (m).
- In the paired tests, the two ratios are scored by their distance from 1, |x − 1|,
  computed per run.

## Conclusions: flow against the tuned spring (from T5)

Paired per walk, flow − tuned spring. A difference counts as significant when the
walk-level 95% bootstrap interval excludes 0. This rule was fixed before the results
were seen. "Actor level" is the cluster bootstrap over the five actors.

| | Short walks (61) | Long walks (12) |
|---|---|---|
| **Flow better** | W1 acceleration (actor level agrees; 5/5 actors). W1 turn rate (actor level agrees; 4/5 actors) | W1 acceleration (actor level agrees; 5/5 actors) |
| **Tie** | \|arrival / capture − 1\|; W1 speed | \|arrival / capture − 1\|; \|mean speed / capture − 1\|; W1 speed; W1 turn rate; max deviation |
| **Flow worse** | Profile RMSE; final distance; max deviation; seam acceleration (all four: actor level agrees, 0/5 actors favour flow). \|mean speed / capture − 1\| (walk level only, CI [+0.009, +0.094]; not at actor level; marginal) | Profile RMSE; final distance; seam acceleration (actor level agrees) |

In words:

- **What survives tuning.** Flow's lasting advantage is the realism of the
  acceleration distribution, on short and long walks and for every actor. On short
  walks the turn-rate distribution is also more human-like.
- **What tuning removes.** The tuned spring closes the timing gap: arrival and speed W1
  tie.
- **What the tuned spring does better.** It is better on the speed profile, on stopping
  at the goal, on seams and (short walks) on path deviation.
- **Direction of the timing error.** Against the person, flow is late on short walks
  (arrival / capture 1.11, mean speed / capture 0.86, fold means in T2) and on par on
  long walks (1.00 / 1.00, T3). The tuned spring is early: 0.93 / 1.04 on short walks
  and 0.87 / 1.17 on long walks.
- **Gap between the methods.** Per walk, flow arrives later than the tuned spring
  (+0.20 short, +0.13 long, in arrival / capture) and walks slower (−0.19, −0.17 in
  mean speed / capture). These are differences between the two methods, not flow's
  error against the capture.

Caveats that belong with every use of this summary:

- The spring was tuned on arrival and the speed profile, the two terms of its
  objective. Flow was not tuned on any closed-loop metric.
- About 18 intervals are reported, so the marginal short-walk mean-speed result may be
  chance. The short-walk arrival tie is also narrow (walk CI [−0.014, +0.085], actor
  CI [−0.002, +0.094]).
- Long walks are 12, so their ties are weak evidence of equality.
- The actor-level bootstrap has five clusters and is coarse.

## T1. Tuned spring: per-fold selection

Protocol: LOAO. Search on the four training actors' walks; selection uses no data from
the held-out actor.

- **Objective.** Mean over walks of (profile RMSE / capture mean speed) +
  |arrival / capture − 1|, with equal weight. A run that never arrives counts as
  3 × the capture's duration; none occurred.
- **Search.** The defaults plus 400 random draws:
  - scale 0.5–1.1;
  - halflife 0.1–2 s, log-uniform;
  - deceleration 0.2–4 m/s², log-uniform;
  - arrival 0.3–2 s;
  - look-ahead 0.05–1.5 s.

  Then 80 log-normal perturbations (σ 0.15) around each fold's best. A fold chooses
  only among the 401 shared draws and its own 80.
- Scripts: `g_search.py` → `results/phase_g/search.json`; `g_eval.py` →
  `selection.json`.

| Fold (held-out actor) | Scale | Halflife | Deceleration | Arrival | Look-ahead | Objective, train: default → tuned | Held-out: default → tuned |
|---|---|---|---|---|---|---|---|
| 1 | 0.81 | 0.27 s | 0.77 | 1.54 s | 0.08 s | 0.64 → 0.46 | 0.64 → 0.44 |
| 2 | 0.75 | 0.29 s | 0.60 | 1.74 s | 0.28 s | 0.69 → 0.48 | 0.39 → 0.34 |
| 3 | 0.83 | 0.33 s | 0.77 | 1.29 s | 0.10 s | 0.58 → 0.40 | 0.82 → 0.60 |
| 4 | 0.77 | 0.29 s | 0.70 | 1.08 s | 0.13 s | 0.63 → 0.45 | 0.68 → 0.47 |
| 5 | 0.84 | 0.39 s | 2.48 | 1.46 s | 0.34 s | 0.64 → 0.44 | 0.62 → 0.51 |

Notes:

- **Bounds.** No chosen value lies within 2% of a searched bound.
- **Look-ahead floor.** The look-ahead distance has a fixed floor of 0.4 m
  (`min_lookahead`, not searched), and every chosen look-ahead time falls under it at
  walking speed. Rerunning each fold's choice with a 0.2 or 0.1 m floor changes the
  training objective by at most 0.011, in no consistent direction (`g_checks.py` →
  `checks.json`).
- **Scale alone.** Most of the gain comes from the cruise scale. The default constants
  with scale 0.8 give short-walk arrival / capture 0.87, W1 speed 0.085 and W1 accel
  0.459, pooled over the 61 short walks (`checks.json`).

## T2. Closed loop on held-out actors, short walks

Protocol: LOAO, n = 61. Each value is the mean ± std over the five folds of the
per-fold value. Flow's per-fold value averages its 3 seeds; the springs are
deterministic. Scripts: `f_loao_eval.py base|spring` and `g_eval.py` produce the
inputs, and `g_stats.py` writes the table (`results/phase_g/tables.md`).

| Method | Arrival / capture | Mean speed / capture | Profile RMSE | W1 speed | W1 accel | W1 turn | Seam / other accel | Final dist | Max dev |
|---|---|---|---|---|---|---|---|---|---|
| Flow | 1.11 ± 0.08 | 0.86 ± 0.09 | 0.30 ± 0.06 | 0.133 ± 0.067 | 0.228 ± 0.085 | 0.166 ± 0.075 | 0.98 / 0.70 | 0.12 | 0.27 |
| Spring, untuned | 0.74 ± 0.07 | 1.29 ± 0.12 | 0.23 ± 0.06 | 0.190 ± 0.081 | 0.368 ± 0.073 | 0.299 ± 0.126 | 0.11 / 0.77 | 0.02 | 0.18 |
| Spring, tuned | 0.93 ± 0.11 | 1.04 ± 0.11 | 0.22 ± 0.05 | 0.128 ± 0.040 | 0.471 ± 0.035 | 0.306 ± 0.127 | 0.06 / 0.48 | 0.03 | 0.16 |

- **Arrival.** Every method arrives on every walk, and no method falls back.
- **Corridor.** Runs beyond the 0.5 m corridor are rare: flow 3 of 219 (worst
  0.57 m), tuned spring 2 of 73 (0.53 m). This count was made ad hoc on the same result
  files; no committed script produces it.

## T3. Closed loop on held-out actors, long walks

Protocol and script as T2; n = 12.

| Method | Arrival / capture | Mean speed / capture | Profile RMSE | W1 speed | W1 accel | W1 turn | Seam / other accel | Final dist | Max dev |
|---|---|---|---|---|---|---|---|---|---|
| Flow | 1.00 ± 0.28 | 1.00 ± 0.29 | 0.46 ± 0.14 | 0.258 ± 0.162 | 0.216 ± 0.131 | 0.281 ± 0.232 | 1.38 / 0.95 | 0.12 | 0.36 |
| Spring, untuned | 0.66 ± 0.14 | 1.48 ± 0.30 | 0.44 ± 0.15 | 0.450 ± 0.226 | 0.459 ± 0.051 | 0.234 ± 0.066 | 0.18 / 1.30 | 0.02 | 0.29 |
| Spring, tuned | 0.87 ± 0.22 | 1.17 ± 0.26 | 0.34 ± 0.11 | 0.303 ± 0.192 | 0.511 ± 0.105 | 0.293 ± 0.105 | 0.10 / 0.88 | 0.03 | 0.35 |

## T4. Per held-out actor

Protocol: LOAO. Script: `g_stats.py`. Each cell gives arrival / capture, mean speed /
capture, profile RMSE, W1 speed / accel / turn, and max dev.

Short walks:

| Actor | n | Flow | Spring, untuned | Spring, tuned |
|---|---|---|---|---|
| 1 | 25 | 1.07, 0.90, 0.22, 0.068/0.207/0.141, 0.27 | 0.73, 1.30, 0.18, 0.171/0.373/0.269, 0.15 | 0.87, 1.07, 0.15, 0.081/0.445/0.319, 0.10 |
| 2 | 10 | 1.22, 0.75, 0.27, 0.207/0.200/0.223, 0.23 | 0.83, 1.14, 0.16, 0.065/0.273/0.504, 0.15 | 1.06, 0.91, 0.21, 0.118/0.423/0.522, 0.09 |
| 3 | 16 | 1.03, 0.98, 0.36, 0.059/0.229/0.064, 0.24 | 0.67, 1.46, 0.30, 0.240/0.420/0.230, 0.16 | 0.82, 1.19, 0.27, 0.134/0.491/0.239, 0.14 |
| 4 | 2 | 1.16, 0.88, 0.32, 0.145/0.138/0.148, 0.30 | 0.80, 1.27, 0.26, 0.279/0.453/0.175, 0.22 | 1.03, 0.99, 0.25, 0.191/0.506/0.204, 0.18 |
| 5 | 8 | 1.08, 0.81, 0.33, 0.184/0.369/0.255, 0.30 | 0.69, 1.30, 0.25, 0.194/0.320/0.315, 0.22 | 0.85, 1.03, 0.25, 0.118/0.489/0.249, 0.30 |

Long walks (one to three per actor; too few to compare methods within an actor):

| Actor | n | Flow | Spring, untuned | Spring, tuned |
|---|---|---|---|---|
| 1 | 3 | 0.81, 1.17, 0.34, 0.144/0.175/0.059, 0.40 | 0.55, 1.71, 0.54, 0.564/0.461/0.208, 0.27 | 0.68, 1.38, 0.39, 0.390/0.494/0.211, 0.26 |
| 2 | 3 | 1.15, 0.85, 0.42, 0.171/0.175/0.284, 0.36 | 0.79, 1.25, 0.36, 0.277/0.513/0.283, 0.36 | 1.07, 0.95, 0.28, 0.145/0.590/0.437, 0.31 |
| 3 | 1 | 0.67, 1.39, 0.69, 0.397/0.093/0.412, 0.34 | 0.47, 1.87, 0.64, 0.777/0.383/0.325, 0.28 | 0.60, 1.50, 0.49, 0.598/0.338/0.260, 0.48 |
| 4 | 3 | 1.00, 0.98, 0.35, 0.111/0.197/0.056, 0.37 | 0.70, 1.41, 0.40, 0.412/0.498/0.178, 0.22 | 0.91, 1.09, 0.33, 0.231/0.541/0.192, 0.19 |
| 5 | 2 | 1.38, 0.62, 0.51, 0.467/0.440/0.595, 0.33 | 0.77, 1.16, 0.24, 0.218/0.441/0.178, 0.35 | 1.08, 0.93, 0.21, 0.153/0.592/0.364, 0.52 |

## T5. Paired comparison, flow − tuned spring

Protocol: LOAO.

- **Pairing.** Each walk is one pair. Flow's value averages its 3 seeds.
- **Bootstrap.** Walk level: 10,000 resamples of walks. For the pooled W1 metrics,
  W1 is recomputed on each of 2,000 resamples. Actor level: a cluster bootstrap over
  the 5 actors, with the same resample counts.
- **Reading the table.** Negative values favour flow.

Script: `g_stats.py` (seeded) → `results/phase_g/paired.json`, `tables.md`.

| Metric | Short (61): flow − tuned [walk 95% CI] | Short: actor CI | Short: actors favouring flow | Long (12): flow − tuned [walk 95% CI] | Long: actor CI | Long: actors favouring flow |
|---|---|---|---|---|---|---|
| \|arrival / capture − 1\| | +0.035 [−0.014, +0.085] tie | [−0.002, +0.094] | 1/5 | +0.044 [−0.040, +0.136] tie | [−0.074, +0.184] | 2/5 |
| \|mean speed / capture − 1\| | +0.052 [+0.009, +0.094] spring better, not at actor level | [−0.008, +0.127] | 1/5 | +0.027 [−0.075, +0.129] tie | [−0.131, +0.188] | 2/5 |
| Profile RMSE (m/s) | +0.077 [+0.053, +0.104] spring better | [+0.065, +0.091] | 0/5 | +0.094 [+0.019, +0.170] spring better | [+0.003, +0.223] | 1/5 |
| W1 speed | −0.018 [−0.058, +0.022] tie | [−0.073, +0.085] | 3/5 | −0.069 [−0.195, +0.130] tie | [−0.201, +0.206] | 3/5 |
| W1 acceleration | **−0.241 [−0.265, −0.216] flow better** | [−0.279, −0.196] | 5/5 | **−0.301 [−0.380, −0.221] flow better** | [−0.398, −0.193] | 5/5 |
| W1 turn rate | **−0.177 [−0.216, −0.133] flow better** | [−0.241, −0.071] | 4/5 | −0.030 [−0.184, +0.111] tie | [−0.179, +0.152] | 3/5 |
| Seam acceleration (m/s²) | +0.895 [+0.840, +0.951] spring better | [+0.870, +0.957] | 0/5 | +1.215 [+1.008, +1.441] spring better | [+0.965, +1.464] | 0/5 |
| Final distance (m) | +0.079 [+0.068, +0.090] spring better | [+0.074, +0.094] | 0/5 | +0.079 [+0.060, +0.100] spring better | [+0.060, +0.102] | 0/5 |
| Max deviation (m) | +0.123 [+0.098, +0.145] spring better | [+0.059, +0.161] | 0/5 | +0.046 [−0.041, +0.127] tie | [−0.104, +0.135] | 2/5 |
| Arrival / capture, signed (descriptive) | +0.195 [+0.154, +0.237], flow later | [+0.175, +0.213] | – | +0.131 [+0.079, +0.192], flow later | [+0.081, +0.216] | – |
| Mean speed / capture, signed (descriptive) | −0.185 [−0.227, −0.145], flow slower | [−0.213, −0.159] | – | −0.166 [−0.220, −0.113], flow slower | [−0.241, −0.102] | – |
| Other-step accel (descriptive) | +0.215 [+0.153, +0.266], flow higher | [+0.129, +0.265] | – | +0.108 [−0.038, +0.240], no difference | [−0.097, +0.248] | – |

The per-actor differences are in `results/phase_g/tables.md`.

## T6. Starting from rest on synthetic routes, against the human reference

Protocol: SYN at 0.9 m/s, with HUM as the human reference.

- **Flow:** STD checkpoints, consistent K = 16, 20 runs (5 checkpoints × 4 sampling
  seeds). Time to 0.81 m/s and arrival are medians; mean speed is a mean.
- **Tuned spring:** each LOAO fold's setting from T1, one run each.
- **Untuned spring:** one run.
- **Humans:** medians over n = 6.

Scripts: flow from `f_resample.py base` (`resample_base_C2.json`); springs from
`g_eval.py` (`synthetic.json`); humans from `e_checks.py`, section 2.

| Method | Cruise (m/s) | Time to 0.45 m/s | Time to 0.81 m/s, 12 m / Unity | Mean speed / asked, 12 m / Unity | Arrival, 12 m / Unity |
|---|---|---|---|---|---|
| Spring, untuned | 0.90 | 0.3 s | 0.8 / 0.8 s | 0.97 / 0.97 | 13.5 / 12.3 s |
| Spring, tuned (folds 1–5) | 0.67 to 0.76 | 0.4 to 0.6 s | never | 0.73 to 0.81 / 0.73 to 0.80 | 16.2 to 18.0 / 14.8 to 16.4 s |
| Flow | – | 0.9 s | 9.0 / 11.5 s | 0.70 / 0.62 | 19.6 / 19.9 s |
| Humans, 8 to 15 m (n = 6) | – | 0.5 s | 2.8 s (p25 1.35, p75 4.8) | 0.72 of their label (mean 0.67 m/s) | – |

- **Flow's time to 0.45 m/s.** This is the 5-run mean from `f_eval.py base`
  (`eval_base.json`).
- **Maximum deviation.** Tuned spring 0.03 to 0.17 m; flow 0.26 to 0.44 m (5-run
  means).
- **Flow.** Its mean speed is close to the human ratio on the 12 m straight and slow on
  the Unity route. Its start is not human-like: it holds a 0.3 to 0.6 m/s plateau for
  9 to 12 s, where a person takes about 3 s to reach 0.81 m/s.
- **Tuned spring.** It matches the human mean only by cruising below 0.81 m/s
  throughout. It never shows the human start-then-cruise profile.

## T7. Open loop, standard split

Protocol: STD, 1,441 test windows (walk and run), 5 seeds.

- **Selection.** With no previous plan, consistent selection reduces to goal-aware.
- **Followers.** Told each window's own label.
- **Values.** Mean ± std over seeds; the followers are deterministic.

Script: `f_eval.py base` runs `python -m gtp.rootmotion.evaluate` on each seed (→
`results/phase_f/open_base_s*.json`). The flow row is printed by `f_table.py`.

| Method | ADE m | FDE m | minADE@16 m | Stop error m | Fallback | W1 speed | W1 accel | W1 turn | Corridor violation |
|---|---|---|---|---|---|---|---|---|---|
| Flow, speed given, K = 16 | 0.413 ± 0.003 | 0.804 ± 0.011 | 0.214 ± 0.002 | 0.168 ± 0.008 | 0.3% ± 0.1 | 0.183 ± 0.004 | 0.316 ± 0.008 | 0.284 ± 0.011 | 0.0% |
| Spring, told the label | 0.521 | 0.996 | – | 0.059 | – | 0.259 | 0.515 | 0.183 | 6.2% |
| Constant speed, told the label | 0.798 | 1.380 | – | 0.091 | – | 0.503 | 1.026 | 0.654 | 0.0% |
| Capture (reference) | 0 | 0 | – | 0.104 | – | 0 | 0 | 0 | 0.0% |

Flow's walk-only ADE is 0.341 ± 0.004. This is the `ade_walk` field that
`gtp.rootmotion.evaluate` writes to `open_base_s*.json`, averaged over the 5 seeds;
`f_table.py` does not print it.

## T8. Choice of selection rule (supporting)

Protocol: STD, 5 seeds.

- **Synthetic routes.** 6 m, 12 m and Unity at 0.9 m/s, pooled.
- **Captured walks.** Actor 5's 10 walks; the short-walk rows use its 8 short walks.

Scripts: `e_experiments.py phase1|phase2`, then `e_analyze1.py` for the synthetic seams,
`e_analyze2.py` for the captured seams, and `e_checks.py` section 1 for the short-walk
ratios.

| | Goal-aware K = 16 | **Consistent K = 16** | Goal-aware K = 64 | Consistent K = 64 | Regressor |
|---|---|---|---|---|---|
| Seam / other accel, synthetic routes (m/s²) | 0.98 / 0.56 | 0.80 / 0.52 | 0.84 / 0.57 | 0.75 / 0.52 | 0.52 / 0.14 |
| Seam / other accel, captured walks (m/s²) | 1.33 / 0.82 | 1.24 / 0.82 | 1.43 / 0.86 | 1.34 / 0.84 | 1.08 / 0.37 |
| Short captured walks: arrival / capture | 1.19 ± 0.03 | 1.08 ± 0.05 | 1.12 ± 0.10 | 1.03 ± 0.04 | 1.25 ± 0.06 |
| Short captured walks: mean speed / capture | 0.74 | 0.77 | 0.75 | 0.76 | 0.63 |

- **Consistent vs goal-aware.** Consistent selection shortens arrival on short walks by
  about 9% and lowers seam jumps by 7 to 18%.
- **K = 64.** It brings no clear further gain and costs about twice the planning time
  of K = 16. That is a relative figure from the uncited timings listed under T10.
- **Seams.** They stay about 1.5 × the other steps.

## T9. Fixes for the starting plateau, screened and not adopted (supporting)

Protocol: STD, 5 seeds.

- **Captured walks:** actor 5's walks, consistent K = 16.
- **Synthetic routes:** time to 0.81 m/s at 0.9 m/s over 20 runs (5 checkpoints ×
  4 sampling seeds).
- **Open loop:** walk and run test windows.

Scripts: `f_train.py screen` trains the variants, `f_eval.py` and `f_table.py` give the
walk and open-loop columns, and `f_resample.py` gives the start times.

Success criteria were fixed in advance:

- time to 0.81 m/s within 1.35 to 4.8 s on both routes;
- short arrival / capture within 0.9 to 1.1;
- open-loop ADE ≤ 0.419 and fallback ≤ 0.5%.

| Variant | Short: arrival / capture | Long: arrival / capture | 12 m: median time to 0.81 | Unity: median time to 0.81 | Open ADE | Open fallback |
|---|---|---|---|---|---|---|
| Baseline (the method) | 1.07 ± 0.04 | 1.33 | 9.0 s | 11.5 s | 0.413 | 0.3% |
| A, speed guidance w = 1.5 / 2 / 3 | 1.03 / 1.01 / 0.99 | 1.29 / 1.21 / 1.14 | 9.9 / 9.4 / 9.1 s | 12.2 / 8.9 / 7.8 s | 0.408 / 0.412 / 0.432 | 0.3 / 0.4 / 1.4% |
| B1, B2: history noise | 1.08 / 1.13 | 1.35 / 1.43 | 9.9 / 9.9 s | 14.3 / 17.2 s | 0.426 / 0.440 | 0.3 / 0.2% |
| C5, C2: 5- / 2-step history | 1.04 / 1.03 | 1.29 / 1.30 | 9.4 / 2.3 s | 11.0 / 8.2 s | 0.415 / 0.437 | 0.2 / 0.1% |
| C2 + A2 | 0.98 ± 0.05 | 1.14 | 3.8 s | 9.2 s | 0.428 | 0.2% |

No variant meets all criteria.

- **2-step history.** Only it shortens the start on the straight, and its start times
  there are bimodal.
- **Guidance.** Costs about twice the planning time (two network passes per flow
  step; relative figure only, see T10).
- **Why the start resists.** The plateau is a near fixed point that already appears in
  open loop. Asked for 0.9 m/s, a real 0.3 to 0.6 m/s history gets a first second of
  about 0.55 m/s (`f_openloop_ask09.py`). The closed loop's own history slows it by only
  0.03 to 0.14 m/s more (`f_replan_bins.py`).

## T10. Planning latency (pending)

To be measured on an otherwise idle machine, with no Unity and no training running.

- **Script:** `scripts/rootmotion/latency.py`. It sends one request at a time through
  `PlanService.plan` in process.
- **Request:** a walk at 0.9 m/s with a moving 20-pose history, a 12 m route with one
  corner, and the unexecuted 2 s of the previous plan. The script checks that flow used
  the previous plan on every request.
- **Measurement:** 20 warm-up requests, then 200 timed requests per method, reporting
  P50 / P95 and the hardware.

| Method | Device | Requests | P50 ms | P95 ms | Hardware |
|---|---|---|---|---|---|
| Flow, consistent, K = 16 | CPU | 200 | pending | pending | pending |
| Regressor | CPU | 200 | pending | pending | pending |

Earlier figures are **not to be cited**. They came from different conditions, and none
of them timed consistent selection with a previous plan:

- **23 ms P50.** Open-loop `evaluate`, flow without speed conditioning, goal-aware.
- **37 to 38 ms.** `e_checks.py` section 3: median of 30, no previous plan, machine
  load unknown.
- **22 / 24 ms P50 / P95.** `f_timing.py`: CPU, 100 requests, no previous plan.
- **The "plan ms" column of `results/phase_f/tables.md`.** Measured while the GPU was
  shared.

## Superseded or corrected claims

Do not reuse these. Each entry gives the old claim, why it does not hold, and what
replaces it.

1. **"Flow beats the spring on arrival time and the speed distribution."**
   - *Old:* in LOAO, the untuned spring arrives at 0.73 to 0.74 of the person's time,
     and its speed W1 (0.190) loses to flow's (0.133). Earlier write-ups also said:
     "the open-loop advantage on the speed distribution does not survive closed loop;
     the advantages on acceleration and turn rate do".
   - *Why it does not hold:* this was against the untuned spring, which cruises at the
     p90 label. Scaling its cruise to 0.8 alone gives arrival 0.87 and speed W1 0.085.
   - *Replacement:* against the tuned spring, arrival and speed W1 tie (T5). Turn rate
     holds on short walks only, and acceleration holds everywhere.
2. **0.67 and 0.14 from capture windows.**
   - *Old:* "humans average about two thirds (0.67) of their label", "flow's
     mean / cruise of 0.65 matches the human 0.67", "from rest humans average 0.14 m/s
     in the first second", "humans enter the goal at 0.58 m/s".
   - *Why it does not hold:* these came from training windows, measured over the span
     up to the path's end (up to 25 m ahead). Those spans mix turning, dawdling before a
     start and walks of every length.
   - *Replacement:* length-matched rest-to-rest walks (HUM). They give 0.72 of the
     label, 0.5 s to 0.45 m/s, 2.8 s to 0.81 m/s and 0.59 m/s entering the last 0.25 m.
   - *Other 0.67s are different quantities:* HUM's mean speed is 0.67 m/s, and the
     goal-aware flow's mean speed / asked on the 12 m straight was 0.67.
3. **The open-loop plateau check mixed in high-speed labels.**
   - *Old:* "Open loop already slows down … The model is too slow on real histories as
     well, and drops off in the third second even when the history is fast. So the
     plateau is not only a closed-loop drift out of distribution: the open-loop plan is
     conservative, and closed loop starts each new plan from that slow output."
   - *Why it does not hold:* that check (`f_openloop_check.py`) used each window's own
     label. Those windows carry labels averaging 1.2 to 1.56 m/s, far above the
     0.9 m/s asked in closed loop, so the claim was stronger than the evidence.
   - *Replacement:* the comparable check asks every window for 0.9
     (`f_openloop_ask09.py`), which gives the near fixed point in T9. The closed loop's
     own history adds only 0.03 to 0.14 m/s of slowing. The own-label table must not be
     set against closed loop at 0.9 m/s.
4. **Goal-aware numbers presented as the method.**
   - *Old:* in `ROOT_MOTION_LAYER.md`, the "Flow at 0.9 m/s" table (8.5 / 16.7 s to
     0.81 m/s), the closed-loop tables at walk 0.9 and the open-loop "flow" row predate
     consistent selection or speed conditioning.
   - *Replacement:* T6 to T8.
5. **The test actor's 10 walks as the human reference for closed loop.** LOAO over all
   73 walks (T2 to T5) replaces it. Actor 5 turned out typical (T4).
6. **First version of the paired ratio test.**
   - *Old:* |x − 1| was taken after averaging flow's seeds, which slightly favours flow.
   - *Replacement:* T5 takes |x − 1| per run. No verdict changed.
7. **Latency figures.** See T10.

## Limitations

- **One dataset.** Only LAFAN1 locomotion: 18 clips and five actors. LOAO has 5 folds,
  so actor-level intervals are coarse.
- **Few long walks.** Long walks are only 12, one to three per actor. Their ties are
  weak evidence.
- **Small capture volume.** LAFAN1 was captured in a roughly 7 × 12 m volume, so long
  paths fold back on themselves.
- **Plateau on long straights.** From rest, flow holds a 0.3 to 0.6 m/s plateau for
  about 9 to 12 s (median) before cruising (T6), where a person needs about 3 s. None of
  the screened fixes removed it (T9).
- **Run is not usable.** With a target speed, the run cruise misses the ask by 0.4 to
  0.8 m/s on average. All results are walk only.
- **Root motion only.** There is no full-body layer: no foot sliding or contact metrics
  and no animation quality. There is no perceptual or user study either, so "human-like"
  means distributional closeness to the capture only.
- **Asymmetric tuning.** The tuned spring was optimised on two of the reported metrics;
  flow was not tuned on closed-loop metrics.
- **No engine test.** All closed-loop numbers come from the Python simulator through the
  service's code path. None comes from Unity's NavMeshAgent.

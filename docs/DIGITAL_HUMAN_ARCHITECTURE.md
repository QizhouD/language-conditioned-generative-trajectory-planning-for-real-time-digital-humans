# Digital-Human GTP Architecture

## Objective

GTP is the middle planning layer of a real-time digital-human stack. Given
recent motion, a high-level task, a local target, and nearby obstacles, it
generates several plausible future root trajectories and motion-intent
segments. It does not generate skeletal animation and does not replace the
runtime collision controller.

## System boundary

```text
user + scene
    -> SimpleTool: select behavior and arguments
    -> GTP: sample and rank root trajectories
    -> SimpleText2Motion: generate full-body SMPL-H motion
    -> motion adapter: warp root motion, blend clips, lock feet
    -> VRM / Unity / Unreal runtime
    -> updated scene state -> replan
```

SimpleTool answers *what should the character do?* GTP answers *where and how
should the root move over the next few seconds?* SimpleText2Motion answers *how
should the body articulate each frame?*

## Planning contract

All model-facing geometry uses the character-local frame. A training example
contains:

- `history`: recent root state at 10 Hz;
- `goal`: relative target position, desired facing, and stop distance;
- `obstacles`: local circular obstacle tokens;
- `future_trajectory`: relative per-step displacement, facing, velocity,
  angular velocity, and speed;
- `future_actions`: per-step motion intent;
- `instruction`: human-readable behavior description.

Trajectory features are:

```text
[delta_x, delta_z, sin_yaw, cos_yaw, velocity_x, velocity_z,
 angular_velocity, speed]
```

The initial intent vocabulary is:

```text
idle, walk, turn, stop, wave
```

The public runtime result is a `TrajectoryPlan` containing timestamped
world-space root points and semantic `MotionSegment` records. Prompts sent to a
text-to-motion backend are produced from segments, not directly from raw model
logits.

## Model

The conditional encoder fuses:

1. a Transformer over recent root states;
2. an MLP over the goal;
3. a Transformer over nearby obstacle tokens;
4. an embedding for the high-level behavior.

The flow model transports Gaussian trajectory noise to a future trajectory:

```text
x_t = (1 - t) * noise + t * trajectory
target velocity = trajectory - noise
```

At inference, Euler integration starts at noise and advances from `t=0` to
`t=1`. Multiple noise samples provide candidate diversity. A separate action
head predicts the motion intent at every future step.

Training combines flow velocity error, action cross-entropy, endpoint error,
and temporal smoothness. Collision safety is enforced again at runtime and is
never assumed from the learned loss alone.

## Candidate selection

Each candidate is converted from relative displacements to cumulative local
positions. Candidates are rejected when they:

- intersect an obstacle plus character clearance;
- exceed configured linear or angular velocity;
- leave the engine-provided navigation region;
- contain non-finite values.

Valid candidates are ranked by goal error, collision clearance, path length,
jerk, final-facing error, and action-transition cost. When no candidate is
valid, the planner returns a stationary fallback and requests replanning.

## Closed-loop execution

- history: 2 seconds;
- prediction horizon: 3 seconds;
- control rate: 10 Hz;
- replan rate: 2–5 Hz;
- executed prefix: 0.2–0.5 seconds;
- candidates per replan: 8–32;
- flow integration steps: 1–4.

Only the executed prefix is committed. Moving users, new obstacles, tracking
error, or controller deviation are incorporated in the next planning call.

## SimpleTool adapter

SimpleTool function calls are normalized into a versioned `PlanningGoal`.
Supported MVP behaviors are:

```text
move_to(target, speed, stop_distance)
approach(target, distance)
greet(target, hand, emotion)
look_at(target, duration)
idle(style, duration)
```

The planner never consumes arbitrary function-call JSON directly. Scene object
IDs must be resolved into local geometry before model inference.

## SimpleText2Motion adapter

The prompt compiler converts motion segments into bounded English prompts, for
example:

```text
A person walks forward naturally, gradually turns right, then stops
while facing another person and waves with the right hand.
```

Generated SMPL-H root translation is not trusted as the navigation path. The
motion adapter maps its motion phase and local gait variation onto the selected
GTP path, aligns forward direction to the path tangent, blends segment
boundaries, and applies contact-aware foot locking.

## Data roadmap

1. Procedural rooms, circular obstacles, and classical reference paths.
2. Unity/Unreal NavMesh, behavior-tree, and player-control trajectories.
3. Consented VR or motion-capture interaction data.
4. Closed-loop logs containing planner proposals, selected candidates,
   controller corrections, and failure labels.

Procedural pretraining is followed by engine-data fine-tuning and, only when
consent and governance are in place, human-interaction fine-tuning.

## Evaluation

Trajectory quality:

- average and final displacement error;
- final-facing error;
- path-length ratio, acceleration, and jerk;
- sample diversity.

Safety and validity:

- collision rate and minimum clearance;
- navigation-region violations;
- speed and angular-speed violations;
- fallback and planning-failure rates.

Animation and interaction:

- foot sliding and ground penetration;
- segment-boundary discontinuity;
- text-motion and trajectory-motion consistency;
- perceived naturalness, responsiveness, and intent alignment.

Systems:

- SimpleTool, GTP sampling, candidate ranking, text-to-motion, adaptation, and
  end-to-end P50/P95 latency.

## MVP acceptance criteria

The initial demo accepts an instruction such as “walk around the table and
greet the visitor,” resolves the target, samples several paths, selects a safe
trajectory, produces walk/stop/wave segments, and replans when the visitor
moves. The target is at least 2 Hz closed-loop planning, zero navigation-region
violations under the hard controller, and less than one percent collision rate
in the procedural benchmark.

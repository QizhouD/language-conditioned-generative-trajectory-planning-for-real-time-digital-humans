# Generative Trajectory Planning for Digital Humans

Research MVP for language-conditioned, collision-aware future root-trajectory
generation for real-time digital humans.

The planner consumes recent character motion, a local goal, nearby obstacles,
and a high-level behavior. It samples several possible 2D root trajectories,
rejects unsafe candidates, ranks the remainder, and executes only the first
part before replanning. Full-body animation is delegated to a text-to-motion
system such as SimpleText2Motion.

See [docs/DIGITAL_HUMAN_ARCHITECTURE.md](docs/DIGITAL_HUMAN_ARCHITECTURE.md)
for the system design and integration contract.

## Embodied demo

[Unity-digital-Human](https://github.com/QizhouD/LLM-NPC) is the embodiment layer
for this planner: a Unity 6 scene where a language model answers the player in
prose, tags each reply with one of the five motion intents below, and that intent
becomes a plan request whose result drives the character's root transform at
10 Hz. It mirrors `TrajectoryPlan` from
[src/gtp/integrations/plan_export.py](src/gtp/integrations/plan_export.py) field
for field in C#, so the two repositories share one contract.

That demo currently runs on a procedural mock plan source, because every export
from the trained checkpoint is a constraint-filter fallback. Fixing that is the
top item in the training handoff.

For training status, experimental 2D loss functions, and continuation instructions,
see [docs/TRAINING_HANDOFF.md](docs/TRAINING_HANDOFF.md).

## MVP scope

- Local-frame planar root trajectories at 10 Hz.
- Five motion intents: idle, walk, turn, stop, and wave.
- Conditional flow matching and a deterministic Transformer baseline.
- Procedural navigation data with circular obstacles.
- Hard safety constraints, multi-candidate ranking, and rolling replanning.
- Stable contracts for SimpleTool and SimpleText2Motion adapters.

## Setup

Python 3.11+ is recommended:

```powershell
python -m pip install -e ".[dev]"
```

## Generate data

```powershell
python -m gtp.data.synthetic_navigation `
  --output data/synthetic_navigation.jsonl `
  --episodes 1000 `
  --seed 7
```

## Train

```powershell
python -m gtp.train --config configs/digital_human_flow.yaml
```

For a short CPU smoke run:

```powershell
python -m gtp.train --config configs/digital_human_flow_cpu.yaml
```

## Evaluate

```powershell
python -m gtp.evaluate `
  --checkpoint runs/digital_human_flow/best.pt `
  --dataset data/synthetic_navigation.jsonl
```

The evaluator reports ADE/FDE, final-facing error, collision and kinematic
violation rates, action accuracy, diversity, and P50/P95 latency.

## Scientific guardrails

- Generated trajectories are proposals, not safety guarantees.
- Collision and kinematic constraints must remain outside the neural model.
- Compare against the deterministic baseline and a classical planner.
- Report hardware, seeds, P50/P95 latency, and failure rates.
- Validate root-motion warping and foot locking before user-facing deployment.

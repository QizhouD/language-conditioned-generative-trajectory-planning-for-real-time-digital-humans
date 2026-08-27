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

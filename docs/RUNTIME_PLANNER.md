# Runtime Planner

`RuntimePlanner` closes the in-process path from a SimpleTool function call to
a world-space `TrajectoryPlan`.

```text
SimpleTool call
    -> resolve target object
    -> PlanningGoal in character-local coordinates
    -> model batch from history, goal, and obstacles
    -> conditional-flow candidate sampling
    -> collision and kinematic filtering
    -> world-space trajectory and Text2Motion prompts
```

## Python API

```python
from gtp.runtime import RuntimePlanner
from gtp.schema import MotionState, Obstacle

planner = RuntimePlanner.from_checkpoint(
    "runs/digital_human_flow_cpu/best.pt",
    candidates=16,
    flow_steps=4,
)

plan = planner.plan_function_call(
    {
        "function": "greet_person",
        "arguments": {
            "target_id": "visitor_7",
            "preferred_distance": 1.2,
        },
    },
    {
        "visitor_7": {
            "position": [3.5, 0.0, 2.0],
            "yaw": 3.14,
        }
    },
    history=[
        MotionState(
            position=[0.0, 0.0, 0.0],
            yaw=0.0,
            velocity=[0.0, 0.0],
            angular_velocity=0.0,
            action=0,
            action_phase=0.0,
            foot_contacts=[1.0, 1.0],
        )
    ],
    obstacles=[Obstacle(center=[1.5, 0.5], radius=0.35)],
    character_position=[0.0, 0.0, 0.0],
    character_yaw=0.0,
)
```

History state and obstacle coordinates are character-local. Scene-object
positions passed to the SimpleTool adapter are world-space and are normalized
before inference.

## JSON request CLI

```powershell
python -m gtp.runtime_request `
  --checkpoint runs/digital_human_flow_cpu/best.pt `
  --request examples/greet_request.json `
  --output results/greet_plan.json `
  --candidates 8 `
  --flow-steps 4
```

The two-epoch CPU checkpoint is a pipeline fixture, not a useful behavior
policy. Its generated trajectories will normally trigger the safe stationary
fallback. Train the full configuration before evaluating motion quality.

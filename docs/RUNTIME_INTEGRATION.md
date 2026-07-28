# Runtime Integration

This milestone turns model tensors into a stable boundary for SimpleTool,
SimpleText2Motion, and a game-engine controller.

## 1. Normalize a SimpleTool call

```python
from gtp.integrations.simpletool_adapter import planning_goal_from_function_call

goal = planning_goal_from_function_call(
    {
        "function": "greet_person",
        "arguments": {
            "target_id": "visitor_7",
            "preferred_distance": 1.2,
        },
    },
    {
        "visitor_7": {
            "position": [4.0, 0.0, 2.0],
            "yaw": 3.14,
        }
    },
    character_position=[0.0, 0.0, 0.0],
    character_yaw=0.0,
)
```

The adapter validates the function and target, resolves scene IDs, converts
world geometry to the character-local frame, and returns a `PlanningGoal`.
Arbitrary tool JSON is therefore kept outside the model contract.

Supported functions are:

```text
move_to, approach, greet, greet_person, look_at, idle
```

## 2. Export a runtime plan

```python
from gtp.integrations.plan_export import build_trajectory_plan, write_plan

plan = build_trajectory_plan(
    trajectory=model_trajectory.tolist(),
    actions=predicted_actions.tolist(),
    origin=[character_x, character_y, character_z],
    origin_yaw=character_yaw,
    score=candidate_cost,
    fallback=used_safe_fallback,
)
write_plan("results/runtime_plan.json", plan)
```

The exporter accumulates local displacements, transforms positions, facing,
and velocity into world space, groups action labels into motion segments, and
attaches bounded English prompts for a text-to-motion backend.

## 3. Checkpoint-to-JSON demo

```powershell
python -m gtp.demo `
  --checkpoint runs/digital_human_flow_cpu/best.pt `
  --dataset data/pilot_navigation.jsonl `
  --output results/runtime_plan.json `
  --candidates 8 `
  --flow-steps 4 `
  --origin 0 0 0 `
  --origin-yaw 0
```

The resulting JSON is the handoff contract for a browser VRM client, Unity,
Unreal, or a motion-adaptation service. Generated plans remain proposals; the
engine must continue to enforce collision, navigation-region, and kinematic
constraints during execution.

# Remote Model Services

GTP now has standard-library HTTP clients for the upstream SimpleTool and
SimpleText2Motion servers. No extra HTTP dependency is required.

## SimpleTool

Start the upstream server on its default port, then:

```python
from gtp.integrations.simpletool_client import SimpleToolClient

client = SimpleToolClient("http://127.0.0.1:8899")
print(client.health())

call = client.function_call(
    "Walk to visitor_7 and greet them with your right hand.",
    environment=[
        "visitor_7: position=[3.5, 0.0, 2.0], yaw=3.14",
    ],
)
```

The upstream response uses `args`; the client normalizes it to:

```json
{
  "function": "greet_person",
  "arguments": {
    "target_id": "visitor_7",
    "preferred_distance": 1.2
  },
  "latency_ms": 31.5
}
```

The default digital-human tool vocabulary contains `move_to`, `approach`,
`greet_person`, `look_at`, and `idle`.

## SimpleTool to GTP

```python
from gtp.remote_pipeline import RemoteRuntimePipeline
from gtp.runtime import RuntimePlanner

pipeline = RemoteRuntimePipeline(
    SimpleToolClient("http://127.0.0.1:8899"),
    RuntimePlanner.from_checkpoint(
        "runs/digital_human_flow/best.pt",
        candidates=16,
        flow_steps=4,
    ),
)

plan = pipeline.plan_command(
    "Go greet the visitor.",
    scene_objects={
        "visitor_7": {
            "position": [3.5, 0.0, 2.0],
            "yaw": 3.14,
        }
    },
    motion_history=local_motion_history,
    obstacles=local_obstacles,
    character_position=[0.0, 0.0, 0.0],
    character_yaw=0.0,
)
```

## SimpleText2Motion

The upstream `run_server` endpoint accepts `{prompt, duration, seed?}` at
`POST /t2m` and returns binary VRMA/GLB data.

```python
from gtp.integrations.simplet2m_client import SimpleText2MotionClient

t2m = SimpleText2MotionClient(
    "http://127.0.0.1:8765",
    timeout=120,
)
asset = t2m.generate(
    "A person stops, faces the visitor, and waves with the right hand.",
    duration=3.0,
    seed=7,
)
asset.write("results/wave.vrma")
```

The client verifies HTTP status, `model/gltf-binary`, and the GLB `glTF`
signature before returning a `MotionAsset`.

## Failure behavior

Connection failures, non-2xx responses, invalid JSON, unsuccessful function
selection, unexpected content types, and invalid VRMA bytes raise
`ServiceError`. Engine integrations should catch this exception and retain the
current safe idle behavior rather than executing a partial remote result.

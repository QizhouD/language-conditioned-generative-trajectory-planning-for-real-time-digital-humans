import json

import pytest

from gtp.integrations.http_transport import HttpResult, ServiceError
from gtp.integrations.simplet2m_client import SimpleText2MotionClient
from gtp.integrations.simpletool_client import SimpleToolClient
from gtp.remote_pipeline import RemoteRuntimePipeline


def test_simpletool_client_normalizes_args():
    captured = {}

    def transport(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return HttpResult(
            200,
            {"Content-Type": "application/json"},
            json.dumps(
                {
                    "success": True,
                    "function": "greet_person",
                    "args": {"target_id": "visitor", "preferred_distance": 1.2},
                    "latency_ms": 31.5,
                }
            ).encode(),
        )

    client = SimpleToolClient("http://simpletool:8899/", timeout=5, transport=transport)
    call = client.function_call("Greet the visitor")
    assert captured["url"] == "http://simpletool:8899/v1/function_call"
    assert captured["timeout"] == 5
    assert captured["payload"]["messages"][0]["content"] == "Greet the visitor"
    assert len(captured["payload"]["tools"]) == 5
    assert call == {
        "function": "greet_person",
        "arguments": {"target_id": "visitor", "preferred_distance": 1.2},
        "latency_ms": 31.5,
    }


def test_simpletool_client_reports_model_failure():
    def transport(request, timeout):
        return HttpResult(
            200,
            {"Content-Type": "application/json"},
            b'{"success":false,"error":"No function called"}',
        )

    with pytest.raises(ServiceError, match="No function called"):
        SimpleToolClient(transport=transport).function_call("do something")


def test_simplet2m_client_validates_and_writes_vrma(tmp_path):
    captured = {}
    glb = b"glTF" + bytes(32)

    def transport(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return HttpResult(200, {"Content-Type": "model/gltf-binary"}, glb)

    client = SimpleText2MotionClient(transport=transport)
    asset = client.generate(
        "A person waves with the right hand.", duration=2.5, seed=7
    )
    assert captured["url"] == "http://127.0.0.1:8765/t2m"
    assert captured["payload"]["duration"] == 2.5
    assert asset.data == glb
    output = asset.write(tmp_path / "wave.vrma")
    assert output.read_bytes() == glb


def test_simplet2m_client_rejects_text_error_response():
    def transport(request, timeout):
        return HttpResult(500, {"Content-Type": "text/plain"}, b"t2m unavailable")

    with pytest.raises(ServiceError, match="t2m unavailable"):
        SimpleText2MotionClient(transport=transport).generate("A person walks.")


def test_remote_pipeline_passes_normalized_call_to_planner():
    class ToolStub:
        def function_call(self, command, **kwargs):
            assert command == "Go greet the visitor"
            assert kwargs["environment"][0].startswith("visitor:")
            return {
                "function": "greet_person",
                "arguments": {"target_id": "visitor"},
            }

    class PlannerStub:
        def plan_function_call(self, call, scene_objects, history, obstacles, **kwargs):
            assert call["function"] == "greet_person"
            assert "visitor" in scene_objects
            assert kwargs["character_yaw"] == 0.0
            return "planned"

    pipeline = RemoteRuntimePipeline(ToolStub(), PlannerStub())
    result = pipeline.plan_command(
        "Go greet the visitor",
        scene_objects={"visitor": {"position": [2.0, 0.0, 1.0]}},
        motion_history=[],
        obstacles=[],
        character_position=[0.0, 0.0, 0.0],
        character_yaw=0.0,
    )
    assert result == "planned"

import json
import math

import pytest

from gtp.integrations.plan_export import build_trajectory_plan, write_plan
from gtp.integrations.simpletool_adapter import planning_goal_from_function_call


def test_simpletool_call_resolves_to_local_goal():
    goal = planning_goal_from_function_call(
        {
            "function": "greet_person",
            "arguments": {"target_id": "visitor", "preferred_distance": 1.2},
        },
        {"visitor": {"position": [10.0, 0.0, 7.0], "yaw": 0.0}},
        character_position=[10.0, 0.0, 5.0],
        character_yaw=math.pi / 2,
    )
    assert goal.behavior == "wave"
    assert goal.stop_distance == 1.2
    assert goal.target_position == pytest.approx([-2.0, 0.0, 0.0], abs=1e-6)
    assert goal.target_yaw == pytest.approx(-math.pi / 2)


def test_runtime_plan_world_transform_and_json_export(tmp_path):
    point = [1.0, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 10.0]
    plan = build_trajectory_plan(
        [point, point],
        [1, 3],
        origin=[10.0, 0.0, 5.0],
        origin_yaw=math.pi / 2,
        control_hz=10,
        plan_id="plan-test",
        score=2.5,
    )
    assert plan.horizon == 0.2
    assert plan.trajectory[-1]["position"] == pytest.approx([10.0, 0.0, 3.0])
    assert [segment.action for segment in plan.segments] == ["walk", "stop"]

    output = tmp_path / "plan.json"
    write_plan(output, plan)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["plan_id"] == "plan-test"
    assert document["segments"][0]["prompt"] == "A person walks forward naturally."

from gtp.integrations.simpletool_adapter import planning_goal_from_function_call
from gtp.integrations.simpletool_client import DIGITAL_HUMAN_TOOLS


def test_default_tool_parameters_share_one_positional_vocabulary():
    canonical = ["target_id", "distance", "hand", "emotion"]
    for tool in DIGITAL_HUMAN_TOOLS:
        properties = tool["function"]["parameters"]["properties"]
        names = list(properties)
        assert names == canonical[: len(names)]


def test_shared_distance_parameter_maps_to_planning_stop_distance():
    goal = planning_goal_from_function_call(
        {
            "function": "greet_person",
            "arguments": {"target_id": "visitor", "distance": 1.25},
        },
        {"visitor": {"position": [2.0, 0.0, 0.0], "yaw": 0.0}},
        character_position=[0.0, 0.0, 0.0],
        character_yaw=0.0,
    )
    assert goal.stop_distance == 1.25

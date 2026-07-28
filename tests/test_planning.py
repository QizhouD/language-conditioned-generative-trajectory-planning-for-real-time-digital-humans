import torch

from gtp.integrations.prompt_compiler import compile_segments
from gtp.planning.constraints import collision_mask, cumulative_positions


def test_collision_check_and_prompt_segments():
    safe = torch.zeros(1, 4, 8)
    safe[..., 0] = 0.5
    safe[..., 3] = 1.0
    positions = cumulative_positions(safe)
    assert positions[0, -1, 0].item() == 2.0
    obstacles = torch.tensor([[[1.0, 0.0, 0.2, 0.0]]])
    valid = torch.tensor([[True]])
    assert collision_mask(safe, obstacles, valid).item()

    segments = compile_segments([1, 1, 3, 3, 4])
    assert [segment.action for segment in segments] == ["walk", "stop", "wave"]

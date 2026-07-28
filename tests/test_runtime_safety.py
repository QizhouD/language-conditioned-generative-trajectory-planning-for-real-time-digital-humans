import torch

from gtp.planning.constraints import kinematic_violation_mask


def test_kinematic_check_uses_displacement_and_rejects_negative_speed():
    trajectory = torch.zeros(2, 3, 8)
    trajectory[..., 3] = 1.0
    trajectory[0, :, 0] = 1.0
    trajectory[1, :, 7] = -0.1
    assert kinematic_violation_mask(trajectory).tolist() == [True, True]

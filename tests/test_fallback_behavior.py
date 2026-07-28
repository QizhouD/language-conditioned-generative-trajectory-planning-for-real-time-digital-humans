import torch

from gtp.planning import RollingTrajectoryPlanner


class UnsafeModel:
    def sample(self, batch, steps, samples):
        count = batch["history"].shape[0] * samples
        trajectory = torch.zeros(count, 2, 8)
        trajectory[..., 0] = 1.0
        trajectory[..., 3] = 1.0
        trajectory[..., 7] = 0.1
        logits = torch.zeros(count, 2, 5)
        logits[..., 1] = 10.0
        return {"trajectory": trajectory, "action_logits": logits}


def test_unsafe_candidate_falls_back_to_idle():
    batch = {
        "history": torch.zeros(1, 1, 12),
        "goal": torch.tensor([[2.0, 0.0, 0.0, 1.0, 0.0]]),
        "obstacles": torch.zeros(1, 1, 4),
        "obstacle_valid": torch.tensor([[False]]),
    }
    result = RollingTrajectoryPlanner(
        UnsafeModel(), candidates=1, flow_steps=1
    ).plan(batch)
    assert result["fallback"].tolist() == [True]
    assert result["trajectory"].abs().sum().item() == 2.0
    assert result["actions"].tolist() == [[0, 0]]

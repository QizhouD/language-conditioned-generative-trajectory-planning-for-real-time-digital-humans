import torch

from gtp.data.synthetic_navigation import write_dataset
from gtp.dataset import DigitalHumanTrajectoryDataset, collate_trajectories
from gtp.models import ConditionalTrajectoryFlow
from gtp.planning import RollingTrajectoryPlanner


def test_flow_model_shapes(tmp_path):
    path = tmp_path / "data.jsonl"
    write_dataset(path, episodes=3, seed=4)
    dataset = DigitalHumanTrajectoryDataset(path)
    batch = collate_trajectories([dataset[0], dataset[1]])
    model = ConditionalTrajectoryFlow(
        state_dim=12, goal_dim=5, obstacle_dim=4, trajectory_dim=8,
        hidden_dim=32, layers=1, heads=4, horizon_steps=30,
    )
    state = torch.randn(2, 30, 8)
    velocity = model.velocity(state, torch.rand(2), batch)
    assert velocity.shape == (2, 30, 8)
    output = model.sample(batch, steps=1, samples=2)
    assert output["trajectory"].shape == (4, 30, 8)
    assert torch.isfinite(output["trajectory"]).all()

    plan = RollingTrajectoryPlanner(
        model, candidates=2, flow_steps=1, execute_steps=3
    ).plan(batch)
    assert plan["trajectory"].shape == (2, 30, 8)
    assert plan["execute_prefix"].shape == (2, 3, 8)

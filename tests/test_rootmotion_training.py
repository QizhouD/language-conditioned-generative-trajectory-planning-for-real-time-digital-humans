from pathlib import Path

import torch
import yaml
from torch import nn

from gtp.rootmotion import HISTORY_STEPS, HORIZON_STEPS, MAX_PATH_VERTICES
from gtp.rootmotion.model import RootMotionModel
from gtp.rootmotion.stats import RootMotionStats
from gtp.rootmotion.train import batch_loss, position_loss, position_loss_weight, validation_loss
from gtp.rootmotion.windows import HISTORY_DIM, PATH_DIM, POSITION_SCALE, trajectory_positions

STATS = RootMotionStats(0.095, 0.105, (0.9, 2.0, 2.5))
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def model(deterministic):
    torch.manual_seed(0)
    return RootMotionModel(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2,
                           deterministic=deterministic)


def batch(size=4):
    generator = torch.Generator().manual_seed(7)
    valid = torch.zeros(size, MAX_PATH_VERTICES, dtype=torch.bool)
    valid[:, :2] = True
    return {
        "history": torch.randn(size, HISTORY_STEPS, HISTORY_DIM, generator=generator),
        "path": torch.randn(size, MAX_PATH_VERTICES, PATH_DIM, generator=generator),
        "path_valid": valid,
        "gait": torch.zeros(size, dtype=torch.long),
        "target": torch.randn(size, HORIZON_STEPS, 3, generator=generator) * 0.1,
    }


def reference_loss(net, data, generator):
    """The loss as it was before position_loss_weight existed."""
    target = STATS.normalise(data["target"])
    if net.deterministic:
        return nn.functional.mse_loss(net.predict(data), target)
    noise = torch.randn(target.shape, generator=generator, device=target.device)
    time = torch.rand(target.shape[0], generator=generator, device=target.device)
    mix = time[:, None, None]
    state = (1.0 - mix) * noise + mix * target
    return nn.functional.mse_loss(net.velocity(state, time, data), target - noise)


def test_zero_position_weight_is_bit_identical_to_the_old_loss():
    data = batch()
    for deterministic in (False, True):
        net = model(deterministic)
        expected = reference_loss(net, data, torch.Generator().manual_seed(3))
        default = batch_loss(net, data, STATS, torch.Generator().manual_seed(3))
        zero = batch_loss(net, data, STATS, torch.Generator().manual_seed(3), position_weight=0.0)
        assert torch.equal(default, expected) and torch.equal(zero, expected)


def test_existing_configs_have_no_position_term():
    for name in ("rootmotion_flow_cpu.yaml", "rootmotion_regressor_cpu.yaml", "rootmotion_flow.yaml"):
        config = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
        assert position_loss_weight(config) == 0.0
    assert position_loss_weight({"training": {"position_loss_weight": 2}}) == 2.0


def test_position_loss_accumulates_the_deltas():
    target = torch.zeros(1, HORIZON_STEPS, 3)
    target[..., 1] = 0.1
    # A constant 1 cm/step lateral error grows into a 30 cm error at the end of the plan.
    shifted = target.clone()
    shifted[..., 0] = 0.01
    expected = (((torch.arange(1, HORIZON_STEPS + 1) * 0.01) / POSITION_SCALE) ** 2).sum() / (2 * HORIZON_STEPS)
    assert abs(float(position_loss(STATS.normalise(shifted), target, STATS)) - float(expected)) < 1e-9
    assert float(position_loss(STATS.normalise(target), target, STATS)) < 1e-12


def test_positive_weight_adds_the_position_term_and_reports_the_parts():
    data = batch()
    for deterministic in (False, True):
        net = model(deterministic)
        base = batch_loss(net, data, STATS, torch.Generator().manual_seed(3))
        parts = {}
        total = batch_loss(net, data, STATS, torch.Generator().manual_seed(3), position_weight=2.0, parts=parts)
        assert set(parts) == {"base", "position"}
        assert abs(parts["base"] - base.item()) < 1e-6
        assert abs(total.item() - (parts["base"] + 2.0 * parts["position"])) < 1e-5
        assert parts["position"] > 0.0


def test_flow_position_term_uses_the_clean_sample_implied_by_the_velocity():
    class Perfect(RootMotionModel):
        """Predicts exactly target - noise, so x1_hat equals the target."""

        def velocity(self, state, time, data):
            return self._target - self._noise

    torch.manual_seed(0)
    net = Perfect(hidden_dim=32, encoder_layers=1, decoder_layers=1, heads=2)
    data = batch(2)
    generator = torch.Generator().manual_seed(5)
    net._target = STATS.normalise(data["target"])
    net._noise = torch.randn(net._target.shape, generator=torch.Generator().manual_seed(5))
    parts = {}
    batch_loss(net, data, STATS, generator, position_weight=1.0, parts=parts)
    assert parts["base"] < 1e-12 and parts["position"] < 1e-10


def test_validation_loss_reports_the_parts():
    class Data:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values["target"])

        def as_batch(self):
            return self.values

    parts = {}
    value = validation_loss(model(True), Data(batch()), STATS, 2, torch.device("cpu"),
                            position_weight=0.5, parts=parts)
    assert abs(value - (parts["base"] + 0.5 * parts["position"])) < 1e-5
    assert trajectory_positions(torch.zeros(1, HORIZON_STEPS, 3)).shape == (1, HORIZON_STEPS + 1, 2)

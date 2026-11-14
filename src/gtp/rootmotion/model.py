"""Path-conditioned root-motion models.

With ``speed_conditioning`` the gait token also carries a target speed, or a learned
"unspecified" vector when none is given (training drops the speed 10% of the time
so both work).

The network generates per-step (dx, dz, dyaw) in normalised units. Three choices
matter, each fixing a failure of the earlier navigation MVP:

* targets are normalised, so signal and noise share a scale;
* the output is a sequence of time tokens, not one flattened vector, so the
  model can express temporal smoothness;
* path and history stay as token sets read through cross-attention, instead of
  being mean-pooled into a single vector that loses geometry.

Speed, velocity and angular velocity are derived from the three generated
channels, which keeps every exported plan kinematically self-consistent.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from gtp.rootmotion import GAITS, HISTORY_STEPS, HORIZON_STEPS, MAX_PATH_VERTICES
from gtp.rootmotion.windows import HISTORY_DIM, PATH_DIM, POSITION_SCALE, TARGET_DIM


def timestep_embedding(time: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        torch.arange(half, device=time.device) * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = time[:, None] * 1_000.0 * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < width:
        embedding = nn.functional.pad(embedding, (0, width - embedding.shape[-1]))
    return embedding


class RootMotionModel(nn.Module):
    """Conditional flow model, or a deterministic regressor when ``deterministic``."""

    def __init__(
        self, hidden_dim: int = 128, encoder_layers: int = 2, decoder_layers: int = 4,
        heads: int = 4, dropout: float = 0.0, deterministic: bool = False,
        horizon_steps: int = HORIZON_STEPS, history_steps: int = HISTORY_STEPS,
        max_vertices: int = MAX_PATH_VERTICES, speed_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon_steps = horizon_steps
        self.deterministic = deterministic
        self.speed_conditioning = speed_conditioning
        if speed_conditioning:
            # Created only when enabled, so checkpoints without it load unchanged.
            self.speed_in = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            self.speed_unspecified = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.normal_(self.speed_unspecified, std=0.02)
        self.history_in = nn.Linear(HISTORY_DIM, hidden_dim)
        self.path_in = nn.Linear(PATH_DIM, hidden_dim)
        self.gait_embedding = nn.Embedding(len(GAITS), hidden_dim)
        self.history_position = nn.Parameter(torch.zeros(history_steps, hidden_dim))
        self.path_position = nn.Parameter(torch.zeros(max_vertices, hidden_dim))
        self.kind = nn.Parameter(torch.zeros(3, hidden_dim))
        self.future_position = nn.Parameter(torch.zeros(horizon_steps, hidden_dim))
        for parameter in (self.history_position, self.path_position, self.kind, self.future_position):
            nn.init.normal_(parameter, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers, enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.state_in = nn.Linear(TARGET_DIM, hidden_dim)
        self.time_in = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, TARGET_DIM)

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (memory, padding mask) for history, path and gait tokens.

        Only the last ``history_steps`` history tokens are read, so a model built with a
        shorter history works on the same 20-step requests and windows.
        """
        recent = batch["history"][:, -self.history_position.shape[0]:]
        history = self.history_in(recent) + self.history_position + self.kind[0]
        path = self.path_in(batch["path"]) + self.path_position + self.kind[1]
        gait = self.gait_embedding(batch["gait"]) + self.kind[2]
        if self.speed_conditioning:
            gait = gait + self.speed_embedding(batch)
        gait = gait.unsqueeze(1)
        tokens = torch.cat((history, path, gait), dim=1)
        present = torch.ones(history.shape[:2], dtype=torch.bool, device=tokens.device)
        padding = torch.cat((~present, ~batch["path_valid"], ~present[:, :1]), dim=1)
        return self.encoder(tokens, src_key_padding_mask=padding), padding

    def speed_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Target speed (m/s, divided by POSITION_SCALE) or the learned "unspecified" vector.

        A missing ``target_speed`` key, or a non-finite value (NaN), means unspecified.
        """
        count = batch["gait"].shape[0]
        unspecified = self.speed_unspecified.expand(count, -1)
        speed = batch.get("target_speed")
        if speed is None:
            return unspecified
        speed = speed.reshape(count, 1).to(unspecified.dtype)
        given = torch.isfinite(speed)
        encoded = self.speed_in(torch.where(given, speed, torch.zeros_like(speed)) / POSITION_SCALE)
        return torch.where(given, encoded, unspecified)

    def decode(
        self, state: torch.Tensor, time: torch.Tensor,
        memory: torch.Tensor, padding: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.state_in(state) + self.future_position
        tokens = tokens + self.time_in(timestep_embedding(time, self.hidden_dim)).unsqueeze(1)
        decoded = self.decoder(tokens, memory, memory_key_padding_mask=padding)
        return self.out(self.out_norm(decoded))

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Deterministic regression of the normalised future."""
        memory, padding = self.encode(batch)
        count = memory.shape[0]
        state = torch.zeros(count, self.horizon_steps, TARGET_DIM, device=memory.device)
        return self.decode(state, torch.ones(count, device=memory.device), memory, padding)

    def velocity(
        self, state: torch.Tensor, time: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        memory, padding = self.encode(batch)
        return self.decode(state, time, memory, padding)

    @torch.no_grad()
    def sample(
        self, batch: dict[str, torch.Tensor], steps: int = 8, samples: int = 1,
        generator: torch.Generator | None = None, guidance: float = 1.0,
    ) -> torch.Tensor:
        """Draw ``samples`` normalised futures per batch row: [B, samples, T, 3].

        ``guidance`` w != 1 applies classifier-free guidance on the target speed (speed-
        conditioned models only): every prediction becomes u + w (c - u), with c conditioned
        on the given speed and u on "unspecified". Rows without a speed get c = u, so they
        are unchanged. It costs a second decoder pass per flow step.
        """
        guided = self.speed_conditioning and guidance != 1.0 and "target_speed" in batch
        memory, padding = self.encode(batch)
        if guided:
            unconditioned = {**batch, "target_speed": torch.full_like(batch["target_speed"], float("nan"))}
            memory_u, _ = self.encode(unconditioned)
        batch_size = memory.shape[0]
        if self.deterministic:
            state = torch.zeros(batch_size, self.horizon_steps, TARGET_DIM, device=memory.device)
            ones = torch.ones(batch_size, device=memory.device)
            out = self.decode(state, ones, memory, padding)
            if guided:
                base = self.decode(state, ones, memory_u, padding)
                out = base + guidance * (out - base)
            return out.unsqueeze(1)
        memory = memory.repeat_interleave(samples, dim=0)
        padding = padding.repeat_interleave(samples, dim=0)
        if guided:
            memory_u = memory_u.repeat_interleave(samples, dim=0)
        count = batch_size * samples
        state = torch.randn(
            count, self.horizon_steps, TARGET_DIM, generator=generator, device=memory.device
        )
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full((count,), index * delta, device=memory.device)
            velocity = self.decode(state, time, memory, padding)
            if guided:
                base = self.decode(state, time, memory_u, padding)
                velocity = base + guidance * (velocity - base)
            state = state + delta * velocity
        return state.view(batch_size, samples, self.horizon_steps, TARGET_DIM)

"""Serializable contracts for digital-human trajectory planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MOTION_INTENTS = ("idle", "walk", "turn", "stop", "wave")
STATE_DIM = 12
GOAL_DIM = 5
OBSTACLE_DIM = 4
TRAJECTORY_DIM = 8


@dataclass(frozen=True)
class MotionState:
    """Character-local root state sampled at the control rate."""

    position: list[float]
    yaw: float
    velocity: list[float]
    angular_velocity: float
    action: int
    action_phase: float
    foot_contacts: list[float]

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.velocity) != 2:
            raise ValueError("position must be xyz and velocity must be xz")
        if len(self.foot_contacts) != 2:
            raise ValueError("foot_contacts must contain left and right values")
        if self.action not in range(len(MOTION_INTENTS)):
            raise ValueError("action is outside the motion-intent vocabulary")

    def features(self) -> list[float]:
        return [
            *self.position,
            self.yaw,
            *self.velocity,
            self.angular_velocity,
            self.action / max(len(MOTION_INTENTS) - 1, 1),
            self.action_phase,
            *self.foot_contacts,
            1.0,
        ]


@dataclass(frozen=True)
class PlanningGoal:
    behavior: str
    target_position: list[float]
    target_yaw: float
    stop_distance: float = 0.0
    instruction: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.behavior not in MOTION_INTENTS:
            raise ValueError(f"unsupported behavior: {self.behavior}")
        if len(self.target_position) != 3:
            raise ValueError("target_position must be xyz")
        if self.stop_distance < 0:
            raise ValueError("stop_distance cannot be negative")


@dataclass(frozen=True)
class Obstacle:
    center: list[float]
    radius: float
    kind: int = 0

    def __post_init__(self) -> None:
        if len(self.center) != 2 or self.radius <= 0:
            raise ValueError("obstacle requires an xz center and positive radius")


@dataclass(frozen=True)
class TrajectoryExample:
    episode_id: str
    history: list[MotionState]
    goal: PlanningGoal
    obstacles: list[Obstacle]
    future_trajectory: list[list[float]]
    future_actions: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.history or not self.future_trajectory:
            raise ValueError("history and future trajectory must be non-empty")
        if len(self.future_trajectory) != len(self.future_actions):
            raise ValueError("future trajectory and actions must have equal length")
        if any(len(point) != TRAJECTORY_DIM for point in self.future_trajectory):
            raise ValueError(f"trajectory points must have {TRAJECTORY_DIM} features")
        if any(action not in range(len(MOTION_INTENTS)) for action in self.future_actions):
            raise ValueError("future action is outside the motion-intent vocabulary")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionSegment:
    start: float
    end: float
    action: str
    prompt: str


@dataclass(frozen=True)
class TrajectoryPlan:
    plan_id: str
    horizon: float
    trajectory: list[dict[str, Any]]
    segments: list[MotionSegment]
    score: float
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

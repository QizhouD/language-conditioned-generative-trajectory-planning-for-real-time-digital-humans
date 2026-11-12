"""Classical path followers: the baselines a learned root-motion layer must beat.

Both consume the same local-frame polyline as the model and emit the same
(dx, dz, dyaw) steps, so every metric and every constraint applies unchanged.
The spring follower doubles as the planner's fallback when no sample is valid.
"""

from __future__ import annotations

import math

import numpy as np

from gtp.rootmotion.paths import arclength, point_at, project

LN2 = math.log(2.0)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _last_direction(polyline: np.ndarray) -> np.ndarray:
    for index in range(len(polyline) - 1, 0, -1):
        segment = polyline[index] - polyline[index - 1]
        length = float(np.hypot(*segment))
        if length > 1e-6:
            return segment / length
    return np.array([0.0, 1.0])


def _target_point(polyline: np.ndarray, cumulative: np.ndarray, distance: float,
                  ends_at_goal: bool) -> np.ndarray:
    total = float(cumulative[-1])
    if distance <= total or ends_at_goal:
        return point_at(polyline, cumulative, distance)
    # The path continues beyond what was sent: keep going straight.
    return polyline[-1] + _last_direction(polyline) * (distance - total)


def spring_follow(
    polyline: np.ndarray, ends_at_goal: bool, velocity: np.ndarray, target_speed: float,
    *, steps: int = 30, control_hz: int = 10, substeps: int = 3, halflife: float = 0.27,
    facing_halflife: float = 0.2, deceleration: float = 3.0, lookahead_seconds: float = 0.5,
    min_lookahead: float = 0.4, arrival_seconds: float = 0.6,
) -> np.ndarray:
    """Pure pursuit steering smoothed by a critically damped spring on velocity.

    This is the simulation-object model used by motion-matching controllers: the
    desired velocity points at a look-ahead point on the path and the simulated
    velocity chases it through a spring, which rounds corners and eases starts
    and stops. Returns [steps, 3] rows of (dx, dz, dyaw).
    """
    polyline = np.asarray(polyline, dtype=np.float64)
    if not ends_at_goal:
        # The path continues beyond what was sent: extend it straight ahead so the
        # look-ahead point never stalls at the last vertex.
        polyline = np.concatenate([polyline, polyline[-1:] + _last_direction(polyline) * 50.0])
    cumulative = arclength(polyline)
    total = float(cumulative[-1])
    dt = 1.0 / (control_hz * substeps)
    damping = (4.0 * LN2) / max(halflife, 1e-5) / 2.0
    decay = math.exp(-damping * dt)
    facing_blend = 1.0 - math.exp(-LN2 * dt / max(facing_halflife, 1e-5))

    position = np.zeros(2)
    velocity = np.asarray(velocity, dtype=np.float64).copy()
    acceleration = np.zeros(2)
    yaw, progress = 0.0, 0.0
    output = np.zeros((steps, 3))
    previous_position, previous_yaw = position.copy(), yaw
    for step in range(steps):
        for _ in range(substeps):
            speed = float(np.hypot(*velocity))
            progress = project(polyline, cumulative, position, progress, window=2.0 + speed)
            lookahead = max(min_lookahead, lookahead_seconds * speed)
            target = _target_point(polyline, cumulative, progress + lookahead, ends_at_goal)
            offset = target - position
            distance = float(np.hypot(*offset))
            direction = offset / distance if distance > 1e-6 else np.zeros(2)
            desired = target_speed
            if ends_at_goal:
                remaining = max(0.0, total - progress)
                if remaining <= lookahead:
                    # Arrival: the target is the goal itself. Slowing in proportion to
                    # the straight-line distance also pulls back overshoot from spring lag.
                    desired = min(desired, distance / arrival_seconds)
                else:
                    desired = min(desired, math.sqrt(2.0 * deceleration * remaining))
            goal_velocity = direction * desired
            j0 = velocity - goal_velocity
            j1 = acceleration + j0 * damping
            position = (
                position
                + decay * ((-j1) / (damping * damping) + (-j0 - j1 * dt) / damping)
                + j1 / (damping * damping) + j0 / damping + goal_velocity * dt
            )
            velocity = decay * (j0 + j1 * dt) + goal_velocity
            acceleration = decay * (acceleration - j1 * damping * dt)
            if np.hypot(*velocity) > 0.3:
                yaw += _wrap(math.atan2(velocity[0], velocity[1]) - yaw) * facing_blend
        output[step, :2] = position - previous_position
        output[step, 2] = _wrap(yaw - previous_yaw)
        previous_position, previous_yaw = position.copy(), yaw
    return output


def constant_speed_follow(
    polyline: np.ndarray, ends_at_goal: bool, speed: float, target_speed: float,
    *, steps: int = 30, control_hz: int = 10, acceleration: float = 8.0,
    angular_speed: float = math.radians(120.0),
) -> np.ndarray:
    """Stay exactly on the polyline with an acceleration-limited speed.

    Uses Unity NavMeshAgent's default acceleration (8 m/s^2) and angular speed
    (120 deg/s). It approximates agent steering rather than reproducing it: the
    real agent is evaluated inside Unity.
    """
    polyline = np.asarray(polyline, dtype=np.float64)
    cumulative = arclength(polyline)
    total = float(cumulative[-1])
    dt = 1.0 / control_hz
    travelled, yaw = 0.0, 0.0
    position = np.zeros(2)
    output = np.zeros((steps, 3))
    for step in range(steps):
        speed = min(target_speed, speed + acceleration * dt)
        if ends_at_goal:
            remaining = max(0.0, total - travelled)
            speed = min(speed, math.sqrt(2.0 * acceleration * remaining), remaining / dt)
        travelled += speed * dt
        target = _target_point(polyline, cumulative, travelled, ends_at_goal)
        move = target - position
        if np.hypot(*move) > 1e-6:
            turn = _wrap(math.atan2(move[0], move[1]) - yaw)
            turn = max(-angular_speed * dt, min(angular_speed * dt, turn))
        else:
            turn = 0.0
        output[step] = (move[0], move[1], turn)
        position, yaw = target, yaw + turn
    return output

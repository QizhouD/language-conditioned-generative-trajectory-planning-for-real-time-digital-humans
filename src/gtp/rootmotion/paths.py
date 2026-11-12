"""Polyline helpers shared by data preparation, constraints and baselines."""

from __future__ import annotations

import numpy as np
import torch


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification. Returns indices of the kept vertices."""
    count = len(points)
    if count <= 2:
        return np.arange(count)
    keep = np.zeros(count, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, count - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        segment = points[last] - points[first]
        inner = points[first + 1 : last] - points[first]
        length = float(np.hypot(*segment))
        if length < 1e-9:
            distances = np.hypot(inner[:, 0], inner[:, 1])
        else:
            # Distance to the segment, not the infinite line, so loops are preserved.
            ratio = np.clip(inner @ segment / (length * length), 0.0, 1.0)
            closest = ratio[:, None] * segment[None, :]
            distances = np.hypot(*(inner - closest).T)
        worst = int(np.argmax(distances))
        if distances[worst] > epsilon:
            pivot = first + 1 + worst
            keep[pivot] = True
            stack.append((first, pivot))
            stack.append((pivot, last))
    return np.flatnonzero(keep)


def arclength(points: np.ndarray) -> np.ndarray:
    """Cumulative length along a polyline, starting at zero."""
    if len(points) < 2:
        return np.zeros(len(points))
    steps = np.hypot(*np.diff(points, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(steps)])


def point_at(points: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    """Position at a given arclength, clamped to the polyline's ends."""
    if len(points) == 1 or distance <= 0:
        return points[0].copy()
    if distance >= cumulative[-1]:
        return points[-1].copy()
    index = int(np.searchsorted(cumulative, distance, side="right"))
    span = cumulative[index] - cumulative[index - 1]
    ratio = 0.0 if span < 1e-9 else (distance - cumulative[index - 1]) / span
    return points[index - 1] + ratio * (points[index] - points[index - 1])


def project(points: np.ndarray, cumulative: np.ndarray, query: np.ndarray,
            start: float = 0.0, window: float | None = None) -> float:
    """Arclength of the closest point on the polyline, searched from ``start`` onwards."""
    if len(points) == 1:
        return 0.0
    best_distance, best_arclength = np.inf, start
    limit = cumulative[-1] if window is None else min(cumulative[-1], start + window)
    for index in range(len(points) - 1):
        if cumulative[index + 1] < start or cumulative[index] > limit:
            continue
        segment = points[index + 1] - points[index]
        length_sq = float(segment @ segment)
        ratio = 0.0 if length_sq < 1e-12 else float(np.clip((query - points[index]) @ segment / length_sq, 0.0, 1.0))
        along = cumulative[index] + ratio * (cumulative[index + 1] - cumulative[index])
        along = float(np.clip(along, start, limit))
        candidate = point_at(points, cumulative, along)
        distance = float(np.hypot(*(query - candidate)))
        if distance < best_distance - 1e-9:
            best_distance, best_arclength = distance, along
    return best_arclength


def distance_to_polyline(
    points: torch.Tensor, polyline: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Distance from each point to a padded polyline.

    points: [..., N, 2]; polyline: [..., M, 2]; valid: [..., M] marks real vertices.
    Vertices must be packed at the front. Returns [..., N].
    """
    start, end = polyline[..., :-1, :], polyline[..., 1:, :]
    segment_valid = valid[..., :-1] & valid[..., 1:]
    segment = end - start
    length_sq = segment.square().sum(-1).clamp_min(1e-12)
    offset = points.unsqueeze(-2) - start.unsqueeze(-3)  # [..., N, M-1, 2]
    ratio = ((offset * segment.unsqueeze(-3)).sum(-1) / length_sq.unsqueeze(-2)).clamp(0.0, 1.0)
    closest = start.unsqueeze(-3) + ratio.unsqueeze(-1) * segment.unsqueeze(-3)
    segment_distance = torch.linalg.vector_norm(points.unsqueeze(-2) - closest, dim=-1)
    segment_distance = segment_distance.masked_fill(~segment_valid.unsqueeze(-2), float("inf"))
    # A single-vertex path has no segments; fall back to distance to its vertices.
    vertex_distance = torch.linalg.vector_norm(
        points.unsqueeze(-2) - polyline.unsqueeze(-3), dim=-1
    ).masked_fill(~valid.unsqueeze(-2), float("inf"))
    return torch.minimum(segment_distance.amin(-1), vertex_distance.amin(-1))

"""Path-conditioned root-motion layer.

Pathfinding stays with the engine (NavMesh). This package learns *how* a character
moves along an engine-provided path: speed profile, corner rounding, facing and
stopping, from real motion-capture root trajectories.
"""

GAITS = ("walk", "run", "sprint")
CONTROL_HZ = 10
HISTORY_STEPS = 20
HORIZON_STEPS = 30
MAX_PATH_VERTICES = 16

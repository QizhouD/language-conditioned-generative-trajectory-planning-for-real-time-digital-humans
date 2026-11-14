"""Shared helpers for phase g (tuned spring): the 73 rest-to-rest walks, capture reference
entries, one spring rollout, and the per-walk objective terms. Imported by g_search.py,
g_eval.py and g_checks.py; not run directly. Needs data/lafan1_root.npz.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from f_eval import CONFIG, gated_arrival, record  # noqa: E402

from gtp.rootmotion import CONTROL_HZ  # noqa: E402
from gtp.rootmotion.closed_loop import baseline_plan_function, capture_segments, simulate, speed_by_progress  # noqa: E402
from gtp.rootmotion.extract import parse_clip_name  # noqa: E402
from gtp.rootmotion.paths import arclength  # noqa: E402

GAIT_SPEED = (0.95, 2.0, 2.5)  # unused: every request carries the walk's target_speed
_WALKS = None


def walks():
    """All rest-to-rest walks, sorted by subject then clip: list of (subject, segment)."""
    global _WALKS
    if _WALKS is None:
        tracks = dict(np.load(CONFIG["data_path"]))
        out = []
        for clip in sorted(tracks):
            parsed = parse_clip_name(clip)
            if parsed and parsed[0] == "walk":
                out += [(parsed[1], seg) for seg in capture_segments(tracks[clip], clip)]
        _WALKS = out
    return _WALKS


def capture_entry(seg):
    capture_arrival = gated_arrival(seg.poses, seg.route.points)
    cap_speed = np.linalg.norm(np.diff(seg.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
    end = min(len(seg.poses) - 1, capture_arrival + CONTROL_HZ)
    d = np.diff(seg.poses[: end + 1], axis=0)
    sp = np.linalg.norm(d[:, :2], axis=1) * CONTROL_HZ
    return {"arrival_time": capture_arrival / CONTROL_HZ, "mean": float(cap_speed[:capture_arrival].mean()),
            "length": float(arclength(seg.poses[:, :2])[-1]), "label": seg.target_speed,
            "profile": speed_by_progress(seg.poses[: capture_arrival + 1, :2], seg.route.points).tolist(),
            "kin": {"speed": sp.tolist(), "acceleration": (np.diff(sp) * CONTROL_HZ).tolist(),
                    "turn_rate": (np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2])) * CONTROL_HZ).tolist()}}


def run_spring(seg, params, keep_kin=True):
    fn = baseline_plan_function("spring", GAIT_SPEED, params)
    duration = (seg.end - seg.start) / CONTROL_HZ
    rollout = simulate(seg.route, fn, time_limit=3.0 * duration, initial_history=seg.history,
                       request_extras={"target_speed": seg.target_speed})
    rec, rollout = record(rollout, seg.target_speed)
    rec.pop("phases", None)
    if not keep_kin:
        rec.pop("kin", None)
    stop = int(round(rec["arrival_time"] * CONTROL_HZ)) if rec["arrival_time"] is not None else len(rollout.poses) - 1
    rec["profile"] = speed_by_progress(rollout.poses[: stop + 1, :2], seg.route.points).tolist()
    return rec


def objective_terms(rec, cap):
    """(speed-progress RMSE, |arrival / capture - 1|) for one walk; a run that never arrives counts as 3x."""
    a, b = np.asarray(rec["profile"], float), np.asarray(cap["profile"], float)
    ok = np.isfinite(a) & np.isfinite(b)
    rmse = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else 1.0
    t = rec["arrival_time"]
    ratio = (t if t is not None else 3.0 * cap["arrival_time"]) / cap["arrival_time"]
    return rmse, abs(ratio - 1.0)

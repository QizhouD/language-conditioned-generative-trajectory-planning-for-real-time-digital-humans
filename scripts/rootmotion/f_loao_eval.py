"""Leave-one-actor-out closed loop on the held-out actor's rest-to-rest walks.

  f_loao_eval.py NAME HOLD     NAME in base, C2, spring; HOLD in 1..5
Writes results/phase_f/loao/NAME_hold{HOLD}.json (same "walks" layout as f_eval.py).

Flow: consistent selection, K = 16, seeds 7, 11, 13 of the fold's checkpoints. Spring: the
untuned spring told each walk's label. Feeds the flow and untuned-spring rows of T2 to T5.
Checkpoints: runs/loao/NAME_hold{HOLD}_s{7,11,13}/best.pt (f_train.py loao NAME); the spring
reads runs/loao/base_hold{HOLD}_s7/best.pt for its statistics only.
Run from the repository root with the project venv:
  for h in 1 2 3 4 5; do python scripts/rootmotion/f_loao_eval.py base $h; python scripts/rootmotion/f_loao_eval.py spring $h; done
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from f_eval import CONFIG, gated_arrival, record  # noqa: E402

from gtp.rootmotion import CONTROL_HZ  # noqa: E402
from gtp.rootmotion.closed_loop import (  # noqa: E402
    baseline_plan_function, capture_segments, model_plan_function, route_seed, simulate, speed_by_progress,
)
from gtp.rootmotion.extract import parse_clip_name  # noqa: E402
from gtp.rootmotion.paths import arclength  # noqa: E402
from gtp.rootmotion.planner import PlannerConfig, RootMotionPlanner  # noqa: E402

SEEDS = [7, 11, 13]
OUT = Path("results/phase_f/loao")


def main(name, hold):
    tracks = dict(np.load(CONFIG["data_path"]))
    walks = []
    for clip in sorted(tracks):
        parsed = parse_clip_name(clip)
        if parsed and parsed[0] == "walk" and parsed[1] == hold:
            walks += capture_segments(tracks[clip], clip)
    if name == "spring":
        # The spring's per-gait speed is unused when target_speed is given; take it from any checkpoint.
        stats = RootMotionPlanner.from_checkpoint(f"runs/loao/base_hold{hold}_s7/best.pt").stats
        fns = lambda route: [baseline_plan_function("spring", stats.gait_speed)]
    else:
        planners = [RootMotionPlanner.from_checkpoint(f"runs/loao/{name}_hold{hold}_s{s}/best.pt",
                                                      PlannerConfig(candidates=16)) for s in SEEDS]
        fns = lambda route: [model_plan_function(p, torch.Generator().manual_seed(route_seed(route))) for p in planners]
    result = {"name": name, "hold": hold, "walks": {}}
    for seg in walks:
        capture_arrival = gated_arrival(seg.poses, seg.route.points)
        cap_speed = np.linalg.norm(np.diff(seg.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
        end = min(len(seg.poses) - 1, capture_arrival + CONTROL_HZ)
        d = np.diff(seg.poses[: end + 1], axis=0)
        sp = np.linalg.norm(d[:, :2], axis=1) * CONTROL_HZ
        entry = {"capture": {"arrival_time": capture_arrival / CONTROL_HZ, "mean": float(cap_speed[:capture_arrival].mean()),
                             "length": float(arclength(seg.poses[:, :2])[-1]), "label": seg.target_speed,
                             "profile": speed_by_progress(seg.poses[: capture_arrival + 1, :2], seg.route.points).tolist(),
                             "kin": {"speed": sp.tolist(), "acceleration": (np.diff(sp) * CONTROL_HZ).tolist(),
                                     "turn_rate": (np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2])) * CONTROL_HZ).tolist()}},
                 "runs": []}
        duration = (seg.end - seg.start) / CONTROL_HZ
        for fn in fns(seg.route):
            rollout = simulate(seg.route, fn, time_limit=3.0 * duration, initial_history=seg.history,
                               request_extras={"target_speed": seg.target_speed})
            rec, rollout = record(rollout, seg.target_speed)
            rec.pop("phases", None)
            stop = int(round(rec["arrival_time"] * CONTROL_HZ)) if rec["arrival_time"] is not None else len(rollout.poses) - 1
            rec["profile"] = speed_by_progress(rollout.poses[: stop + 1, :2], seg.route.points).tolist()
            entry["runs"].append(rec)
        result["walks"][seg.route.name] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_hold{hold}.json").write_text(json.dumps(result), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] done {name} hold{hold} ({len(walks)} walks)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))

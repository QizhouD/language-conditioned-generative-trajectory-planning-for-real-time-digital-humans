"""Pick each fold's spring (stage 1 + that fold's own perturbations only), run it on the held-out
actor's walks and on the synthetic routes; also rerun the default through the same driver as a check.

Produces: results/phase_g/selection.json (T1), springT_hold{1..5}.json (tuned-spring rows of
T2 to T5), springD_hold{1..5}.json (default check), synthetic.json (tuned-spring rows of T6).
Checkpoints: none. Needs results/phase_g/search.json (g_search.py).
Run from the repository root with the project venv:
  python scripts/rootmotion/g_eval.py            # or --select-only to write selection.json only
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g_common import CONTROL_HZ, GAIT_SPEED, capture_entry, run_spring, walks  # noqa: E402
from g_search import RANGES  # noqa: E402

from gtp.rootmotion.closed_loop import SHAPES, Route, baseline_plan_function, rollout_metrics, simulate, unity_route  # noqa: E402

OUT = Path("results/phase_g")
STAGE1 = 401


def main():
    data = json.loads((OUT / "search.json").read_text())
    configs, scores, subjects = data["configs"], data["scores"], data["subjects"]
    selection = {}
    for hold in range(1, 6):
        pool = list(range(STAGE1)) + list(range(STAGE1 + 80 * (hold - 1), STAGE1 + 80 * hold))
        train = [i for i, s in enumerate(subjects) if s != hold]
        test = [i for i, s in enumerate(subjects) if s == hold]

        def objective(c, rows_index):
            r = np.asarray(scores[c])[rows_index]
            return float(r[:, 0].mean() + r[:, 1].mean())

        best = min(pool, key=lambda c: objective(c, train))
        params = configs[best]
        edges = [k for k, (lo, hi, _) in RANGES.items() if min(params[k] - lo, hi - params[k]) <= 0.02 * (hi - lo)]
        selection[hold] = {"index": best, "params": params, "at_edge": edges,
                           "train_objective": objective(best, train), "train_objective_default": objective(0, train),
                           "test_objective": objective(best, test), "test_objective_default": objective(0, test)}
        print(hold, json.dumps(selection[hold]), flush=True)
    (OUT / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    if "--select-only" in sys.argv:
        return
    w = walks()
    for hold in range(1, 6):
        for tag, params in (("springT", selection[hold]["params"]), ("springD", None)):
            result = {"name": tag, "hold": hold, "params": params, "walks": {}}
            for subject, seg in w:
                if subject != hold:
                    continue
                result["walks"][seg.route.name] = {"capture": capture_entry(seg), "runs": [run_spring(seg, params)]}
            (OUT / f"{tag}_hold{hold}.json").write_text(json.dumps(result), encoding="utf-8")
        print("evaluated hold", hold, flush=True)
    synthetic = {}
    for route in (Route("straight_12/aligned/walk", SHAPES["straight_12"], 0.0, "walk"), unity_route()):
        synthetic[route.name] = {}
        for tag, params in [("default", None)] + [(f"fold{h}", selection[h]["params"]) for h in range(1, 6)]:
            r = simulate(route, baseline_plan_function("spring", GAIT_SPEED, params), gait_speed=0.9,
                         request_extras={"target_speed": 0.9})
            m = rollout_metrics(r)
            speed = np.linalg.norm(np.diff(r.poses[:, :2], axis=0), axis=1) * CONTROL_HZ
            end = int(round(m["arrival_time"] * CONTROL_HZ)) if m["arrival_time"] is not None else len(speed)
            s = speed[:end]
            hit = lambda v: (float(np.flatnonzero(s >= v)[0] / CONTROL_HZ) if (s >= v).any() else None)
            synthetic[route.name][tag] = {
                "to_045": hit(0.45), "to_081": hit(0.81), "mean_speed": m["follow_mean_speed"],
                "mean_over_label": m["follow_mean_speed"] / 0.9, "arrival_time": m["arrival_time"],
                "cruise_p90": m["cruise_p90"], "max_speed": float(s.max()), "max_deviation": m["max_deviation"],
                "target": 0.9 * (params or {}).get("speed_scale", 1.0)}
            print(route.name, tag, json.dumps(synthetic[route.name][tag]), flush=True)
    (OUT / "synthetic.json").write_text(json.dumps(synthetic, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

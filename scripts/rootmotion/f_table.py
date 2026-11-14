"""Tables from results/phase_f/eval_*.json (standard split).

  f_table.py NAME ...   the first name is the baseline for the open-loop regression check

Produces: captured-walk tables (short / long), synthetic-route tables, the open-loop table
(T7 flow row) and the success-criteria table (T9), on stdout; writes results/phase_f/summary.json.
Also imported by f_loao_*.py, g_checks.py and g_stats.py for SHORT (35 m), w1 and walk_block.
Checkpoints: none.
Run from the repository root with the project venv:
  python scripts/rootmotion/f_table.py base A1.5 A2 A3 B1 B2 C5 C2 C2+A2 spring
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

from gtp.rootmotion import CONTROL_HZ

OUT = Path("results/phase_f")
SHORT = 35.0
PM = " ± "


def w1(a, b):
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    q = np.linspace(0, 1, 200, endpoint=False) + 0.0025
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def ms(v, d=2):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return f"{st.fmean(v):.{d}f}" + (PM + f"{st.stdev(v):.{d}f}" if len(v) > 1 else "")


def seedwise(runs_per_walk, fn):
    """runs_per_walk: list over walks of list over seeds -> per-seed mean of fn over walks."""
    seeds = len(runs_per_walk[0])
    return [st.fmean([fn(w, s) for w in range(len(runs_per_walk))]) for s in range(seeds)]


def walk_block(data, keep):
    walks = [w for w in data["walks"].values() if keep(w["capture"]["length"])]
    if not walks:
        return None
    runs = [w["runs"] for w in walks]
    cap = [w["capture"] for w in walks]

    def arrival_ratio(w, s):
        t = runs[w][s]["arrival_time"]
        return (t if t is not None else np.nan) / cap[w]["arrival_time"]

    def mean_ratio(w, s):
        return runs[w][s]["follow_mean_speed"] / cap[w]["mean"]

    def rmse(w, s):
        a, b = np.asarray(runs[w][s]["profile"], float), np.asarray(cap[w]["profile"], float)
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))

    seeds = len(runs[0])
    w1s = {k: [] for k in ("speed", "acceleration", "turn_rate")}
    for s in range(seeds):
        for k in w1s:
            model = np.concatenate([runs[w][s]["kin"][k] for w in range(len(runs))])
            human = np.concatenate([cap[w]["kin"][k] for w in range(len(runs))])
            w1s[k].append(w1(model, human))
    get = lambda key: seedwise(runs, lambda w, s: runs[w][s][key] if runs[w][s][key] is not None else np.nan)
    arrived = seedwise(runs, lambda w, s: float(runs[w][s]["arrival_time"] is not None))
    return {
        "n": len(walks), "arrival/human": seedwise(runs, arrival_ratio), "mean/human": seedwise(runs, mean_ratio),
        "profile RMSE": seedwise(runs, rmse), "W1 speed": w1s["speed"], "W1 accel": w1s["acceleration"],
        "W1 turn": w1s["turn_rate"], "seam accel": get("seam_accel_mean"), "other accel": get("other_accel_mean"),
        "final dist": get("final_distance"), "max dev": get("max_deviation"), "fallback": get("fallback_rate"),
        "arrived": arrived,
    }


def synth_block(runs):
    keys = ("to_045", "to_081", "stall", "follow_mean_speed", "arrival_time", "max_deviation", "cruise_p90")
    return {k: [r[k] for r in runs] for k in keys}


def main(names):
    data = {n: json.loads((OUT / f"eval_{n}.json").read_text()) for n in names}
    summary = {}
    for label, keep in (("short", lambda L: L <= SHORT), ("long", lambda L: L > SHORT)):
        blocks = {n: walk_block(d, keep) for n, d in data.items()}
        any_block = next(b for b in blocks.values() if b)
        print(f"\n### Captured walks, subject 5, {label} (n = {any_block['n']})\n")
        cols = [k for k in any_block if k != "n"]
        print("| variant | " + " | ".join(cols) + " |")
        print("|---|" + "---|" * len(cols))
        for n, b in blocks.items():
            print(f"| {n} | " + " | ".join(ms(b[c], 3 if c.startswith("W1") or c == "fallback" else 2) for c in cols) + " |")
            summary.setdefault(n, {})[label] = {c: st.fmean([x for x in b[c] if np.isfinite(x)]) for c in cols}
    for route in next(iter(data.values()))["synthetic"]:
        print(f"\n### {route} at 0.9 m/s\n")
        cols = ("to_045", "to_081", "stall", "follow_mean_speed", "arrival_time", "max_deviation", "cruise_p90")
        print("| variant | " + " | ".join(cols) + " |")
        print("|---|" + "---|" * len(cols))
        for n, d in data.items():
            b = synth_block(d["synthetic"][route])
            print(f"| {n} | " + " | ".join(ms(b[c]) for c in cols) + " |")
            summary[n][route] = {c: (st.fmean([x for x in b[c] if x is not None]) if any(x is not None for x in b[c]) else None)
                                 for c in cols}
            summary[n][route]["to_081_all"] = b["to_081"]
    print("\n### Open loop (test split, walk+run, 5 seeds)\n")
    cols = ("ade", "fde", "min_ade", "stop_error", "fallback_rate", "w1_speed", "w1_acceleration", "w1_turn_rate",
            "corridor_violation")
    base = data[names[0]].get("open")
    print("| variant | plan ms | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for n, d in data.items():
        if "open" not in d:
            continue
        cells = []
        for c in cols:
            vals = [r.get(c) for r in d["open"]]
            if any(v is None for v in vals):
                cells.append("?")
                continue
            cells.append(ms(vals, 3))
        print(f"| {n} | {d['plan_ms']:.0f} | " + " | ".join(cells) + " |")
        if base:
            ok = {}
            for c in ("ade", "fallback_rate"):
                b = [r[c] for r in base]
                ok[c] = st.fmean([r[c] for r in d["open"]]) <= st.fmean(b) + 2 * st.stdev(b)
            summary[n]["open_ok"] = ok
            summary[n]["open"] = {c: st.fmean([r[c] for r in d["open"]]) for c in ("ade", "fallback_rate")}
    print("\n### Success criteria\n")
    print("| variant | 12 m to 0.81 | Unity to 0.81 | short arrival/human | open ADE/fallback | verdict |")
    print("|---|---|---|---|---|---|")
    routes = list(next(iter(data.values()))["synthetic"])
    for n in names:
        s = summary[n]
        t = [s[r]["to_081_all"] for r in routes]
        mean_t = [s[r]["to_081"] if all(x is not None for x in ts) else None for r, ts in zip(routes, t)]
        in_band = [m is not None and 1.35 <= m <= 4.8 for m in mean_t]
        hits = [sum(x is not None and 1.35 <= x <= 4.8 for x in ts) for ts in t]
        ar = s["short"]["arrival/human"]
        ar_ok = 0.9 <= ar <= 1.1
        op = s.get("open_ok", {"ade": True, "fallback_rate": True})
        verdict = all(in_band) and ar_ok and all(op.values())
        fmt = lambda m, ok, h: (f"{m:.1f} s" if m is not None else "not always") + (" ✓" if ok else " ✗") + f" ({h}/5 seeds in band)"
        print(f"| {n} | {fmt(mean_t[0], in_band[0], hits[0])} | {fmt(mean_t[1], in_band[1], hits[1])} | {ar:.2f}"
              f"{' ✓' if ar_ok else ' ✗'} | {'✓' if all(op.values()) else '✗ ' + str(op)} | "
              f"{'达到' if verdict else '未达到'} |")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

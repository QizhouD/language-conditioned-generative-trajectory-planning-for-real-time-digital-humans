"""Phase-g tables and paired bootstrap (flow - tuned spring), short / long separate.

Decision rule (fixed before looking): a difference is significant when the walk-level 95% bootstrap CI
excludes 0; the actor-level cluster bootstrap, the per-actor differences and their sign count are
reported beside it. Ratio metrics are judged on |x - 1| (closeness to the person), per run.
Produces: results/phase_g/tables.md and paired.json -> T2, T3, T4, T5 (seeded, deterministic).
Checkpoints: none. Reads results/phase_f/loao/{base,spring}_hold*.json (f_loao_eval.py) and
  results/phase_g/springT_hold*.json (g_eval.py). Takes a few minutes (pooled W1 per resample).
Run from the repository root with the project venv:
  python scripts/rootmotion/g_stats.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from f_table import SHORT, w1, walk_block  # noqa: E402

FLOW, ORIG = Path("results/phase_f/loao"), Path("results/phase_f/loao")
TUNED = Path("results/phase_g")
OUT = Path("results/phase_g")
METHODS = {"flow": (FLOW, "base"), "spring": (ORIG, "spring"), "tuned": (TUNED, "springT")}
COLS = ["arrival/human", "mean/human", "profile RMSE", "W1 speed", "W1 accel", "W1 turn", "seam accel", "other accel",
        "final dist", "max dev", "fallback", "arrived"]
SCALARS = {  # name: (record key, kind)  kind: ratio_arrival / ratio_mean / plain
    "arrival/human |x-1|": ("arrival_dev", "plain"), "mean/human |x-1|": ("mean_dev", "plain"),
    "arrival/human (signed)": ("arrival", "signed"), "mean/human (signed)": ("mean", "signed"),
    "profile RMSE": ("rmse", "plain"), "final dist": ("final_distance", "plain"),
    "max dev": ("max_deviation", "plain"), "seam accel": ("seam_accel_mean", "plain"),
    "other accel": ("other_accel_mean", "plain"),
}
KIN = ("speed", "acceleration", "turn_rate")
B_SCALAR, B_W1 = 10000, 2000


def load(method, hold):
    root, name = METHODS[method]
    return json.loads((root / f"{name}_hold{hold}.json").read_text())


def per_walk(entry):
    cap, runs = entry["capture"], entry["runs"]

    def rmse(r):
        a, b = np.asarray(r["profile"], float), np.asarray(cap["profile"], float)
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))

    vals = {"arrival": st.fmean(r["arrival_time"] / cap["arrival_time"] for r in runs),
            "mean": st.fmean(r["follow_mean_speed"] / cap["mean"] for r in runs),
            "rmse": st.fmean(rmse(r) for r in runs),
            # Distance from the person per run, then averaged: seed scatter must not cancel.
            "arrival_dev": st.fmean(abs(r["arrival_time"] / cap["arrival_time"] - 1) for r in runs),
            "mean_dev": st.fmean(abs(r["follow_mean_speed"] / cap["mean"] - 1) for r in runs)}
    for k in ("final_distance", "max_deviation", "seam_accel_mean", "other_accel_mean"):
        vals[k] = st.fmean(r[k] for r in runs)
    return vals


def gather(keep):
    rows = []
    for hold in range(1, 6):
        data = {m: load(m, hold)["walks"] for m in METHODS}
        for name, entry in data["flow"].items():
            if not keep(entry["capture"]["length"]):
                continue
            rows.append({"actor": hold, "name": name, "capture_kin": entry["capture"]["kin"],
                         **{m: per_walk(data[m][name]) for m in METHODS},
                         **{f"{m}_kin": [r["kin"] for r in data[m][name]["runs"]] for m in METHODS}})
    return rows


def diff_vector(rows, key, kind):
    f = np.array([r["flow"][key] for r in rows])
    t = np.array([r["tuned"][key] for r in rows])
    if kind == "ratio":
        return np.abs(f - 1) - np.abs(t - 1)
    return f - t


def pooled_w1(rows, idx, method, k):
    human = np.concatenate([rows[i]["capture_kin"][k] for i in idx])
    seeds = len(rows[idx[0]][f"{method}_kin"])
    return st.fmean(w1(np.concatenate([rows[i][f"{method}_kin"][s][k] for i in idx]), human) for s in range(seeds))


def ci(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def analyse(rows, rng):
    n = len(rows)
    actors = sorted({r["actor"] for r in rows})
    by_actor = {a: [i for i, r in enumerate(rows) if r["actor"] == a] for a in actors}
    walk_idx = rng.integers(0, n, size=(B_SCALAR, n))
    actor_draws = rng.integers(0, len(actors), size=(B_SCALAR, len(actors)))
    out = {}
    for label, (key, kind) in SCALARS.items():
        d = diff_vector(rows, key, kind)
        boot_w = d[walk_idx].mean(axis=1)
        sums = np.array([d[by_actor[a]].sum() for a in actors])
        counts = np.array([len(by_actor[a]) for a in actors])
        boot_a = sums[actor_draws].sum(axis=1) / counts[actor_draws].sum(axis=1)
        per_actor = {a: float(d[by_actor[a]].mean()) for a in actors}
        out[label] = {"diff": float(d.mean()), "walk_ci": ci(boot_w), "actor_ci": ci(boot_a), "per_actor": per_actor,
                      "flow_better_actors": int(sum(v < 0 for v in per_actor.values())), "n_actors": len(actors)}
    for k, label in zip(KIN, ("W1 speed", "W1 accel", "W1 turn")):
        stat = lambda idx: pooled_w1(rows, idx, "flow", k) - pooled_w1(rows, idx, "tuned", k)
        full = list(range(n))
        boot_w = [stat(list(rng.integers(0, n, size=n))) for _ in range(B_W1)]
        boot_a = []
        for _ in range(B_W1):
            pick = rng.integers(0, len(actors), size=len(actors))
            boot_a.append(stat([i for a in pick for i in by_actor[actors[a]]]))
        per_actor = {a: stat(by_actor[a]) for a in actors}
        out[label] = {"diff": stat(full), "walk_ci": ci(boot_w), "actor_ci": ci(boot_a), "per_actor": per_actor,
                      "flow_better_actors": int(sum(v < 0 for v in per_actor.values())), "n_actors": len(actors)}
    return out


DESCRIPTIVE = ("arrival/human (signed)", "mean/human (signed)", "other accel")


def verdict(entry, metric=""):
    lo, hi = entry["walk_ci"]
    if metric in DESCRIPTIVE:  # no better/worse direction: report the sign only
        return "descriptive: " + ("flow lower" if hi < 0 else "flow higher" if lo > 0 else "no difference")
    if hi < 0:
        v = "flow better"
    elif lo > 0:
        v = "spring better"
    else:
        return "tie"
    alo, ahi = entry["actor_ci"]
    agree = (ahi < 0) if v == "flow better" else (alo > 0)
    return v + (" (actor level agrees)" if agree else " (not at actor level)")


def tables():
    lines = []
    for label, keep in (("short", lambda L: L <= SHORT), ("long", lambda L: L > SHORT)):
        lines.append(f"\n### {label} walks: mean ± std over the five folds\n")
        lines.append("| method | " + " | ".join(COLS) + " |")
        lines.append("|---|" + "---|" * len(COLS))
        per_actor_rows = []
        for m in METHODS:
            folds = []
            for hold in range(1, 6):
                b = walk_block(load(m, hold), keep)
                folds.append({c: float(np.nanmean(b[c])) for c in COLS} | {"n": b["n"]})
            cells = []
            for c in COLS:
                v = [f[c] for f in folds]
                d = 3 if c.startswith("W1") or c == "fallback" else 2
                cells.append(f"{st.fmean(v):.{d}f} ± {st.stdev(v):.{d}f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
            per_actor_rows.append((m, folds))
        lines.append(f"\n{label} walks per actor (arrival/human, mean/human, profile RMSE, W1 speed/accel/turn, max dev):\n")
        lines.append("| actor | n | " + " | ".join(m for m, _ in per_actor_rows) + " |")
        lines.append("|---|---|" + "---|" * len(per_actor_rows))
        for h in range(5):
            cells = []
            for m, folds in per_actor_rows:
                f = folds[h]
                cells.append(f"{f['arrival/human']:.2f}, {f['mean/human']:.2f}, {f['profile RMSE']:.2f}, "
                             f"{f['W1 speed']:.3f}/{f['W1 accel']:.3f}/{f['W1 turn']:.3f}, {f['max dev']:.2f}")
            lines.append(f"| {h + 1} | {per_actor_rows[0][1][h]['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    rng = np.random.default_rng(0)
    report = [tables()]
    result = {}
    for label, keep in (("short", lambda L: L <= SHORT), ("long", lambda L: L > SHORT)):
        rows = gather(keep)
        res = analyse(rows, rng)
        result[label] = res
        report.append(f"\n### Paired flow − tuned spring, {label} walks (n = {len(rows)}); negative = flow better\n")
        report.append("| metric | flow − tuned | walk 95% CI | actor 95% CI | per actor (1..5) | actors favouring flow | verdict |")
        report.append("|---|---|---|---|---|---|---|")
        for metric, e in res.items():
            pa = " / ".join(f"{e['per_actor'][a]:+.3f}" for a in sorted(e["per_actor"]))
            report.append(f"| {metric} | {e['diff']:+.3f} | [{e['walk_ci'][0]:+.3f}, {e['walk_ci'][1]:+.3f}] | "
                          f"[{e['actor_ci'][0]:+.3f}, {e['actor_ci'][1]:+.3f}] | {pa} | "
                          f"{e['flow_better_actors']}/{e['n_actors']} | {verdict(e, metric)} |")
            e["verdict"] = verdict(e, metric)
    text = "\n".join(report)
    (OUT / "tables.md").write_text(text, encoding="utf-8")
    (OUT / "paired.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

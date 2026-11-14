"""Train flow variants (phase f).

  f_train.py screen            B1, B2, C5, C2 on the standard split, 5 seeds -> runs/phase_f/NAME_sSEED/
  f_train.py variant NAME ...  named variants on the standard split, 5 seeds
  f_train.py loao NAME ...     leave-one-actor-out, 3 seeds per fold -> runs/loao/NAME_holdH_sSEED/

Produces the checkpoints behind T9 (variants) and, with `loao base`, the flow of T2 to T6
(runs/loao/base_hold{1..5}_s{7,11,13}/best.pt, last-epoch checkpoints).
Needs: data/lafan1_root.npz, configs/rootmotion_flow_speed_cpu.yaml.
Run from the repository root with the project venv:
  python scripts/rootmotion/f_train.py loao base
  python scripts/rootmotion/f_train.py screen
"""
import subprocess
import sys
import time
from pathlib import Path

import yaml

SEEDS = [7, 11, 13, 17, 19]
LOAO_SEEDS = [7, 11, 13]
SUBJECTS = [1, 2, 3, 4, 5]
VARIANTS = {
    "base": {},
    "B1": {"training": {"history_noise_position": 0.01, "history_noise_yaw": 0.02}},
    "B2": {"training": {"history_noise_position": 0.03, "history_noise_yaw": 0.05}},
    "C5": {"model": {"history_steps": 5}},
    "C2": {"model": {"history_steps": 2}},
}


def merged(base, override):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in override.items():
        out.setdefault(key, {}).update(value)
    return out


def train(name, overrides, seed, root, data_override=None):
    config = yaml.safe_load(Path("configs/rootmotion_flow_speed_cpu.yaml").read_text(encoding="utf-8"))
    config = merged(config, overrides)
    if data_override:
        config = merged(config, data_override)
    run = f"{name}_s{seed}"
    config.update(seed=seed, experiment=run, run_dir=(root / run).as_posix())
    (root / "configs").mkdir(parents=True, exist_ok=True)
    path = root / "configs" / f"{run}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if (root / run / "best.pt").exists():
        print(f"skip {run} (exists)", flush=True)
        return
    started = time.time()
    with open(root / f"{run}.train.log", "w", encoding="utf-8") as handle:
        code = subprocess.run([sys.executable, "-u", "-m", "gtp.rootmotion.train", "--config", str(path)],
                              stdout=handle, stderr=subprocess.STDOUT).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {root.name}/{run} exit={code} {time.time() - started:.0f}s", flush=True)
    if code:
        raise SystemExit(code)


def variant_overrides(name):
    """'C5+B2' style combinations merge their parts."""
    out = {}
    for part in name.split("+"):
        out = merged(out, VARIANTS[part])
    return out


if __name__ == "__main__":
    command, names = sys.argv[1], sys.argv[2:]
    if command == "screen":
        names = ["B1", "B2", "C5", "C2"]
    if command in ("screen", "variant"):
        for name in names:
            for seed in SEEDS:
                train(name, variant_overrides(name), seed, Path("runs/phase_f"))
    elif command == "loao":
        for name in names:
            for held in SUBJECTS:
                rest = [s for s in SUBJECTS if s != held]
                data = {"data": {"train_subjects": rest, "val_subjects": [held], "test_subjects": [held]},
                        "training": {"checkpoint": "last"}}
                for seed in LOAO_SEEDS:
                    train(f"{name}_hold{held}", variant_overrides(name), seed, Path("runs/loao"), data)

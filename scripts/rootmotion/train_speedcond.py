"""Train the speed-conditioned CPU-size models, standard split, five seeds each.

Produces: checkpoints runs/speedcond/rootmotion_{flow,regressor}_speed_cpu_s{7,11,13,17,19}/best.pt
  (and their configs and train logs). Every standard-split table in docs/ROOT_MOTION_RESULTS.md
  depends on these checkpoints (T7, T8, T9 and the flow rows of T6).
Needs: data/lafan1_root.npz (see docs/ROOT_MOTION_LAYER.md, Data). Trains on the GPU if one is
  available; about a minute per run.
Run from the repository root with the project venv:
  python scripts/rootmotion/train_speedcond.py
"""
import subprocess
import sys
import time
from pathlib import Path

import yaml

SEEDS = [7, 11, 13, 17, 19]
ROOT = Path("runs/speedcond")
(ROOT / "configs").mkdir(parents=True, exist_ok=True)
for model in ("flow", "regressor"):
    for seed in SEEDS:
        config = yaml.safe_load(Path(f"configs/rootmotion_{model}_speed_cpu.yaml").read_text(encoding="utf-8"))
        name = f"rootmotion_{model}_speed_cpu_s{seed}"
        config.update(seed=seed, experiment=name, run_dir=(ROOT / name).as_posix())
        path = ROOT / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        started = time.time()
        with open(ROOT / f"{name}.train.log", "w", encoding="utf-8") as handle:
            code = subprocess.run([sys.executable, "-u", "-m", "gtp.rootmotion.train", "--config", str(path)],
                                  stdout=handle, stderr=subprocess.STDOUT).returncode
        print(f"[{time.strftime('%H:%M:%S')}] train {name} exit={code} {time.time() - started:.0f}s", flush=True)
        if code:
            raise SystemExit(code)

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "train"))

from config_loader import load_experiment_config
from training_log import TrainingLogger

EXPERIMENTS = [
    ("configs/exp_e1_yolov8n.yaml", "ultralytics", "weights/best.pt"),
    ("configs/exp_e2_yolo11m.yaml", "ultralytics", "weights/best.pt"),
    ("configs/exp_e3_rtdetr.yaml", "ultralytics", "weights/best.pt"),
    ("configs/exp_e4_faster_rcnn.yaml", "torchvision", "weights/best.pth"),
    ("configs/exp_e5_retinanet.yaml", "torchvision", "weights/best.pth"),
]

LOG_PATH = ROOT / "outputs" / "log.md"


def is_done(config_rel: str, weights_rel: str) -> bool:
    cfg = load_experiment_config(ROOT / config_rel)
    weights = ROOT / "outputs" / "experiments" / cfg["name"] / weights_rel
    return weights.exists()


def main() -> None:
    subprocess.run([sys.executable, "scripts/recover_experiments.py"], cwd=ROOT, check=False)

    logger = TrainingLogger(LOG_PATH)
    logger.log_session_start(
        "run_remaining_experiments",
        {"started_at": datetime.now(timezone.utc).isoformat()},
    )

    failed: list[str] = []
    for config, backend, weights_rel in EXPERIMENTS:
        if is_done(config, weights_rel):
            cfg = load_experiment_config(ROOT / config)
            print(f"SKIP (done): {cfg['experiment_id']} {cfg['name']}")
            continue

        if backend == "ultralytics":
            cmd = [sys.executable, "src/train/train_ultralytics.py", "--config", config, "--log", str(LOG_PATH)]
        else:
            cmd = [sys.executable, "src/train/train_torchvision.py", "--config", config, "--log", str(LOG_PATH)]
        print(f"\n{'=' * 60}\nStarting {config}\n{'=' * 60}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failed.append(config)

    subprocess.run([
        sys.executable, "scripts/fix_dataset_paths.py",
    ], cwd=ROOT, check=False)
    subprocess.run([
        sys.executable, "src/eval/metrics.py",
        "--experiments-dir", "outputs/experiments",
        "--output", "outputs/experiments_table.csv",
    ], cwd=ROOT, check=False)

    best_pt = ROOT / "outputs" / "experiments" / "YOLO11m_main" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        (ROOT / "models" / "best").mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, ROOT / "models" / "best" / "weights.pt")

    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    print("Remaining experiments complete.")


if __name__ == "__main__":
    main()
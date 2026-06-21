from __future__ import annotations

import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.training_log import TrainingLogger

EXPERIMENTS = [
    ("configs/exp_e1_yolov8n.yaml", "ultralytics"),
    ("configs/exp_e2_yolo11m.yaml", "ultralytics"),
    ("configs/exp_e3_rtdetr.yaml", "ultralytics"),
    ("configs/exp_e4_faster_rcnn.yaml", "torchvision"),
    ("configs/exp_e5_retinanet.yaml", "torchvision"),
]

LOG_PATH = ROOT / "outputs" / "log.md"


PLAN_12H = {
    "budget_hours": 12,
    "common_config": "configs/train_common.yaml",
    "shared_params": "imgsz=640, batch=4, epochs=50, AdamW, lr0=0.001, cosine, patience=15",
    "note": "Одинаковые гиперпараметры для E1–E5 по заданию (п. 2.6, 3)",
}


def main() -> None:
    data_yaml = ROOT / "data/processed/dota_ship_patches/dota_ship_patches.yaml"
    if not data_yaml.exists():
        print("Run scripts/prepare_all.py first.")
        sys.exit(1)

    subprocess.run([sys.executable, "scripts/fix_dataset_paths.py"], cwd=ROOT, check=True)

    logger = TrainingLogger(LOG_PATH)
    session_start = time.time()
    logger.log_session_start(
        "run_all_experiments (план 12ч)",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "experiments": ", ".join(e[0] for e in EXPERIMENTS),
            "data_yaml": str(data_yaml),
            **PLAN_12H,
        },
    )

    failed: list[str] = []
    for config, backend in EXPERIMENTS:
        if backend == "ultralytics":
            cmd = [sys.executable, "src/train/train_ultralytics.py", "--config", config, "--log", str(LOG_PATH)]
        else:
            cmd = [sys.executable, "src/train/train_torchvision.py", "--config", config, "--log", str(LOG_PATH)]
        print(f"\n{'='*60}\nStarting {config}\n{'='*60}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failed.append(config)
            logger.append(f"\n**WARNING:** `{config}` завершился с кодом {result.returncode}\n")

    subprocess.run([
        sys.executable, "src/eval/metrics.py",
        "--experiments-dir", "outputs/experiments",
        "--output", "outputs/experiments_table.csv",
    ], cwd=ROOT, check=not failed)

    best_pt = ROOT / "outputs" / "experiments" / "YOLO11m_main" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, ROOT / "models" / "best" / "weights.pt")
        subprocess.run([
            sys.executable, "src/eval/metrics.py",
            "--bootstrap", "--best-model", str(best_pt),
        ], cwd=ROOT, check=False)
        subprocess.run([
            sys.executable, "src/eval/ablation.py", "--model", str(best_pt),
        ], cwd=ROOT, check=False)

    logger.log_comparison_summary(ROOT / "outputs" / "experiments")
    elapsed = time.time() - session_start
    logger.log_run_end(
        experiment_id="ALL",
        name="run_all_experiments",
        status="success" if not failed else "partial",
        elapsed_sec=elapsed,
        extra={
            "failed_configs": ", ".join(failed) if failed else "none",
            "experiments_table": "outputs/experiments_table.csv",
        },
    )

    if failed:
        print(f"Failed experiments: {failed}")
        sys.exit(1)
    print("All experiments complete. See outputs/experiments_table.csv and outputs/log.md")


if __name__ == "__main__":
    main()
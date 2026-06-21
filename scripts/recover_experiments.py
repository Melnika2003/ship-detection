from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "train"))

from config_loader import load_experiment_config
from training_log import TrainingLogger, extract_ultralytics_final_metrics, read_ultralytics_results_csv

TARGET = ROOT / "outputs" / "experiments"
FALLBACK_ROOT = ROOT / "runs" / "detect" / "outputs" / "experiments"
LOG_PATH = ROOT / "outputs" / "log.md"

ULTRALYTICS_EXPS = [
    "configs/exp_e1_yolov8n.yaml",
    "configs/exp_e2_yolo11m.yaml",
]


def recover_one(config_rel: str) -> bool:
    cfg = load_experiment_config(ROOT / config_rel)
    name = cfg["name"]
    exp_id = cfg["experiment_id"]
    src = FALLBACK_ROOT / name
    dst = TARGET / name

    best_pt = src / "weights" / "best.pt"
    results_csv = src / "results.csv"
    if not best_pt.exists() and not results_csv.exists():
        print(f"SKIP {exp_id}: no artifacts in {src}")
        return False

    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    epoch_rows = read_ultralytics_results_csv(dst)
    final_metrics = extract_ultralytics_final_metrics(None, dst)
    if results_csv.exists():
        lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > 1:
            last = lines[-1].split(",")
            final_metrics.setdefault("final_metrics/mAP50(B)", float(last[7]))
            final_metrics.setdefault("final_metrics/mAP50-95(B)", float(last[8]))

    meta = {
        "experiment_id": exp_id,
        "architecture": cfg["architecture"],
        "pretrained": cfg.get("pretrained", "COCO"),
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "optimizer": cfg.get("optimizer"),
        "lr0": cfg.get("lr0"),
        "recovered_from": str(src),
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "best_weights": str(dst / "weights" / "best.pt"),
        "metrics": final_metrics,
        "model_type": "ultralytics",
    }
    if results_csv.exists():
        times = results_csv.read_text(encoding="utf-8").strip().splitlines()
        if len(times) > 1:
            meta["training_time_sec"] = round(float(times[-1].split(",")[1]), 1)

    (dst / "experiment_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger = TrainingLogger(LOG_PATH)
    if epoch_rows:
        logger.log_epochs_table(exp_id, epoch_rows)
    logger.log_run_end(
        experiment_id=exp_id,
        name=name,
        status="success",
        elapsed_sec=meta.get("training_time_sec", 0),
        weights_path=dst / "weights" / "best.pt",
        metrics=final_metrics,
        extra={"recovered": True, "source": str(src)},
    )
    print(f"OK {exp_id}: recovered -> {dst}")
    return True


def main() -> None:
    ok = sum(recover_one(c) for c in ULTRALYTICS_EXPS)
    print(f"Recovered {ok}/{len(ULTRALYTICS_EXPS)} ultralytics experiments.")


if __name__ == "__main__":
    main()
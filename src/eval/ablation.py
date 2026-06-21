from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def run_threshold_ablation(model_path: Path, data_yaml: Path, conf_values: list[float], iou_values: list[float]) -> pd.DataFrame:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    rows = []
    for conf in conf_values:
        for iou in iou_values:
            from src.eval.metrics import load_train_imgsz
            metrics = model.val(data=str(data_yaml), conf=conf, iou=iou, imgsz=load_train_imgsz(), verbose=False)
            rows.append({
                "conf": conf,
                "iou_nms": iou,
                "mAP50": float(metrics.box.map50),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/dota_ship_patches/dota_ship_patches.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = run_threshold_ablation(
        args.model, args.data,
        conf_values=[0.15, 0.25, 0.40],
        iou_values=[0.5, 0.7],
    )
    out = args.output_dir / "nms_conf_ablation.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {out}")

    test_dir = Path("data/processed/dota_ship_hbb/images/test")
    if test_dir.exists():
        import subprocess
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "tiling_ablation.py"),
            "--model", str(args.model),
            "--test-dir", str(test_dir),
            "--output-dir", str(args.output_dir),
        ], check=False)
    else:
        print(f"Tiling ablation skipped: {test_dir} not found")


if __name__ == "__main__":
    main()
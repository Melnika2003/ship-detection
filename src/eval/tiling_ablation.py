from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


def run_tiling_ablation(
    model_path: Path,
    test_images_dir: Path,
    conf: float = 0.25,
    iou: float = 0.5,
    patch_size: int = 1024,
) -> pd.DataFrame:
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from src.inference.predictor import ShipPredictor

    configs = [
        {"mode": "no_tiling", "use_tiling": False, "overlap": 0},
        {"mode": "tiling_overlap_0", "use_tiling": True, "overlap": 0},
        {"mode": "tiling_overlap_200", "use_tiling": True, "overlap": 200},
    ]

    test_images = sorted(
        p for p in test_images_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".tif", ".bmp"}
    )[:20]

    rows = []
    for cfg in configs:
        predictor = ShipPredictor(
            model_path, conf=conf, iou=iou,
            patch_size=patch_size, overlap=cfg["overlap"],
            use_tiling=cfg["use_tiling"],
        )
        total_dets, total_latency = 0, 0.0
        for img_path in test_images:
            result = predictor.predict_image(img_path)
            total_dets += len(result.detections)
            total_latency += result.latency_ms

        rows.append({
            "mode": cfg["mode"],
            "use_tiling": cfg["use_tiling"],
            "overlap": cfg["overlap"],
            "avg_detections": round(total_dets / max(1, len(test_images)), 2),
            "avg_latency_ms": round(total_latency / max(1, len(test_images)), 2),
            "images_tested": len(test_images),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=Path("data/processed/dota_ship_hbb/images/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.test_dir.exists():
        print(f"Test dir not found: {args.test_dir}. Run data preparation first.")
        return

    df = run_tiling_ablation(args.model, args.test_dir)
    out = args.output_dir / "tiling_ablation.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
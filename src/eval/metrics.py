from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


DEFAULT_IMGSZ = 640


def load_train_imgsz() -> int:
    common = Path("configs/train_common.yaml")
    if common.exists():
        data = yaml.safe_load(common.read_text(encoding="utf-8")) or {}
        return int(data.get("imgsz", DEFAULT_IMGSZ))
    return DEFAULT_IMGSZ


def resolve_eval_split(data: dict) -> str:
    root = Path(data["path"])
    image_exts = {".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for split in ("test", "val"):
        rel = data.get(split)
        if not rel:
            continue
        images_dir = root / rel
        if not images_dir.exists():
            continue
        if any(p.suffix.lower() in image_exts for p in images_dir.iterdir() if p.is_file()):
            return split
    return "val"


def measure_latency_ultralytics(model_path: Path, imgsz: int = DEFAULT_IMGSZ, n_warmup: int = 10, n_runs: int = 50) -> float:
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(str(model_path))
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(n_warmup):
        model.predict(dummy, verbose=False)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return float(sum(times) / len(times))


def evaluate_ultralytics(model_path: Path, data_yaml: Path, imgsz: int = DEFAULT_IMGSZ) -> dict:
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(str(model_path))
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    eval_split = resolve_eval_split(data)
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False, split=eval_split)

    result = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "fp": int(getattr(metrics.box, "f1", [0])[0] if hasattr(metrics.box, "f1") else 0),
        "fn": 0,
        "weights_mb": round(model_path.stat().st_size / 1024 / 1024, 2),
        "latency_ms": round(measure_latency_ultralytics(model_path, imgsz), 2),
        "model_type": "ultralytics",
    }

    data_root = Path(data["path"])
    split_labels = data_root / "labels" / eval_split
    split_images = data_root / "images" / eval_split
    image_exts = {".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    sample_images = sorted(
        p for p in split_images.glob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    )[:50]
    if sample_images:
        preds = model.predict(
            [str(p) for p in sample_images],
            conf=0.25, iou=0.5, verbose=False,
        )
        gt_count = sum(
            len([l for l in (split_labels / f"{Path(r.path).stem}.txt").read_text(encoding="utf-8").splitlines() if l.strip()])
            for r in preds if (split_labels / f"{Path(r.path).stem}.txt").exists()
        )
        pred_count = sum(len(r.boxes) if r.boxes is not None else 0 for r in preds)
        result["fp"] = max(0, pred_count - gt_count)
        result["fn"] = max(0, gt_count - pred_count)
    result["eval_split"] = eval_split

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), verbose=False)
        result["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

    return result


def evaluate_experiment(exp_dir: Path, data_yaml: Path) -> dict | None:
    meta_path = exp_dir / "experiment_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    pt_weights = exp_dir / "weights" / "best.pt"
    pth_weights = exp_dir / "weights" / "best.pth"

    imgsz = int(meta.get("imgsz", load_train_imgsz()))
    if pt_weights.exists():
        metrics = evaluate_ultralytics(pt_weights, data_yaml, imgsz=imgsz)
    elif pth_weights.exists():
        from src.eval.eval_torchvision import evaluate_torchvision
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        metrics = evaluate_torchvision(
            pth_weights, Path(data["path"]), split="val", imgsz=imgsz,
        )
        metrics["model_type"] = "torchvision"
    else:
        return None

    return {
        "experiment": exp_dir.name,
        "experiment_id": meta.get("experiment_id", ""),
        "architecture": meta.get("architecture", exp_dir.name),
        "training_time_sec": meta.get("training_time_sec"),
        **metrics,
    }


def collect_experiments(experiments_dir: Path, data_yaml: Path) -> pd.DataFrame:
    rows = []
    if not experiments_dir.exists():
        return pd.DataFrame()
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        print(f"Evaluating {exp_dir.name}...")
        row = evaluate_experiment(exp_dir, data_yaml)
        if row:
            rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def bootstrap_map_estimates(
    model_path: Path,
    data_yaml: Path,
    n_bootstrap: int = 100,
    seed: int = 42,
    imgsz: int | None = None,
    max_pool: int = 80,
    sample_size: int = 40,
) -> dict:
    from ultralytics import YOLO
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    imgsz = imgsz or load_train_imgsz()
    model = YOLO(str(model_path))
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data_root = Path(data["path"])
    eval_split = resolve_eval_split(data)
    test_images = sorted((data_root / "images" / eval_split).glob("*"))
    test_images = [p for p in test_images if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    if len(test_images) < 5:
        return {"error": f"Not enough {eval_split} images for bootstrap"}

    pool = test_images if len(test_images) <= max_pool else random.sample(test_images, max_pool)
    scores = []
    n = min(n_bootstrap, 20)
    for _ in range(n):
        sample = random.choices(pool, k=min(sample_size, len(pool)))
        det_count = 0
        for img_path in sample:
            preds = model.predict(
                str(img_path), conf=0.25, verbose=False, imgsz=imgsz, stream=False,
            )
            det_count += sum(len(r.boxes) if r.boxes is not None else 0 for r in preds)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        scores.append(det_count / max(1, len(sample)))

    arr = np.array(scores)
    return {
        "metric": "detections_per_image",
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "ci95_low": round(float(np.percentile(arr, 2.5)), 4),
        "ci95_high": round(float(np.percentile(arr, 97.5)), 4),
        "n_bootstrap": n,
        "eval_split": eval_split,
    }


def main() -> None:
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/dota_ship_patches/dota_ship_patches.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments_table.csv"))
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--best-model", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from dataset_yaml import normalize_dataset_yaml

    data_yaml = normalize_dataset_yaml(args.data.resolve())
    df = collect_experiments(args.experiments_dir, data_yaml)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No experiments found. Train models first.")
    print(f"Saved: {args.output}")

    if args.bootstrap and args.best_model and args.best_model.exists():
        try:
            stability = bootstrap_map_estimates(args.best_model, data_yaml, imgsz=load_train_imgsz())
            stability_path = args.output.parent / "bootstrap_stability.json"
            stability_path.write_text(json.dumps(stability, indent=2), encoding="utf-8")
            print(f"Bootstrap: {stability}")
        except Exception as exc:
            print(f"Bootstrap skipped: {exc}")


if __name__ == "__main__":
    main()
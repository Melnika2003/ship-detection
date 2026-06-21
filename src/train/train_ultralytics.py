from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from config_loader import load_experiment_config
from training_log import (
    TrainingLogger,
    adjust_batch_for_device,
    collect_environment,
    extract_ultralytics_final_metrics,
    read_ultralytics_results_csv,
    resolve_train_device,
)


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg and "memory" in msg


def train_with_oom_retry(model, train_kw: dict):
    import torch

    batch = int(train_kw["batch"])
    last_exc: BaseException | None = None
    while batch >= 2:
        train_kw["batch"] = batch
        try:
            return model.train(**train_kw), batch
        except RuntimeError as exc:
            last_exc = exc
            if not _is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch = max(2, batch // 2)
            print(f"CUDA OOM — повтор с batch={batch}")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("CUDA OOM даже при batch=2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/dota_ship_patches/dota_ship_patches.yaml"))
    parser.add_argument("--project", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--log", type=Path, default=Path("outputs/log.md"))
    parser.add_argument("--device", default=None, help="Override device: 0, cpu, etc.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count (e.g. 2 for smoke test)")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    logger = TrainingLogger(args.log)
    exp_id = cfg.get("experiment_id", "?")
    name = cfg.get("name", "unknown")
    start = time.time()
    project_dir = (Path.cwd() / args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = project_dir / name

    device, device_notes = resolve_train_device(cfg.get("device", 0), args.device)
    batch, batch_notes = adjust_batch_for_device(cfg["batch"], device)
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]

    env = collect_environment()
    env.update(device_notes)
    env.update(batch_notes)

    if device_notes.get("warning"):
        print(f"WARNING: {device_notes['warning']}")

    effective_cfg = {**cfg, "device": device, "batch": batch, "epochs": epochs}
    logger.log_run_start(
        experiment_id=exp_id,
        name=name,
        backend="ultralytics",
        config_path=args.config,
        cfg=effective_cfg,
        environment=env,
    )

    try:
        from ultralytics import YOLO

        sys_path = Path(__file__).resolve().parents[1] / "data"
        if str(sys_path) not in sys.path:
            sys.path.insert(0, str(sys_path))
        from dataset_yaml import normalize_dataset_yaml

        data_yaml = normalize_dataset_yaml(args.data.resolve())
        print(f"Dataset yaml path (this machine): {data_yaml}")
        import yaml as _yaml
        _data = _yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        print(f"Dataset root: {_data['path']}")

        use_cpu = str(device).lower() == "cpu"
        model = YOLO(cfg["model"])
        model_name = str(cfg["model"]).lower()
        mosaic = cfg.get("mosaic", 1.0)
        if "rtdetr" in model_name and mosaic:
            mosaic = 0.0
            print("RT-DETR: mosaic отключён (иначе mixed tensor sizes 640/1280)")
        train_kw: dict = {
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": cfg["imgsz"],
            "batch": batch,
            "optimizer": cfg.get("optimizer", "AdamW"),
            "lr0": cfg.get("lr0", 0.001),
            "patience": cfg.get("patience", 10),
            "device": device,
            "project": str(project_dir),
            "name": cfg["name"],
            "exist_ok": True,
            "pretrained": True,
            "save": True,
            "plots": cfg.get("plots", False),
            "cache": cfg.get("cache", "ram"),
            "amp": cfg.get("amp", True) and not use_cpu,
            "cos_lr": cfg.get("cos_lr", True),
            "close_mosaic": cfg.get("close_mosaic", 5),
            "workers": 0 if use_cpu else cfg.get("workers", 4),
            "fraction": cfg.get("fraction", 1.0),
            "val": cfg.get("val", True),
            "augment": cfg.get("augment", True),
            "mosaic": mosaic,
        }
        if cfg.get("multi_scale"):
            train_kw["multi_scale"] = True
        print(
            f"Train: imgsz={train_kw['imgsz']} epochs={train_kw['epochs']} "
            f"batch={train_kw['batch']} fraction={train_kw['fraction']} val={train_kw['val']}"
        )
        initial_batch = batch
        results, batch = train_with_oom_retry(model, train_kw)
        if batch != initial_batch:
            print(f"Использован batch={batch} (в конфиге было {initial_batch})")
        elapsed = time.time() - start

        save_dir = Path(getattr(results, "save_dir", exp_dir))
        if hasattr(results, "trainer") and getattr(results.trainer, "save_dir", None):
            save_dir = Path(results.trainer.save_dir)
        exp_dir = save_dir.resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)

        best_weights = exp_dir / "weights" / "best.pt"
        final_metrics = extract_ultralytics_final_metrics(results, exp_dir)
        epoch_rows = read_ultralytics_results_csv(exp_dir)
        logger.log_epochs_table(exp_id, epoch_rows)

        meta = {
            "experiment_id": cfg["experiment_id"],
            "architecture": cfg["architecture"],
            "pretrained": cfg["pretrained"],
            "imgsz": cfg["imgsz"],
            "batch": batch,
            "epochs": epochs,
            "device": str(device),
            "optimizer": cfg.get("optimizer"),
            "lr0": cfg.get("lr0"),
            "training_time_sec": round(elapsed, 1),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "best_weights": str(best_weights),
            "metrics": final_metrics,
            **device_notes,
        }
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.log_run_end(
            experiment_id=exp_id,
            name=name,
            status="success",
            elapsed_sec=elapsed,
            weights_path=best_weights,
            metrics=final_metrics,
            extra={"device": str(device), "batch": batch},
        )
        print(f"Training done in {elapsed:.1f}s. Weights: {best_weights}")
        print(f"Log appended to {args.log}")

    except Exception as exc:
        elapsed = time.time() - start
        logger.log_run_end(
            experiment_id=exp_id,
            name=name,
            status="failed",
            elapsed_sec=elapsed,
            error=traceback.format_exc(),
            extra={"exception": str(exc), "device": str(device)},
        )
        raise


if __name__ == "__main__":
    main()
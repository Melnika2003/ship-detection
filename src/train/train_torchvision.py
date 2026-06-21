from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from config_loader import load_experiment_config
from training_log import TrainingLogger, collect_environment
from yolo_dataset import YoloDetectionDataset, collate_fn


def build_model(model_name: str, num_classes: int = 2):
    if model_name == "faster_rcnn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        model = retinanet_resnet50_fpn(weights=weights)
        num_anchors = model.head.classification_head.num_anchors
        model.head = RetinaNetHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )
    return model, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/dota_ship_patches"))
    parser.add_argument("--project", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--log", type=Path, default=Path("outputs/log.md"))
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    logger = TrainingLogger(args.log)
    exp_id = cfg.get("experiment_id", "?")
    name = cfg.get("name", "unknown")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgsz = cfg.get("imgsz")
    train_ds = YoloDetectionDataset(
        args.data_root / "images" / "train",
        args.data_root / "labels" / "train",
        imgsz=imgsz,
    )
    val_ds = YoloDetectionDataset(
        args.data_root / "images" / "val",
        args.data_root / "labels" / "val",
        imgsz=imgsz,
    )
    if imgsz:
        print(f"Resize patches to {imgsz}x{imgsz} for torchvision training")
    if len(train_ds) == 0:
        raise RuntimeError("No training images with labels found. Run data preparation first.")

    fraction = cfg.get("fraction", 1.0)
    if fraction < 1.0:
        k = max(1, int(len(train_ds.images) * fraction))
        train_ds.images = random.sample(train_ds.images, k)
        print(f"Fast mode: using {k}/{int(k / fraction)} train images (fraction={fraction})")

    val_every = max(1, cfg.get("val_every", 1))
    print(f"Fast mode: val_every={val_every} epochs, batch={cfg['batch']}")

    logger.log_run_start(
        experiment_id=exp_id,
        name=name,
        backend="torchvision",
        config_path=args.config,
        cfg=cfg,
        environment=collect_environment(train_samples=len(train_ds), val_samples=len(val_ds)),
    )

    num_workers = 0 if device.type == "cpu" else cfg.get("workers", 4)
    pin_memory = device.type == "cuda"
    loader_kw = dict(collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch"], shuffle=False, **loader_kw)

    model, weights = build_model(cfg["model"])
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = str(cfg.get("optimizer", "AdamW")).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=cfg["lr0"], momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg["lr0"], weight_decay=0.0005)
    scheduler = None
    if cfg.get("cos_lr", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    exp_dir = args.project / cfg["name"]
    (exp_dir / "weights").mkdir(parents=True, exist_ok=True)
    start = time.time()
    best_loss = float("inf")
    epochs_without_improve = 0
    patience = cfg.get("patience", 10)
    val_every = max(1, cfg.get("val_every", 3))
    use_amp = cfg.get("amp", True) and device.type == "cuda" and cfg["model"] != "retinanet"
    if cfg["model"] == "retinanet" and cfg.get("amp", True):
        print("RetinaNet: AMP отключён (стабильность AdamW, без NaN)")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epoch_rows: dict[int, dict[str, float]] = {}
    stopped_early = False

    try:
        for epoch in range(cfg["epochs"]):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for images, targets in train_loader:
                images = [weights.transforms()(img).to(device, non_blocking=pin_memory) for img in images]
                targets = [{k: v.to(device, non_blocking=pin_memory) for k, v in t.items()} for t in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                if cfg["model"] == "retinanet":
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(1, n_batches)
            run_val = (epoch + 1) % val_every == 0 or epoch + 1 == cfg["epochs"]
            avg_val = float("nan")

            if run_val:
                model.train()
                val_loss = 0.0
                n_val = 0
                for images, targets in val_loader:
                    images = [weights.transforms()(img).to(device, non_blocking=pin_memory) for img in images]
                    targets = [{k: v.to(device, non_blocking=pin_memory) for k, v in t.items()} for t in targets]
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss_dict = model(images, targets)
                        batch_loss = sum(loss_dict.values())
                    if torch.isfinite(batch_loss):
                        val_loss += batch_loss.item()
                    n_val += 1
                model.eval()
                avg_val = val_loss / max(1, n_val)

            if scheduler is not None:
                scheduler.step()

            val_str = f"{avg_val:.4f}" if run_val else "skipped"
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{cfg['epochs']} train_loss={avg_train:.4f} "
                f"val_loss={val_str} lr={lr_now:.6f}"
            )

            epoch_metrics = {"train_loss": round(avg_train, 4)}
            if run_val:
                epoch_metrics["val_loss"] = round(avg_val, 4)
            epoch_rows[epoch + 1] = epoch_metrics
            logger.log_epoch(exp_id, epoch + 1, cfg["epochs"], epoch_metrics)

            if run_val:
                if avg_val < best_loss:
                    best_loss = avg_val
                    epochs_without_improve = 0
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_name": cfg["model"],
                        "num_classes": 2,
                    }, exp_dir / "weights" / "best.pth")
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve * val_every >= patience:
                        print(f"Early stopping: no val improvement for {patience} epochs")
                        stopped_early = True
                        break
            elif epoch == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": cfg["model"],
                    "num_classes": 2,
                }, exp_dir / "weights" / "best.pth")

        elapsed = time.time() - start
        logger.log_epochs_table(exp_id, epoch_rows)

        best_weights = exp_dir / "weights" / "best.pth"
        meta = {
            "experiment_id": cfg["experiment_id"],
            "architecture": cfg["architecture"],
            "pretrained": cfg.get("pretrained", "COCO"),
            "imgsz": cfg["imgsz"],
            "batch": cfg["batch"],
            "epochs": cfg["epochs"],
            "training_time_sec": round(elapsed, 1),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "best_val_loss": round(best_loss, 4),
            "model_type": "torchvision",
            "epoch_history": {str(k): v for k, v in epoch_rows.items()},
        }
        (exp_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.log_run_end(
            experiment_id=exp_id,
            name=name,
            status="success",
            elapsed_sec=elapsed,
            weights_path=best_weights,
            metrics={
                "best_val_loss": round(best_loss, 4),
                "final_train_loss": epoch_rows[max(epoch_rows)]["train_loss"],
                "early_stop": stopped_early,
            },
        )
        print(f"Done in {elapsed:.1f}s. Weights: {best_weights}")
        print(f"Log appended to {args.log}")

    except Exception as exc:
        elapsed = time.time() - start
        logger.log_run_end(
            experiment_id=exp_id,
            name=name,
            status="failed",
            elapsed_sec=elapsed,
            error=traceback.format_exc(),
            extra={"exception": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
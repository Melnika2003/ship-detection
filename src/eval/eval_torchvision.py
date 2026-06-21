from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.transforms import functional as F

from src.train.yolo_dataset import YoloDetectionDataset, collate_fn


def load_torchvision_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "faster_rcnn")
    num_classes = ckpt.get("num_classes", 2)

    if model_name == "faster_rcnn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        model = retinanet_resnet50_fpn(weights=None)
        num_anchors = model.head.classification_head.num_anchors
        model.head = RetinaNetHead(in_channels=256, num_anchors=num_anchors, num_classes=num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, weights


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_torchvision(
    checkpoint_path: Path,
    data_root: Path,
    split: str = "val",
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    imgsz: int | None = 640,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weights = load_torchvision_model(checkpoint_path, device)
    ds = YoloDetectionDataset(
        data_root / "images" / split,
        data_root / "labels" / split,
        skip_empty=True,
        imgsz=imgsz,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    tp, fp, fn = 0, 0, 0
    latencies = []

    with torch.no_grad():
        for images, targets in loader:
            t0 = time.perf_counter()
            imgs = [weights.transforms()(img).to(device) for img in images]
            outputs = model(imgs)
            latencies.append((time.perf_counter() - t0) * 1000)

            gt_boxes = targets[0]["boxes"].tolist()
            pred = outputs[0]
            mask = pred["scores"] >= conf_thresh
            pred_boxes = pred["boxes"][mask].cpu().tolist()
            pred_scores = pred["scores"][mask].cpu().tolist()

            matched_gt = set()
            for pb, score in sorted(zip(pred_boxes, pred_scores), key=lambda x: -x[1]):
                best_iou, best_idx = 0.0, -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_idx = iou, gi
                if best_iou >= iou_thresh and best_idx >= 0:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    return {
        "mAP50": round(precision * recall, 4),
        "mAP50_95": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "weights_mb": round(checkpoint_path.stat().st_size / 1024 / 1024, 2),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/dota_ship_patches"))
    args = parser.parse_args()
    result = evaluate_torchvision(args.checkpoint, args.data_root)
    print(json.dumps(result, indent=2))
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str = "ship"


@dataclass
class PredictionResult:
    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    image_path: str = ""
    model_version: str = "1.0.0"


def _load_model_config(weights_path: Path) -> dict:
    config_path = weights_path.parent / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {}


def _load_model_version(weights_path: Path) -> str:
    meta_path = weights_path.parent / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("version", meta.get("architecture", "1.0.0"))
    return "1.0.0"


class ShipPredictor:
    def __init__(
        self,
        weights_path: Path,
        conf: float | None = None,
        iou: float | None = None,
        patch_size: int | None = None,
        overlap: int | None = None,
        use_tiling: bool | None = None,
    ):
        from ultralytics import YOLO

        cfg = _load_model_config(weights_path)
        self.model = YOLO(str(weights_path))
        self.conf = conf if conf is not None else cfg.get("conf", 0.25)
        self.iou = iou if iou is not None else cfg.get("iou", 0.5)
        self.patch_size = patch_size if patch_size is not None else cfg.get("patch_size", 1024)
        self.overlap = overlap if overlap is not None else cfg.get("overlap", 200)
        self.use_tiling = use_tiling if use_tiling is not None else cfg.get("use_tiling", True)
        self.model_version = _load_model_version(weights_path)

    def _predict_patch(self, patch: np.ndarray) -> list[Detection]:
        results = self.model.predict(patch, conf=self.conf, iou=self.iou, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(Detection(x1, y1, x2, y2, float(box.conf[0]), "ship"))
        return dets

    def _nms(self, detections: list[Detection], iou_thresh: float = 0.5) -> list[Detection]:
        if not detections:
            return []
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
        scores = np.array([d.confidence for d in detections])
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            iou = inter / (area_i + area_o - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]
        return [detections[i] for i in keep]

    def _tile_positions(self, length: int) -> list[int]:
        if length <= self.patch_size:
            return [0]
        stride = self.patch_size - self.overlap
        positions = list(range(0, length - self.patch_size + 1, stride))
        last = length - self.patch_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def predict_image(self, image_path: Path) -> PredictionResult:
        import cv2

        t0 = time.perf_counter()
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]
        all_dets: list[Detection] = []

        if self.use_tiling and (h > self.patch_size or w > self.patch_size):
            for y in self._tile_positions(h):
                for x in self._tile_positions(w):
                    patch = img[y:y + self.patch_size, x:x + self.patch_size]
                    if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                        patch = cv2.copyMakeBorder(
                            patch,
                            0, self.patch_size - patch.shape[0],
                            0, self.patch_size - patch.shape[1],
                            cv2.BORDER_CONSTANT, value=(114, 114, 114),
                        )
                    patch_dets = self._predict_patch(patch)
                    for d in patch_dets:
                        all_dets.append(Detection(d.x1 + x, d.y1 + y, d.x2 + x, d.y2 + y, d.confidence))
            all_dets = self._nms(all_dets, self.iou)
        else:
            resized = cv2.resize(img, (self.patch_size, self.patch_size))
            scale_x, scale_y = w / self.patch_size, h / self.patch_size
            patch_dets = self._predict_patch(resized)
            for d in patch_dets:
                all_dets.append(Detection(d.x1 * scale_x, d.y1 * scale_y, d.x2 * scale_x, d.y2 * scale_y, d.confidence))

        latency = (time.perf_counter() - t0) * 1000
        return PredictionResult(
            detections=all_dets, latency_ms=latency,
            image_path=str(image_path), model_version=self.model_version,
        )

    def draw_detections(self, image_path: Path, result: PredictionResult) -> np.ndarray:
        import cv2

        img = cv2.imread(str(image_path))
        for d in result.detections:
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{d.confidence:.2f}", (int(d.x1), max(0, int(d.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img
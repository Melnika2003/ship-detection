from __future__ import annotations

from pathlib import Path

import torch
from torchvision.transforms import functional as F


class YoloDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        skip_empty: bool = True,
        imgsz: int | None = None,
    ):
        all_images = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".tif", ".bmp"}
        )
        self.labels_dir = labels_dir
        self.images: list[Path] = []
        for img_path in all_images:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if skip_empty and label_path.exists():
                has_box = any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())
                if not has_box:
                    continue
            elif skip_empty:
                continue
            self.images.append(img_path)
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        from PIL import Image

        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        boxes, labels = [], []
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = max(0.0, (cx - bw / 2) * w)
                y1 = max(0.0, (cy - bh / 2) * h)
                x2 = min(w, (cx + bw / 2) * w)
                y2 = min(h, (cy + bh / 2) * h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)

        if self.imgsz and (w != self.imgsz or h != self.imgsz):
            sx, sy = self.imgsz / w, self.imgsz / h
            image = image.resize((self.imgsz, self.imgsz), Image.BILINEAR)
            boxes = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))
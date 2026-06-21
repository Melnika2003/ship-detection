from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from PIL import Image


def load_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        labels.append((cls, cx, cy, w, h))
    return labels


def yolo_to_abs(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    bw, bh = w * img_w, h * img_h
    x1 = cx * img_w - bw / 2
    y1 = cy * img_h - bh / 2
    x2 = x1 + bw
    y2 = y1 + bh
    return x1, y1, x2, y2


def abs_to_yolo(x1: float, y1: float, x2: float, y2: float, patch_w: int, patch_h: int) -> str | None:
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(patch_w), x2), min(float(patch_h), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2 / patch_w
    cy = (y1 + y2) / 2 / patch_h
    w = (x2 - x1) / patch_w
    h = (y2 - y1) / patch_h
    if w < 0.005 or h < 0.005:
        return None
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def box_intersects_patch(
    bx1: float, by1: float, bx2: float, by2: float,
    px: int, py: int, patch_size: int,
    min_overlap: float = 0.3,
) -> bool:
    ix1 = max(bx1, px)
    iy1 = max(by1, py)
    ix2 = min(bx2, px + patch_size)
    iy2 = min(by2, py + patch_size)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    box_area = (bx2 - bx1) * (by2 - by1)
    return inter / box_area >= min_overlap


def split_image(
    image_path: Path,
    label_path: Path,
    out_images: Path,
    out_labels: Path,
    patch_size: int,
    overlap: int,
    keep_empty: bool = False,
) -> int:
    try:
        img = Image.open(image_path).convert("RGB")
    except OSError:
        return 0

    img_w, img_h = img.size
    labels = load_yolo_labels(label_path)
    abs_boxes = [yolo_to_abs(*l[1:], img_w, img_h) for l in labels]

    stride = patch_size - overlap
    saved = 0
    stem = image_path.stem

    def _positions(length: int) -> list[int]:
        if length <= patch_size:
            return [0]
        positions = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    for y in _positions(img_h):
        for x in _positions(img_w):
            x2 = min(x + patch_size, img_w)
            y2 = min(y + patch_size, img_h)
            patch = img.crop((x, y, x2, y2))
            pw, ph = patch.size

            if pw < patch_size or ph < patch_size:
                padded = Image.new("RGB", (patch_size, patch_size), (114, 114, 114))
                padded.paste(patch, (0, 0))
                patch = padded

            patch_labels = []
            for (bx1, by1, bx2, by2) in abs_boxes:
                if not box_intersects_patch(bx1, by1, bx2, by2, x, y, patch_size):
                    continue
                line = abs_to_yolo(bx1 - x, by1 - y, bx2 - x, by2 - y, patch_size, patch_size)
                if line:
                    patch_labels.append(line)

            if not patch_labels and not keep_empty:
                continue

            patch_name = f"{stem}_{x}_{y}"
            patch.save(out_images / f"{patch_name}.jpg", "JPEG", quality=95)
            (out_labels / f"{patch_name}.txt").write_text(
                "\n".join(patch_labels) + ("\n" if patch_labels else ""),
                encoding="utf-8",
            )
            saved += 1

    return saved


def process_split(
    src_dir: Path,
    dst_dir: Path,
    split: str,
    patch_size: int,
    overlap: int,
) -> int:
    src_images = src_dir / "images" / split
    src_labels = src_dir / "labels" / split
    dst_images = dst_dir / "images" / split
    dst_labels = dst_dir / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    total = 0
    for img_path in sorted(src_images.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".png", ".tif", ".bmp"}:
            continue
        lbl_path = src_labels / f"{img_path.stem}.txt"
        total += split_image(img_path, lbl_path, dst_images, dst_labels, patch_size, overlap)
    return total


def write_dataset_yaml(output_dir: Path) -> None:
    from dataset_yaml import write_dataset_yaml as write_yaml

    write_yaml(
        output_dir / "dota_ship_patches.yaml",
        {
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "ship"},
            "nc": 1,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split DOTA images into patches")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/dota_ship_hbb"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dota_ship_patches"))
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    grand_total = 0
    for split in ("train", "val", "test"):
        n = process_split(args.input_dir, args.output_dir, split, args.patch_size, args.overlap)
        print(f"{split}: {n} patches")
        grand_total += n

    write_dataset_yaml(args.output_dir)
    print(f"Total patches: {grand_total}")
    print(f"Dataset YAML: {args.output_dir / 'dota_ship_patches.yaml'}")


if __name__ == "__main__":
    main()
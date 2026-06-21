from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

SHIP_CLASS = "ship"
TARGET_CLASS_ID = 0


def parse_dota_label(label_path: Path) -> list[tuple[str, list[float]]]:
    objects = []
    if not label_path.exists():
        return objects
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        coords = [float(x) for x in parts[:8]]
        class_name = parts[8]
        difficult = parts[9] if len(parts) > 9 else "0"
        if difficult == "1":
            continue
        objects.append((class_name, coords))
    return objects


def obb_to_hbb(coords: list[float]) -> tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def to_yolo_line(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> str | None:
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(float(img_w), xmax)
    ymax = min(float(img_h), ymax)
    if xmax <= xmin or ymax <= ymin:
        return None
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    if w < 0.001 or h < 0.001:
        return None
    return f"{TARGET_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def get_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size


def convert_split(
    raw_dir: Path,
    output_dir: Path,
    split: str,
    split_list_path: Path | None = None,
) -> int:
    images_src = raw_dir / "images" / split
    labels_src = raw_dir / "labelTxt" / split
    if not images_src.exists():
        images_src = raw_dir / split / "images"
        labels_src = raw_dir / split / "labelTxt"

    images_dst = output_dir / "images" / split
    labels_dst = output_dir / "labels" / split
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    image_ids: list[str] | None = None
    if split_list_path and split_list_path.exists():
        image_ids = []
        for line in split_list_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            stem = Path(line.replace("\\", "/")).stem
            image_ids.append(stem)

    count = 0
    for label_file in sorted(labels_src.glob("*.txt")):
        stem = label_file.stem
        if image_ids is not None and stem not in image_ids:
            continue

        image_path = None
        for ext in (".png", ".jpg", ".tif", ".bmp"):
            candidate = images_src / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        objects = parse_dota_label(label_file)
        ships = [obj for obj in objects if obj[0].lower() == SHIP_CLASS]
        if not ships:
            continue

        try:
            img_w, img_h = get_image_size(image_path)
        except OSError:
            continue

        yolo_lines = []
        for _, coords in ships:
            xmin, ymin, xmax, ymax = obb_to_hbb(coords)
            line = to_yolo_line(xmin, ymin, xmax, ymax, img_w, img_h)
            if line:
                yolo_lines.append(line)

        if not yolo_lines:
            continue

        out_img = images_dst / image_path.name
        out_lbl = labels_dst / f"{stem}.txt"
        if not out_img.exists():
            shutil.copy2(image_path, out_img)
        out_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        count += 1

    return count


def write_dataset_yaml(output_dir: Path) -> None:
    from dataset_yaml import write_dataset_yaml as write_yaml

    write_yaml(
        output_dir / "dota_ship.yaml",
        {
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {TARGET_CLASS_ID: SHIP_CLASS},
            "nc": 1,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DOTA v1 to YOLO format (ships only)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/DOTA"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dota_ship_hbb"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": args.raw_dir / "train.txt",
        "val": args.raw_dir / "val.txt",
        "test": args.raw_dir / "test.txt",
    }

    total = 0
    for split, list_path in splits.items():
        n = convert_split(args.raw_dir, args.output_dir, split, list_path if list_path.exists() else None)
        print(f"{split}: {n} images with ships")
        total += n

    write_dataset_yaml(args.output_dir)
    print(f"Total: {total} images. Dataset YAML: {args.output_dir / 'dota_ship.yaml'}")


if __name__ == "__main__":
    main()
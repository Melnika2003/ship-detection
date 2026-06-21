from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/best/weights.pt"
IMG_DIR = ROOT / "data/processed/dota_ship_patches/images/val"
LBL_DIR = ROOT / "data/processed/dota_ship_patches/labels/val"
OUT = ROOT / "outputs/error_analysis"


def count_gt(stem: str) -> int:
    lbl = LBL_DIR / f"{stem}.txt"
    if not lbl.exists():
        return 0
    return sum(1 for line in lbl.read_text(encoding="utf-8").splitlines() if line.strip())


def classify_sample(model: YOLO, images: list[Path], seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    sample = random.sample(images, min(30, len(images)))
    rows = []
    for img in sample:
        gt = count_gt(img.stem)
        result = model.predict(str(img), conf=0.25, imgsz=640, verbose=False)[0]
        pred = len(result.boxes) if result.boxes is not None else 0
        diff = pred - gt
        if diff == 0 and gt > 0:
            kind = "good"
        elif diff > 0:
            kind = "fp"
        else:
            kind = "fn"
        rows.append(
            {"image": img.name, "gt": gt, "pred": pred, "kind": kind, "path": img, "result": result}
        )
    return pd.DataFrame(rows)


def save_example(row: pd.Series, prefix: str, idx: int) -> Path:
    img = Image.open(row["path"]).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    r = row["result"]
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=2
            )
            ax.add_patch(rect)
    ax.set_title(f"{row['kind']}: gt={row['gt']} pred={row['pred']}")
    ax.axis("off")
    out = OUT / f"{prefix}_{idx:02d}_{row['image']}"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def main() -> None:
    if not WEIGHTS.exists():
        print(f"Missing {WEIGHTS}. Run: python scripts/collect_results.py")
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted(IMG_DIR.glob("*.jpg"))
    print(f"Модель: {WEIGHTS}")
    print(f"Val images: {len(images)}")

    model = YOLO(str(WEIGHTS))
    df = classify_sample(model, images)
    print("\nКлассификация примеров:")
    print(df.groupby("kind").size().to_string())

    saved = []
    for kind, prefix in (("good", "good"), ("fp", "error_fp"), ("fn", "error_fn")):
        part = df[df["kind"] == kind].head(5)
        for i, (_, row) in enumerate(part.iterrows()):
            saved.append(save_example(row, prefix, i))

    print(f"\nСохранено: {len(saved)} файлов в {OUT}")
    for p in saved[:5]:
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/dota_ship_patches"
OUT = ROOT / "outputs"


def count_splits() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        img_dir = DATA / "images" / split
        lbl_dir = DATA / "labels" / split
        if not img_dir.exists():
            continue
        n_img = len(list(img_dir.glob("*")))
        n_box = 0
        for lbl in lbl_dir.glob("*.txt"):
            n_box += sum(
                1 for line in lbl.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        rows.append({"split": split, "images": n_img, "boxes": n_box})
    return pd.DataFrame(rows)


def collect_bbox_areas() -> list[float]:
    areas = []
    for split in ("train", "val"):
        for lbl in (DATA / "labels" / split).glob("*.txt"):
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    w, h = float(parts[3]), float(parts[4])
                    areas.append(w * h)
    return areas


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print(f"DATA: {DATA}")
    print(f"exists: {DATA.exists()}")

    stats = count_splits()
    print("\nSplit statistics:")
    print(stats.to_string(index=False))

    areas = collect_bbox_areas()
    print(f"\nВсего bbox: {len(areas)}")
    print(f"Мелкие (<0.001): {sum(a < 0.001 for a in areas)}")
    print(f"Средние: {sum(0.001 <= a < 0.01 for a in areas)}")
    print(f"Крупные (>=0.01): {sum(a >= 0.01 for a in areas)}")

    plt.figure(figsize=(8, 4))
    plt.hist(areas, bins=40)
    plt.title("Распределение площади bbox")
    plt.xlabel("w * h (норм.)")
    plt.ylabel("число")
    plt.tight_layout()
    out_path = OUT / "eda_bbox_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs/experiments_table.csv"
OUT = ROOT / "outputs"


def main() -> None:
    if not CSV.exists():
        print(f"Missing {CSV}. Run: python scripts/collect_results.py")
        sys.exit(1)

    df = pd.read_csv(CSV).sort_values("mAP50", ascending=False)

    cols = [
        "experiment_id",
        "architecture",
        "mAP50",
        "mAP50_95",
        "precision",
        "recall",
        "latency_ms",
        "weights_mb",
        "training_time_sec",
    ]
    cols = [c for c in cols if c in df.columns]
    report = df[cols].round(4)
    print("Experiments table:")
    print(report.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df.plot.bar(x="architecture", y="mAP50", ax=axes[0], legend=False, color="steelblue")
    axes[0].set_title("mAP@0.5")
    axes[0].tick_params(axis="x", rotation=30)

    df.plot.bar(x="architecture", y="latency_ms", ax=axes[1], legend=False, color="coral")
    axes[1].set_title("Latency (ms)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = OUT / "experiments_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")

    best = df.iloc[0]
    print(f"Лучшая модель: {best['architecture']} (mAP50={best['mAP50']:.4f})")


if __name__ == "__main__":
    main()
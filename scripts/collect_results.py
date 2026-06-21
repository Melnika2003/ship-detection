from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([sys.executable, "scripts/fix_dataset_paths.py"], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "src/eval/metrics.py",
        "--experiments-dir", "outputs/experiments",
        "--output", "outputs/experiments_table.csv",
    ], cwd=ROOT, check=True)

    best_pt = ROOT / "outputs" / "experiments" / "YOLO11m_main" / "weights" / "best.pt"
    if best_pt.exists():
        (ROOT / "models" / "best").mkdir(parents=True, exist_ok=True)
        (ROOT / "weights").mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, ROOT / "models" / "best" / "weights.pt")
        shutil.copy2(best_pt, ROOT / "weights" / "best.pt")
        subprocess.run([
            sys.executable, "src/eval/metrics.py",
            "--bootstrap", "--best-model", str(best_pt),
        ], cwd=ROOT, check=False)
        print("Best model copied to models/best/weights.pt and weights/best.pt")
    print("Done: outputs/experiments_table.csv")


if __name__ == "__main__":
    main()
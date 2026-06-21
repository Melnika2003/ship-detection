from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/eda_dota.py",
    "scripts/experiments_analysis.py",
    "scripts/error_analysis.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n{'=' * 60}\n>>> {script}\n{'=' * 60}")
        subprocess.run([sys.executable, script], cwd=ROOT, check=True)
    print("\nDone. Outputs in outputs/ and outputs/error_analysis/")


if __name__ == "__main__":
    main()
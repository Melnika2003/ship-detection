from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    raw_candidates = [
        ROOT / "data" / "raw" / "DOTA",
        Path("Z:/work/практика/ship-detection-dota/data/raw/DOTA"),
    ]
    raw = next((p for p in raw_candidates if (p / "images" / "train").exists()), raw_candidates[0])
    if not raw.exists() or not (raw / "images" / "train").exists():
        print("DOTA not found — downloading...")
        run([sys.executable, "scripts/download_dota.py", "--source", "auto", "--keep-cache"])

    raw_rel = raw.relative_to(ROOT) if raw.is_relative_to(ROOT) else raw
    run([sys.executable, "src/data/convert_dota_to_yolo.py",
         "--raw-dir", str(raw_rel), "--output-dir", "data/processed/dota_ship_hbb"])
    run([sys.executable, "src/data/split_patches.py",
         "--input-dir", "data/processed/dota_ship_hbb", "--output-dir", "data/processed/dota_ship_patches"])
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
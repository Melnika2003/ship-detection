from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))

from dataset_yaml import normalize_dataset_yaml

YAML_FILES = [
    ROOT / "data/processed/dota_ship_patches/dota_ship_patches.yaml",
    ROOT / "data/processed/dota_ship_hbb/dota_ship.yaml",
]


def main() -> None:
    print(f"Project root: {ROOT}")
    ok = 0
    for yaml_path in YAML_FILES:
        if not yaml_path.exists():
            print(f"SKIP (missing): {yaml_path}")
            continue
        try:
            normalize_dataset_yaml(yaml_path)
            import yaml
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            print(f"OK: {yaml_path.name}")
            print(f"    path -> {data['path']}")
            for split in ("train", "val", "test"):
                if split in data:
                    n = len(list((Path(data['path']) / data[split]).glob('*')))
                    print(f"    {split}: {n} files")
            ok += 1
        except FileNotFoundError as exc:
            print(f"FAIL: {yaml_path.name}\n  {exc}")
            sys.exit(1)
    if ok == 0:
        print("No yaml files found. Run: python scripts/prepare_all.py")
        sys.exit(1)
    print("\nPaths fixed for this machine. You can start training.")


if __name__ == "__main__":
    main()
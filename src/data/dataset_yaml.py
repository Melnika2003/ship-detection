from __future__ import annotations

from pathlib import Path

import yaml


def dataset_root_from_yaml(yaml_path: Path) -> Path:
    return yaml_path.resolve().parent


def write_dataset_yaml(yaml_path: Path, data: dict) -> Path:
    payload = {
        "path": ".",
        **{k: v for k, v in data.items() if k != "path"},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.dump(payload, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


def normalize_dataset_yaml(yaml_path: Path) -> Path:
    yaml_path = yaml_path.resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    root = dataset_root_from_yaml(yaml_path)
    data["path"] = str(root)

    for split in ("train", "val", "test"):
        rel = data.get(split)
        if not rel:
            continue
        split_dir = (root / rel).resolve()
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split '{split}' not found: {split_dir}\n"
                f"YAML: {yaml_path}\n"
                f"Запустите на этой машине: python scripts/fix_dataset_paths.py"
            )

    yaml_path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path
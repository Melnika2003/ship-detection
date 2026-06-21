from __future__ import annotations

from pathlib import Path

import yaml

COMMON_FILE = "train_common.yaml"


def load_experiment_config(config_path: Path) -> dict:
    config_path = config_path.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    common_path = config_path.parent / COMMON_FILE
    if common_path.exists():
        common = yaml.safe_load(common_path.read_text(encoding="utf-8")) or {}
        return {**common, **cfg}
    return cfg
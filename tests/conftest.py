from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_image(tmp_path) -> Path:
    try:
        import cv2
        import numpy as np
    except ImportError:
        pytest.skip("opencv not installed")

    img_path = tmp_path / "sample.jpg"
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.imwrite(str(img_path), img)
    return img_path
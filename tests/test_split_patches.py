from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip("cv2")

from src.data.split_patches import split_image


def test_split_image_creates_patches(tmp_path):
    img_path = tmp_path / "test.png"
    lbl_path = tmp_path / "test.txt"
    out_img = tmp_path / "out_img"
    out_lbl = tmp_path / "out_lbl"
    out_img.mkdir()
    out_lbl.mkdir()

    img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    lbl_path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    n = split_image(img_path, lbl_path, out_img, out_lbl, patch_size=1024, overlap=200)
    assert n >= 4
    assert len(list(out_img.glob("*.jpg"))) == n
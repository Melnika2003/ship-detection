from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.convert_dota_to_yolo import obb_to_hbb, parse_dota_label, to_yolo_line


def test_obb_to_hbb():
    coords = [100, 50, 200, 50, 200, 150, 100, 150]
    xmin, ymin, xmax, ymax = obb_to_hbb(coords)
    assert xmin == 100
    assert ymin == 50
    assert xmax == 200
    assert ymax == 150


def test_to_yolo_line():
    line = to_yolo_line(100, 50, 200, 150, 1000, 1000)
    assert line is not None
    parts = line.split()
    assert parts[0] == "0"
    cx, cy, w, h = map(float, parts[1:])
    assert 0 < cx < 1
    assert 0 < cy < 1
    assert 0 < w < 1
    assert 0 < h < 1


def test_parse_dota_label_skips_difficult():
    with tempfile.TemporaryDirectory() as tmp:
        label_file = Path(tmp) / "test.txt"
        label_file.write_text(
            "100 50 200 50 200 150 100 150 ship 0\n"
            "300 50 400 50 400 150 300 150 ship 1\n",
            encoding="utf-8",
        )
        objects = parse_dota_label(label_file)
        assert len(objects) == 1
        assert objects[0][0] == "ship"


def test_parse_dota_label_multiple_ships():
    with tempfile.TemporaryDirectory() as tmp:
        label_file = Path(tmp) / "test.txt"
        label_file.write_text(
            "100 50 200 50 200 150 100 150 ship 0\n"
            "300 50 400 50 400 150 300 150 harbor 0\n"
            "500 50 600 50 600 150 500 150 ship 0\n",
            encoding="utf-8",
        )
        objects = parse_dota_label(label_file)
        ships = [o for o in objects if o[0] == "ship"]
        assert len(ships) == 2
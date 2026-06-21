from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service.api.main import app


def _weights_exist() -> bool:
    return (ROOT / "weights" / "best.pt").exists() or (
        ROOT / "models" / "best" / "weights.pt"
    ).exists()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_predict_sample_ship(client):
    samples_dir = ROOT / "data" / "samples"
    images = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
    if not images:
        pytest.skip("No sample images in data/samples/")
    if not _weights_exist():
        pytest.skip("Model weights not found (weights/best.pt)")

    with open(images[0], "rb") as f:
        resp = client.post("/predict", files={"file": (images[0].name, f, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "detections" in data
    assert "latency_ms" in data
    assert "num_detections" in data
    assert data["latency_ms"] >= 0
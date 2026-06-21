from __future__ import annotations

import base64
import os
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from service.db.database import RunDatabase
from src.inference.predictor import ShipPredictor


def _resolve_weights() -> Path:
    env = os.getenv("MODEL_WEIGHTS")
    if env:
        return Path(env)
    for candidate in (
        ROOT / "weights" / "best.pt",
        ROOT / "models" / "best" / "weights.pt",
    ):
        if candidate.exists():
            return candidate
    return ROOT / "weights" / "best.pt"


WEIGHTS = _resolve_weights()
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "service" / "db" / "runs.db"))
VIDEO_MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "0"))
db = RunDatabase(DB_PATH)
predictor: ShipPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    if WEIGHTS.exists():
        predictor = ShipPredictor(WEIGHTS)
    else:
        predictor = None
    yield


app = FastAPI(title="Ship Detection DOTA API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if predictor else "no_model",
        "model_loaded": predictor is not None,
        "model_version": predictor.model_version if predictor else None,
        "weights": str(WEIGHTS),
    }


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "image.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp:
        shutil.copyfileobj(upload.file, tmp)
    return Path(tmp.name)


def _result_to_dict(result, vis_path: Path | None = None) -> dict:
    payload = {
        "detections": [
            {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2, "confidence": d.confidence, "class": d.class_name}
            for d in result.detections
        ],
        "num_detections": len(result.detections),
        "avg_confidence": round(
            sum(d.confidence for d in result.detections) / max(1, len(result.detections)), 4
        ),
        "latency_ms": round(result.latency_ms, 2),
        "model_version": result.model_version,
    }
    if vis_path and vis_path.exists():
        img_bytes = vis_path.read_bytes()
        payload["visualization_b64"] = base64.b64encode(img_bytes).decode("ascii")
    return payload


def _cleanup(*paths: Path) -> None:
    for p in paths:
        p.unlink(missing_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if predictor is None:
        raise HTTPException(503, "Model weights not found. Place weights at models/best/weights.pt")
    tmp_path = _save_upload(file)
    vis_path = tmp_path.with_suffix(".vis.jpg")
    try:
        result = predictor.predict_image(tmp_path)
        import cv2

        vis = predictor.draw_detections(tmp_path, result)
        cv2.imwrite(str(vis_path), vis)
        avg_conf = sum(d.confidence for d in result.detections) / max(1, len(result.detections))
        db.log_run(str(file.filename), result.model_version, len(result.detections), avg_conf, result.latency_ms)
        return _result_to_dict(result, vis_path)
    except Exception as e:
        db.log_run(str(file.filename), "unknown", 0, 0.0, 0.0, status="error", error_message=str(e))
        raise HTTPException(500, str(e)) from e
    finally:
        _cleanup(tmp_path, vis_path)


@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)) -> dict:
    if predictor is None:
        raise HTTPException(503, "Model weights not found")
    results = []
    for f in files:
        tmp_path = _save_upload(f)
        try:
            result = predictor.predict_image(tmp_path)
            avg_conf = sum(d.confidence for d in result.detections) / max(1, len(result.detections))
            db.log_run(str(f.filename), result.model_version, len(result.detections), avg_conf, result.latency_ms)
            results.append(_result_to_dict(result))
        finally:
            _cleanup(tmp_path)
    return {"results": results, "count": len(results)}


@app.post("/video_predict")
async def video_predict(file: UploadFile = File(...)) -> dict:
    if predictor is None:
        raise HTTPException(503, "Model weights not found")
    tmp_path = _save_upload(file)
    frame_paths: list[Path] = []
    try:
        import cv2
        import time as _time

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise HTTPException(400, "Cannot open video")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_results = []
        all_confidences: list[float] = []
        frame_idx = 0
        t0_total = _time.perf_counter()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = tmp_path.parent / f"frame_{frame_idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            result = predictor.predict_image(frame_path)
            frame_confs = [d.confidence for d in result.detections]
            all_confidences.extend(frame_confs)
            frame_avg = (
                round(sum(frame_confs) / len(frame_confs), 4) if frame_confs else 0.0
            )
            frame_results.append({
                "frame": frame_idx,
                "detections": len(result.detections),
                "avg_confidence": frame_avg,
                "latency_ms": result.latency_ms,
            })
            frame_idx += 1
            if VIDEO_MAX_FRAMES > 0 and frame_idx >= VIDEO_MAX_FRAMES:
                break
        cap.release()
        total_time = (_time.perf_counter() - t0_total) * 1000
        fps = frame_idx / (total_time / 1000) if total_time > 0 else 0
        avg_confidence = (
            round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0
        )
        total_detections = sum(r["detections"] for r in frame_results)
        db.log_run(
            str(file.filename),
            predictor.model_version,
            total_detections,
            avg_confidence,
            total_time / max(1, frame_idx),
        )
        truncated = total_frames > 0 and frame_idx < total_frames
        return {
            "frames_processed": frame_idx,
            "total_frames": total_frames,
            "truncated": truncated,
            "avg_confidence": avg_confidence,
            "total_detections": total_detections,
            "fps": round(fps, 2),
            "frame_results": frame_results,
        }
    finally:
        _cleanup(tmp_path, *frame_paths)


@app.get("/stats")
def stats() -> dict:
    return {"history": db.get_recent(20), "summary": db.get_stats()}
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import httpx
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")


def predict_image(image_path: str) -> tuple[str, str | None, pd.DataFrame]:
    if not image_path:
        return "Загрузите изображение", None, pd.DataFrame()
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        resp = httpx.post(f"{API_URL}/predict", files=files, timeout=120.0)
    if resp.status_code != 200:
        return f"Ошибка: {resp.text}", None, pd.DataFrame()
    data = resp.json()
    summary = (
        f"Обнаружено: {data['num_detections']}\n"
        f"Средняя уверенность: {data.get('avg_confidence', 0)}\n"
        f"Задержка: {data['latency_ms']} ms\n"
        f"Модель: {data['model_version']}"
    )
    vis_path = None
    if "visualization_b64" in data:
        import base64
        import tempfile
        vis_bytes = base64.b64decode(data["visualization_b64"])
        vis_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        Path(vis_path).write_bytes(vis_bytes)
    return summary, vis_path or image_path, _fetch_history()


def predict_video(video_path: str) -> tuple[str, pd.DataFrame]:
    if not video_path:
        return "Загрузите видео", pd.DataFrame()
    timeout = float(os.getenv("VIDEO_TIMEOUT_SEC", "1800"))
    with open(video_path, "rb") as f:
        files = {"file": (Path(video_path).name, f, "video/mp4")}
        resp = httpx.post(f"{API_URL}/video_predict", files=files, timeout=timeout)
    if resp.status_code != 200:
        return f"Ошибка: {resp.text}", pd.DataFrame()
    data = resp.json()
    total_frames = data.get("total_frames", 0)
    frames_line = f"Кадров обработано: {data['frames_processed']}"
    if total_frames:
        frames_line += f" / {total_frames}"
    if data.get("truncated"):
        frames_line += " (обрезано лимитом VIDEO_MAX_FRAMES)"
    summary = (
        f"{frames_line}\n"
        f"FPS: {data['fps']}\n"
        f"Детекций: {data.get('total_detections', sum(r['detections'] for r in data['frame_results']))}\n"
        f"Средняя уверенность: {data.get('avg_confidence', 0)}"
    )
    return summary, _fetch_history()


def _fetch_history() -> pd.DataFrame:
    try:
        resp = httpx.get(f"{API_URL}/stats", timeout=10.0)
        if resp.status_code == 200:
            return pd.DataFrame(resp.json().get("history", []))
    except Exception:
        pass
    return pd.DataFrame()


def check_health() -> str:
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5.0)
        return str(resp.json())
    except Exception as e:
        return f"API недоступен: {e}"


with gr.Blocks(title="Ship Detection DOTA") as demo:
    gr.Markdown("# Детекция кораблей на аэрофотоснимках (DOTA v1)")
    gr.Markdown(f"API: `{API_URL}`")
    gr.Textbox(label="Статус API", value=check_health(), interactive=False)

    with gr.Tab("Изображение"):
        with gr.Row():
            img_input = gr.Image(type="filepath", label="Загрузить изображение")
            img_output = gr.Image(label="Результат детекции")
        img_summary = gr.Textbox(label="Статистика")
        img_history = gr.Dataframe(label="История")
        img_input.change(predict_image, inputs=img_input, outputs=[img_summary, img_output, img_history])

    with gr.Tab("Видео"):
        vid_input = gr.Video(label="Загрузить видео")
        vid_summary = gr.Textbox(label="Статистика видео")
        vid_history = gr.Dataframe(label="История")
        vid_input.change(predict_video, inputs=vid_input, outputs=[vid_summary, vid_history])

    with gr.Tab("История запусков"):
        history_out = gr.Dataframe(label="Последние 20 запусков", value=_fetch_history())
        refresh_btn = gr.Button("Обновить")
        refresh_btn.click(_fetch_history, outputs=history_out)

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    print(f"Gradio UI: http://{host}:{port}")
    print(f"API endpoint: {API_URL}")
    if host == "127.0.0.1":
        print("Сначала запустите API: uvicorn service.api.main:app --host 127.0.0.1 --port 8000")
    demo.launch(
        server_name=host,
        server_port=port,
        inbrowser=False,
        show_error=True,
    )
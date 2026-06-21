# Ship Object Detection Service

Сервис детекции кораблей на аэрофотоснимках **DOTA v1**. Production-модель: **YOLO11m** (`weights/best.pt`), класс `ship`, mAP@0.5 = **0.9037** на val.

## Возможности

- REST API (FastAPI): детекция на изображении, пакетная обработка, видео
- Gradio UI для загрузки изображений и видео
- SQLite — история запусков и статистика
- Docker: `docker compose up --build`
- Тесты: smoke + regression на реальных снимках
- Пайплайн практики: сравнение 5 архитектур (E1–E5)

## Быстрый старт

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn service.api.main:app --reload --host 0.0.0.0 --port 8000
```

Проверка: http://127.0.0.1:8000/health

### Gradio UI (опционально)

Во втором терминале:

```bash
python service/ui/app.py
```

Открыть: http://127.0.0.1:7860

## Docker

```bash
docker compose up --build
```

- API: http://127.0.0.1:8000
- Gradio: http://127.0.0.1:7860

## API

### `GET /health`

Статус сервиса и загрузки модели.

### `POST /predict`

Загрузка изображения (`multipart/form-data`, поле `file`).

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@data/samples/sample_ship.jpg"
```

### `POST /batch_predict`

Пакетная детекция (несколько файлов).

### `POST /video_predict`

Детекция на видео (все кадры; лимит `VIDEO_MAX_FRAMES`).

### `GET /stats`

История последних запусков и сводная статистика.

Документация OpenAPI: http://127.0.0.1:8000/docs

## Структура проекта

```
ship-detection-dota/
├── configs/
│   ├── inference.yaml      # runtime: conf, tiling, paths
│   ├── train_common.yaml   # общие гиперпараметры E1–E5
│   └── exp_e*.yaml
├── models/
│   └── model_info.yaml     # метаданные лучшей модели
├── weights/
│   └── best.pt             # YOLO11m (~39 MB)
├── gif/
│   └── recording.gif       # демонстрация работы UI
├── service/
│   ├── api/main.py         # FastAPI
│   └── ui/app.py           # Gradio
├── src/                    # обучение, eval, inference
├── scripts/
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Тесты

```bash
pytest tests/test_smoke.py -q
pytest tests/ -v
```

## Примечания

- Веса лежат в `weights/best.pt` (в репозитории)
- Конфиг инференса: `configs/inference.yaml`
- Данные DOTA v1 (~15 GB) скачиваются локально → `data/raw/DOTA/`
- GPU: `device: "0"` в `configs/inference.yaml`

## Обучение (практика, E1–E5)

```bash
python scripts/prepare_all.py
python scripts/run_all_experiments.py
python scripts/collect_results.py
python scripts/run_analysis.py
```

| ID | Модель | mAP@0.5 (val) |
|----|--------|---------------|
| E2 | YOLO11m | **0.904** |
| E1 | YOLOv8n | 0.884 |
| E3 | RT-DETR-L | 0.814 |
| E4 | Faster R-CNN | 0.466 |
| E5 | RetinaNet | 0.451 |

## Демонстрация

![demo](gif/recording.gif)

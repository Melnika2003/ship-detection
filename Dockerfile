FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENV MODEL_WEIGHTS=/app/weights/best.pt
ENV DB_PATH=/app/service/db/runs.db
ENV PYTHONPATH=/app

EXPOSE 8000 7860

CMD ["uvicorn", "service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
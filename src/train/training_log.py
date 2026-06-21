from __future__ import annotations

import csv
import json
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path("outputs/log.md")

_LOG_HEADER = """# Training Log — Ship Detection DOTA

Автоматический журнал обучения E1–E5.

| Поле | Описание |
|------|----------|
| `experiment_id` | Идентификатор E1–E5 |
| `mAP50` / `mAP50_95` | Метрики на валидации (Ultralytics) |
| `train_loss` / `val_loss` | Loss по эпохам (torchvision) |
| `training_time_sec` | Время обучения в секундах |

---
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _md_table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Параметр | Значение |", "|----------|----------|"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def _md_table_from_dict(data: dict[str, Any], columns: list[str] | None = None) -> str:
    if not data:
        return "_нет данных_\n"
    cols = columns or list(next(iter(data.values())).keys())
    header = "| " + " | ".join(["epoch", *cols]) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = [header, sep]
    for epoch, row in sorted(data.items(), key=lambda x: int(x[0])):
        lines.append("| " + " | ".join([str(epoch), *[str(row.get(c, "")) for c in cols]]) + " |")
    return "\n".join(lines) + "\n"


class TrainingLogger:
    def __init__(self, log_path: Path | str = DEFAULT_LOG) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            self.log_path.write_text(_LOG_HEADER, encoding="utf-8")

    def append(self, text: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")

    def log_session_start(self, label: str, details: dict[str, Any] | None = None) -> None:
        block = f"\n## Сессия: {label}\n\n**Время:** {_now()}\n\n"
        if details:
            block += _md_table(list(details.items()))
        block += "\n---\n"
        self.append(block)

    def log_run_start(
        self,
        *,
        experiment_id: str,
        name: str,
        backend: str,
        config_path: Path | str,
        cfg: dict[str, Any],
        environment: dict[str, Any] | None = None,
    ) -> None:
        hp_rows = [(k, v) for k, v in cfg.items() if k not in {"name"}]
        block = (
            f"\n## [{experiment_id}] {name} — START\n\n"
            f"**Время начала:** {_now()}\n"
            f"**Конфиг:** `{config_path}`\n"
            f"**Backend:** {backend}\n\n"
            f"### Гиперпараметры\n\n{_md_table(hp_rows)}\n"
        )
        if environment:
            block += f"### Окружение\n\n{_md_table(list(environment.items()))}\n"
        block += "---\n"
        self.append(block)

    def log_epoch(
        self,
        experiment_id: str,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        row = ", ".join(f"{k}={v}" for k, v in metrics.items())
        self.append(f"- **{experiment_id}** epoch {epoch}/{total_epochs}: {row}")

    def log_epochs_table(self, experiment_id: str, epoch_rows: dict[int, dict[str, Any]]) -> None:
        if not epoch_rows:
            return
        cols = list(next(iter(epoch_rows.values())).keys())
        block = (
            f"\n### Эпохи — {experiment_id}\n\n"
            f"{_md_table_from_dict({str(k): v for k, v in epoch_rows.items()}, cols)}\n"
        )
        self.append(block)

    def log_run_end(
        self,
        *,
        experiment_id: str,
        name: str,
        status: str,
        elapsed_sec: float,
        weights_path: Path | str | None = None,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        rows: list[tuple[str, Any]] = [
            ("status", status),
            ("training_time_sec", round(elapsed_sec, 1)),
            ("training_time_min", round(elapsed_sec / 60, 1)),
        ]
        if weights_path:
            rows.append(("best_weights", f"`{weights_path}`"))
        if metrics:
            for k, v in metrics.items():
                rows.append((k, v))
        if extra:
            rows.extend(extra.items())
        if error:
            rows.append(("error", f"`{error}`"))

        block = (
            f"\n### Итог — [{experiment_id}] {name} — {status.upper()}\n\n"
            f"**Время окончания:** {_now()}\n\n"
            f"{_md_table(rows)}\n"
        )
        if error:
            block += f"\n<details><summary>Traceback</summary>\n\n```\n{error}\n```\n\n</details>\n"
        block += "---\n"
        self.append(block)

    def log_comparison_summary(self, experiments_dir: Path, output_name: str = "experiments_table.csv") -> None:
        rows: list[dict[str, Any]] = []
        for meta_path in sorted(experiments_dir.glob("*/experiment_meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            metrics = meta.get("metrics", {})
            rows.append({
                "id": meta.get("experiment_id", "?"),
                "architecture": meta.get("architecture", "?"),
                "time_min": round(meta.get("training_time_sec", 0) / 60, 1),
                "mAP50": metrics.get("metrics/mAP50(B)", metrics.get("mAP50", "—")),
                "mAP50_95": metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50_95", "—")),
                "val_loss": meta.get("best_val_loss", "—"),
            })
        if not rows:
            return

        header = "| ID | Архитектура | Время (мин) | mAP50 | mAP50-95 | val_loss |"
        sep = "|----|-------------|-------------|-------|----------|----------|"
        lines = [f"\n## Сводка сравнения — {_now()}\n", header, sep]
        for r in rows:
            lines.append(
                f"| {r['id']} | {r['architecture']} | {r['time_min']} | {r['mAP50']} | {r['mAP50_95']} | {r['val_loss']} |"
            )
        lines.append(f"\nПолная таблица: `outputs/{output_name}`\n\n---\n")
        self.append("\n".join(lines))


def resolve_train_device(
    requested: Any = 0,
    cli_override: str | int | None = None,
) -> tuple[str | int, dict[str, Any]]:
    import torch

    notes: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_requested": cli_override if cli_override is not None else requested,
    }
    if torch.cuda.is_available():
        notes["gpu"] = torch.cuda.get_device_name(0)

    target = cli_override if cli_override is not None else requested
    if str(target).lower() == "cpu":
        return "cpu", notes

    if torch.cuda.is_available():
        if isinstance(target, str) and target.isdigit():
            return int(target), notes
        return target if target is not None else 0, notes

    notes["device_fallback"] = f"{target} -> cpu"
    notes["warning"] = (
        "CUDA недоступна (torch CPU-сборка или нет GPU в VM). "
        "Обучение на CPU — очень медленно; для практики нужен GPU или хост с CUDA."
    )
    return "cpu", notes


def adjust_batch_for_device(batch: int, device: str | int) -> tuple[int, dict[str, Any]]:
    if str(device).lower() == "cpu" and batch > 4:
        adjusted = 4
        return adjusted, {"batch_adjusted": f"{batch} -> {adjusted} (CPU)"}
    return batch, {}


def collect_environment(train_samples: int | None = None, val_samples: int | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_vram_gb"] = round(props.total_memory / 1024**3, 1)
        else:
            env["torch_build"] = "cpu"
    except ImportError:
        env["cuda_available"] = False
    if train_samples is not None:
        env["train_samples"] = train_samples
    if val_samples is not None:
        env["val_samples"] = val_samples
    return env


def read_ultralytics_results_csv(exp_dir: Path) -> dict[int, dict[str, Any]]:
    csv_path = exp_dir / "results.csv"
    if not csv_path.exists():
        return {}

    epoch_rows: dict[int, dict[str, Any]] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        preferred = [
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "val/box_loss", "lr/pg0", "time",
        ]
        for row in reader:
            epoch = int(float(row.get("epoch", len(epoch_rows) + 1)))
            metrics = {}
            for col in preferred:
                if col in row and row[col] not in ("", "nan"):
                    try:
                        metrics[col] = round(float(row[col]), 4)
                    except ValueError:
                        metrics[col] = row[col]
            if metrics:
                epoch_rows[epoch] = metrics
    return epoch_rows


def extract_ultralytics_final_metrics(results: Any, exp_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if hasattr(results, "results_dict") and results.results_dict:
        for k, v in results.results_dict.items():
            if isinstance(v, (int, float)):
                metrics[k] = round(float(v), 4) if isinstance(v, float) else v
            else:
                metrics[k] = v

    epoch_rows = read_ultralytics_results_csv(exp_dir)
    if epoch_rows:
        last = epoch_rows[max(epoch_rows)]
        for k, v in last.items():
            metrics.setdefault(f"final_{k}", v)
    return metrics
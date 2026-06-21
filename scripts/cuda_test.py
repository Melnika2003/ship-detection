from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def run_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        return subprocess.check_output(
            [exe], stderr=subprocess.STDOUT, text=True, timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"<nvidia-smi error: {exc}>"


def main() -> int:
    print("CUDA Diagnostic — ship-detection-dota")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"CWD: {Path.cwd()}")

    section("1. NVIDIA Driver (nvidia-smi)")
    smi = run_nvidia_smi()
    if smi is None:
        fail("nvidia-smi не найден в PATH")
        warn("Драйвер NVIDIA не установлен или GPU не виден ОС / VM")
    else:
        ok("nvidia-smi доступен")
        for line in smi.splitlines()[:12]:
            print(f"    {line}")

    section("2. Переменные окружения")
    for var in ("CUDA_VISIBLE_DEVICES", "CUDA_PATH", "PATH"):
        val = os.environ.get(var)
        if var == "PATH":
            cuda_in_path = any("cuda" in p.lower() for p in (val or "").split(os.pathsep))
            print(f"  {var}: ...cuda в PATH = {cuda_in_path}")
        else:
            print(f"  {var}: {val!r}")

    section("3. PyTorch")
    try:
        import torch
    except ImportError:
        fail("torch не установлен")
        print("\nРешение: pip install torch torchvision")
        return 1

    ok(f"torch {torch.__version__}")
    cuda_built = torch.backends.cuda.is_built()
    print(f"  torch.backends.cuda.is_built(): {cuda_built}")
    print(f"  torch.version.cuda (сборка):    {torch.version.cuda}")

    if "+cpu" in torch.__version__ or torch.version.cuda is None:
        fail("Установлена CPU-сборка PyTorch (без CUDA)")
        warn("Переустановите: https://pytorch.org/get-started/locally/")
        warn("Пример: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    else:
        ok(f"CUDA-сборка PyTorch (cu{torch.version.cuda})")

    section("4. torch.cuda")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    print(f"  torch.cuda.is_available(): {available}")
    print(f"  torch.cuda.device_count(): {count}")

    if not available:
        fail("PyTorch не видит CUDA")
        if smi and "+cpu" not in torch.__version__:
            warn("nvidia-smi работает, но torch.cuda=False — несовпадение версий CUDA/driver и torch")
        elif not smi:
            warn("Возможно: VirtualBox без GPU passthrough, или драйвер не установлен в VM")
    else:
        ok("CUDA доступна для PyTorch")
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            mem_gb = round(props.total_memory / 1024**3, 1)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb} GB)")

    section("5. Тест вычислений на GPU")
    if available:
        try:
            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x.T
            torch.cuda.synchronize()
            ok(f"Матричное умножение на cuda:0 — {tuple(y.shape)}")
            peak_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            print(f"  peak GPU memory: {peak_mb} MB")
        except Exception as exc:
            fail(f"Ошибка при вычислении на GPU: {exc}")
    else:
        warn("Пропущено — CUDA недоступна")

    section("6. Ultralytics device")
    try:
        from ultralytics.utils.torch_utils import select_device

        for req in ("0", "cpu"):
            try:
                dev = select_device(req)
                ok(f"select_device('{req}') -> {dev}")
            except Exception as exc:
                fail(f"select_device('{req}') -> {exc}")
    except ImportError:
        warn("ultralytics не установлен — пропуск")

    section("7. Итог и рекомендации")
    issues: list[str] = []
    if smi is None:
        issues.append("GPU не виден системе (nvidia-smi). Проверьте драйвер и GPU passthrough в VM.")
    if "+cpu" in torch.__version__ or torch.version.cuda is None:
        issues.append("PyTorch CPU-only. Переустановите torch с CUDA (cu124/cu121).")
    if not available:
        issues.append("torch.cuda.is_available()=False — обучение с device=0 не запустится.")

    if not issues:
        ok("Всё в порядке — можно обучать с device=0")
        return 0

    for i, msg in enumerate(issues, 1):
        print(f"  {i}. {msg}")

    print("\nКоманда для переустановки torch (CUDA 12.4, пример):")
    print("  pip uninstall torch torchvision -y")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    print("\nПовторная проверка:")
    print("  python scripts/cuda_test.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
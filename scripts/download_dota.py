from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw" / "DOTA"
FALLBACK_OUT = Path("Z:/work/практика/ship-detection-dota/data/raw/DOTA")


def resolve_out_dir(path: Path) -> Path:
    if path.resolve() != DEFAULT_OUT.resolve():
        return path.resolve()
    try:
        c_free = shutil.disk_usage("C:/").free
        if c_free < 15 * 1024**3 and FALLBACK_OUT.drive:
            z_free = shutil.disk_usage(FALLBACK_OUT.drive + "/").free
            if z_free > 5 * 1024**3:
                print(f"Low disk space on C: ({c_free // 1024**3} GB free) -> using {FALLBACK_OUT}")
                return FALLBACK_OUT
    except OSError:
        pass
    return path.resolve()
HF_BASE = "https://huggingface.co/datasets/isaaccorley/dota/resolve/main"

HF_FILES: dict[str, int] = {
    "dotav1.0_annotations_train.tar.gz": 1_495_981,
    "dotav1.0_annotations_val.tar.gz": 436_355,
    "dotav1.0_images_train.tar.gz": 10_183_034_823,
    "dotav1.0_images_val.tar.gz": 3_331_270_337,
}

HF_ARCHIVES: dict[str, dict[str, str]] = {
    "train": {
        "images": "dotav1.0_images_train.tar.gz",
        "annotations": "dotav1.0_annotations_train.tar.gz",
    },
    "val": {
        "images": "dotav1.0_images_val.tar.gz",
        "annotations": "dotav1.0_annotations_val.tar.gz",
    },
}

GDRIVE_SPLITS: dict[str, dict] = {
    "train": {
        "images": [
            ("part1.zip", "1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2"),
            ("part2.zip", "1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v"),
            ("part3.zip", "1pEmwJtugIWhiwgBqOtplNUtTG2T454zn"),
        ],
        "labelTxt": ("labelTxt.zip", "1I-faCP-DOxf6mxcjUTc8mYVPqUgSQxx6"),
        "split_list": ("Train_Task2_gt.zip", "1sS9hveKtYAiTsGVxC4msF5qJjhn3wYpY"),
    },
    "val": {
        "images": [("part1.zip", "1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP")],
        "labelTxt": ("labelTxt.zip", "1uFwxA4B7H8zcI1oD11bj0U8z88qroMlG"),
        "split_list": ("Val_Task2_gt.zip", "1roMkDBK9753uS5tCmtYlRTyzrObjjJ83"),
    },
    "test": {
        "images": [
            ("part1.zip", "1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK"),
            ("part2.zip", "1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv"),
        ],
        "test_info": ("test_info.json", "1nQokIxSy3DEHImJribSCODTRkWlPJLE3"),
    },
}


def curl_download(url: str, output: Path, retries: int = 5, expected_size: int | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if expected_size is None:
        expected_size = HF_FILES.get(output.name)
    if output.exists() and output.stat().st_size > 0:
        if expected_size and output.stat().st_size < expected_size:
            print(
                f"  resume: {output.name} "
                f"({output.stat().st_size // 1024 // 1024} / {expected_size // 1024 // 1024} MB)"
            )
        else:
            print(f"  skip (exists): {output.name} ({output.stat().st_size // 1024 // 1024} MB)")
            return output

    last_err = ""
    for attempt in range(1, retries + 1):
        print(f"  curl [{attempt}/{retries}]: {output.name}")
        cmd = [
            "curl", "-L", "--retry", "5", "--retry-delay", "3",
            "-C", "-", "-o", str(output), url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output.exists() and output.stat().st_size > 0:
            if expected_size and output.stat().st_size < expected_size:
                print(
                    f"  incomplete: {output.name} "
                    f"({output.stat().st_size // 1024 // 1024} / {expected_size // 1024 // 1024} MB)"
                )
            else:
                print(f"  done: {output.name} ({output.stat().st_size // 1024 // 1024} MB)")
                return output
        last_err = result.stderr or result.stdout or f"exit {result.returncode}"
        print(f"  curl error: {last_err[:200]}")
        time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"curl failed for {output.name}: {last_err}")


def gdrive_download(file_id: str, output: Path, retries: int = 5) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"  skip (exists): {output.name}")
        return output

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  gdown [{attempt}/{retries}]: {output.name}")
            gdown.download(id=file_id, output=str(output), quiet=False, resume=True)
            if output.exists() and output.stat().st_size > 0:
                return output
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  retry: {exc}")
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"gdown failed for {output.name}: {last_err}")


def extract_zip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)
    print(f"  extracted zip: {archive.name}")


def extract_tar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)
    print(f"  extracted tar: {archive.name}")


def flatten_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    moved = 0
    for path in src_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        target = dst_dir / path.name
        if target.exists():
            continue
        shutil.move(str(path), str(target))
        moved += 1
    return moved


def collect_labeltxts(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in src_dir.rglob("*.txt"):
        if path.name.lower() in {"train.txt", "val.txt", "test.txt"}:
            continue
        target = dst_dir / path.name
        if target.exists():
            continue
        shutil.copy2(path, target)
        copied += 1
    return copied


def write_split_list(labels_dir: Path, out_txt: Path) -> None:
    names = sorted(p.stem for p in labels_dir.glob("*.txt"))
    out_txt.write_text("\n".join(f"{name}.png" for name in names) + "\n", encoding="utf-8")
    print(f"  wrote {out_txt.name} ({len(names)} entries)")


def find_split_list(extract_dir: Path) -> Path | None:
    for name in ("train.txt", "val.txt", "test.txt"):
        matches = list(extract_dir.rglob(name))
        if matches:
            return matches[0]
    txts = [p for p in extract_dir.rglob("*.txt") if p.is_file()]
    return txts[0] if len(txts) == 1 else None


def write_test_list(test_info: Path, images_dir: Path, out_txt: Path) -> None:
    if test_info.exists():
        data = json.loads(test_info.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "images" in data:
            names = [str(x) for x in data["images"]]
        elif isinstance(data, list):
            names = [str(x) for x in data]
        else:
            names = []
        if names:
            out_txt.write_text("\n".join(names) + "\n", encoding="utf-8")
            print(f"  wrote {out_txt.name} from test_info.json ({len(names)} entries)")
            return

    exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in exts)
    out_txt.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"  wrote {out_txt.name} from filenames ({len(names)} entries)")


def process_hf_split(split: str, out_dir: Path, cache: Path) -> None:
    print(f"\n=== {split} (HuggingFace) ===")
    cfg = HF_ARCHIVES[split]
    hf_cache = cache / "hf"
    work = cache / f"hf_{split}"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    if not work.exists():
        work.mkdir(parents=True, exist_ok=True)

    ann_arc = curl_download(
        f"{HF_BASE}/{cfg['annotations']}",
        hf_cache / cfg["annotations"],
        expected_size=HF_FILES[cfg["annotations"]],
    )
    img_arc = curl_download(
        f"{HF_BASE}/{cfg['images']}",
        hf_cache / cfg["images"],
        expected_size=HF_FILES[cfg["images"]],
    )

    extract_tar(ann_arc, work / "ann")
    extract_tar(img_arc, work / "img")

    images_dst = out_dir / "images" / split
    labels_dst = out_dir / "labelTxt" / split
    n_img = flatten_images(work / "img", images_dst)
    n_lbl = collect_labeltxts(work / "ann", labels_dst)
    print(f"  images/{split}: {n_img}, labelTxt/{split}: {n_lbl}")

    if labels_dst.exists():
        write_split_list(labels_dst, out_dir / f"{split}.txt")


def process_gdrive_split(split: str, cfg: dict, out_dir: Path, cache: Path) -> None:
    print(f"\n=== {split} (Google Drive) ===")
    images_dst = out_dir / "images" / split
    labels_dst = out_dir / "labelTxt" / split
    work = cache / f"gdrive_{split}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    for fname, fid in cfg.get("images", []):
        archive = gdrive_download(fid, cache / "gdrive" / split / fname)
        extract_zip(archive, work / "images_raw")

    n_img = flatten_images(work / "images_raw", images_dst)
    print(f"  images/{split}: {n_img} files")

    if "labelTxt" in cfg:
        fname, fid = cfg["labelTxt"]
        archive = gdrive_download(fid, cache / "gdrive" / split / fname)
        extract_zip(archive, work / "labels_raw")
        n_lbl = collect_labeltxts(work / "labels_raw", labels_dst)
        print(f"  labelTxt/{split}: {n_lbl} files")

    if "split_list" in cfg:
        fname, fid = cfg["split_list"]
        archive = gdrive_download(fid, cache / "gdrive" / split / fname)
        extract_zip(archive, work / "split_raw")
        src = find_split_list(work / "split_raw")
        if src:
            shutil.copy2(src, out_dir / f"{split}.txt")
            print(f"  copied {split}.txt")
        elif labels_dst.exists():
            write_split_list(labels_dst, out_dir / f"{split}.txt")

    if split == "test":
        fname, fid = cfg["test_info"]
        info_path = gdrive_download(fid, cache / "gdrive" / split / fname)
        write_test_list(info_path, images_dst, out_dir / "test.txt")


def verify(out_dir: Path, splits: list[str]) -> bool:
    ok = True
    for split in splits:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labelTxt" / split
        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        has_list = (out_dir / f"{split}.txt").exists()
        print(f"{split}: images={n_img}, labels={n_lbl}, {split}.txt={'yes' if has_list else 'no'}")
        if n_img == 0:
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DOTA v1.0 into data/raw/DOTA/")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", choices=["huggingface", "gdrive", "auto"], default="auto")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    cache_dir = out_dir / "_cache"
    for path in (out_dir, cache_dir):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        if split == "test" or args.source == "gdrive":
            process_gdrive_split(split, GDRIVE_SPLITS[split], out_dir, cache_dir)
        elif split in HF_ARCHIVES and args.source in ("huggingface", "auto"):
            process_hf_split(split, out_dir, cache_dir)
        else:
            process_gdrive_split(split, GDRIVE_SPLITS[split], out_dir, cache_dir)

    if not args.keep_cache and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("\nRemoved cache directory.")

    print("\nVerification:")
    if not verify(out_dir, args.splits):
        sys.exit(1)
    print(f"\nDOTA v1.0 ready at {out_dir}")


if __name__ == "__main__":
    main()
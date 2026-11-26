"""Prepare the full RDD split at a configurable maximum image edge (default 320 px)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from filelock import FileLock
from PIL import Image
import yaml

from core.data.rdd import NAMES, read
from scripts.prepare_rdd import fetch, labels, verify


def target_size(row, imgsz):
    """Expected longest-edge resize, retaining original smaller images."""
    width, height = row["width"], row["height"]
    longest = max(width, height)
    if longest <= imgsz:
        return width, height
    if width >= height:
        return imgsz, max(1, round(height * imgsz / width))
    return max(1, round(width * imgsz / height)), imgsz


def ready(image, label, expected_labels, expected_size):
    """Only reuse images saved at the requested resolution, never a 320 px copy."""
    if not image.is_file() or not label.is_file():
        return False
    if label.read_text() != expected_labels:
        return False
    try:
        with Image.open(image) as existing:
            width, height = expected_size
            correct_size = (
                max(existing.size) == max(expected_size)
                and abs(existing.width - width) <= 1
                and abs(existing.height - height) <= 1
            )
            if existing.format != "JPEG" or not correct_size:
                return False
            existing.verify()
        return True
    except (OSError, ValueError):
        return False


def prepare(rows, root, workers, imgsz=320, local=None):
    """Download or import verified originals, then create a resumable YOLO dataset."""
    if imgsz < 32 or imgsz % 32:
        raise ValueError("--imgsz must be a multiple of 32, at least 32")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "preparation.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved.get("imgsz") != imgsz:
            raise ValueError(
                "Output folder was prepared at another resolution; choose a new --out"
            )
    elif (root / "data.yaml").exists() and imgsz != 320:
        raise ValueError(
            "Existing dataset has no resolution marker; use a new --out for 640 px"
        )
    marker.write_text(json.dumps({"imgsz": imgsz}, indent=2) + "\n")

    for split in ("train", "val"):
        for folder in ("images", "labels"):
            (root / folder / split).mkdir(parents=True, exist_ok=True)

    def convert(row):
        name, split = row["file"], row["split"]
        image = root / "images" / split / name
        label = root / "labels" / split / (image.stem + ".txt")
        expected_labels = labels(row)
        try:
            if ready(image, label, expected_labels, target_size(row, imgsz)):
                return name, None

            if local is not None:
                data = (local / name).read_bytes()
                rgb = verify(data, row)
            else:
                cached = Path("data/processed/demo/images") / split / name
                if cached.is_file():
                    try:
                        data = cached.read_bytes()
                        rgb = verify(data, row)  # Reject downscaled cached images.
                    except (OSError, ValueError):
                        data = fetch(row["url"], [16_000_000])
                        rgb = verify(data, row)
                else:
                    data = fetch(row["url"], [16_000_000])
                    rgb = verify(data, row)

            rgb.thumbnail((imgsz, imgsz), Image.Resampling.LANCZOS)
            temp = image.with_suffix(".part")
            rgb.save(temp, "JPEG", quality=90)
            label.write_text(expected_labels)
            temp.replace(image)
            return name, None
        except (ValueError, OSError) as exc:
            return name, f"{type(exc).__name__}: {exc}"

    errors = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (name, error) in enumerate(pool.map(convert, rows), 1):
            if error:
                errors[name] = error
            if i % 1000 == 0:
                print(f"{i}/{len(rows)} processed, {len(errors)} errors", flush=True)
    (root / "errors.json").write_text(json.dumps(errors, indent=2))
    if errors:
        raise SystemExit("Incomplete full split. See errors.json and retry.")
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            dict(
                path=str(root.resolve()),
                train="images/train",
                val="images/val",
                names=NAMES,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-images",
        required=True,
        action="store_true",
        help="Explicitly prepare the entire RDD export",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Max image edge (e.g. 640)")
    parser.add_argument("--out", type=Path, help="Output folder; default full{imgsz}")
    parser.add_argument(
        "--local",
        type=Path,
        help="Verified original images with exact export filenames (optional)",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=8)
    args = parser.parse_args()
    root = args.out or Path(f"data/processed/full{args.imgsz}")
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".prepare.lock"), timeout=0):
        prepare(read(), root, args.workers, imgsz=args.imgsz, local=args.local)
    print(f"Prepared {root} at maximum edge {args.imgsz} px")


if __name__ == "__main__":
    main()

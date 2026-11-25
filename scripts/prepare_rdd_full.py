"""Explicitly prepare the entire official split for the 320-pixel CPU experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from filelock import FileLock
from PIL import Image
import yaml
from core.data.rdd import read, NAMES
from scripts.prepare_rdd import fetch, labels, verify

ROOT = Path("data/processed/full320")


def prepare(rows, root, workers):
    for split in ("train", "val"):
        for folder in ("images", "labels"):
            (root / folder / split).mkdir(parents=True, exist_ok=True)

    def convert(row):
        name, split = row["file"], row["split"]
        image = root / "images" / split / name
        label = root / "labels" / split / (image.stem + ".txt")
        expected = labels(row)
        try:
            if image.exists() and label.exists() and label.read_text() == expected:
                with Image.open(image) as existing:
                    if existing.format == "JPEG" and max(existing.size) <= 320:
                        existing.verify()
                        return name, None
            cached = Path("data/processed/demo/images") / split / name
            data = (
                cached.read_bytes()
                if cached.is_file()
                else fetch(row["url"], [16_000_000])
            )
            rgb = verify(data, row)
            rgb.thumbnail((320, 320), Image.Resampling.LANCZOS)
            temp = image.with_suffix(".part")
            rgb.save(temp, "JPEG", quality=90)
            label.write_text(expected)
            temp.replace(image)
            return name, None
        except (ValueError, OSError) as exc:
            return name, type(exc).__name__

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
        help="Download the entire RDD export",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=8)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(ROOT / ".prepare.lock"), timeout=0):
        prepare(read(), ROOT, args.workers)


if __name__ == "__main__":
    main()

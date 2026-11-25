"""Verify the prepared image/label pairs before handing them to Ultralytics."""

import hashlib
import json
from pathlib import Path

from PIL import Image
import yaml

from core.data.rdd import NAMES


def check_layout(config):
    spec = yaml.safe_load(config.read_text())
    if spec.get("names") != NAMES:
        raise ValueError("Expected the RoadFusion five-class mapping")
    root = Path(spec.get("path", config.parent)).resolve()
    report = root / "report.json"
    verified = json.loads(report.read_text())["sha256"] if report.is_file() else None
    selection = root / "selection.json"
    rows = (
        {r["file"]: r for r in json.loads(selection.read_text())["images"]}
        if selection.is_file()
        else None
    )
    train_hashes = set()
    for split in ("train", "val"):
        directory = root / spec[split]
        images = sorted(
            p
            for p in directory.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        label_dir = root / "labels" / split
        if not images or {p.stem for p in images} != {
            p.stem for p in label_dir.glob("*.txt")
        }:
            raise ValueError(f"Missing or unmatched {split} pairs")
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if verified is not None and verified.get(image.name) != digest:
                raise ValueError(f"Image checksum mismatch: {image.name}")
            if split == "train":
                train_hashes.add(digest)
            elif digest in train_hashes:
                raise ValueError("Identical image bytes across train and val")
            boxes = []
            for line in (label_dir / (image.stem + ".txt")).read_text().splitlines():
                cls, *coords = line.split()
                x, y, w, h = map(float, coords)
                if int(cls) not in range(len(NAMES)) or not (
                    0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                ):
                    raise ValueError(f"Invalid label: {image.name}")
                boxes.append([int(cls), x, y, w, h])
            with Image.open(image) as im:
                if rows is not None:
                    r = rows.get(image.name)
                    if (
                        not r
                        or r["split"] != split
                        or boxes != r["boxes"]
                        or im.size != (r["width"], r["height"])
                    ):
                        raise ValueError(f"Selection mismatch: {image.name}")
                im.verify()

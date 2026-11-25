"""Read the mixed-schema RDD export without exposing signed URLs in reports."""

import json
import random
import re
from collections import Counter
from pathlib import Path

SOURCE = Path("datasets/rdd2022.ndjson")
MAPPING = {0: 0, 2: 1, 3: 2, 4: 3, 6: 4}
NAMES = [
    "alligator crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "transverse crack",
]


def read(source=SOURCE):
    rows, metadata, seen = [], None, set()
    with open(source) as stream:
        for line, raw in enumerate(stream, 1):
            try:
                item = json.loads(raw)
                if item["type"] == "dataset":
                    if metadata is not None:
                        raise ValueError("multiple dataset records")
                    metadata = item
                    continue
                name = item["file"]
                if item["type"] != "image" or not re.fullmatch(
                    r"[\w-]+\.(?:jpg|jpeg|png)", name, re.I
                ):
                    raise ValueError("unsafe filename or record type")
                if Path(name).stem.casefold() in seen or item["split"] not in (
                    "train",
                    "val",
                ):
                    raise ValueError("duplicate label path or unknown split")
                seen.add(Path(name).stem.casefold())
                if any(
                    type(item[k]) is not int or item[k] <= 0
                    for k in ("width", "height")
                ):
                    raise ValueError("invalid dimensions")
                boxes = []
                for cls, x, y, w, h in item.get("annotations", {}).get("boxes", []):
                    if (
                        type(cls) is not int
                        or cls not in MAPPING
                        or not (
                            0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                        )
                    ):
                        raise ValueError("invalid box")
                    # Export rounding can put an edge 0.00005 outside the image.
                    if (
                        any(type(v) not in (int, float) for v in (x, y, w, h))
                        or max(w / 2 - x, h / 2 - y, x + w / 2 - 1, y + h / 2 - 1)
                        > 0.0001
                    ):
                        raise ValueError("box extends beyond image")
                    boxes.append([MAPPING[cls], x, y, w, h])
                rows.append(
                    dict(
                        file=name,
                        width=item["width"],
                        height=item["height"],
                        split=item["split"],
                        boxes=boxes,
                        url=item["url"],
                    )
                )
            except (KeyError, ValueError, TypeError, AttributeError):
                raise ValueError(f"Invalid RDD record on line {line}") from None
    if not metadata or any(
        metadata["class_names"].get(str(k)) != NAMES[v] for k, v in MAPPING.items()
    ):
        raise ValueError("missing or incompatible class definitions")
    return rows


def public(row):
    return {k: v for k, v in row.items() if k != "url"}


def summary(rows):
    hist = Counter(NAMES[b[0]] for r in rows for b in r["boxes"])
    return dict(
        images=len(rows),
        splits=dict(Counter(r["split"] for r in rows)),
        negatives=sum(not r["boxes"] for r in rows),
        boxes=sum(hist.values()),
        classes=dict(hist),
    )


def sample(rows, train, val, seed):
    rng = random.Random(seed)
    selected = []
    for split, count in (("train", train), ("val", val)):
        pool = [r for r in rows if r["split"] == split]
        rng.shuffle(pool)
        chosen, covered = {}, set()
        for row in pool:
            classes = {b[0] for b in row["boxes"]}
            if classes - covered and len(chosen) < count:
                chosen[row["file"]] = row
                covered.update(classes)
        for row in pool:
            if len(chosen) >= count:
                break
            chosen[row["file"]] = row
        selected.extend(chosen.values())
    return selected


def normalize(rows, directory):
    from datasets import load_dataset

    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, hf_split in (("train", "train"), ("val", "validation")):
        path = directory / f"{split}.jsonl"
        with path.open("w") as out:
            for row in rows:
                if row["split"] == split:
                    print(json.dumps(public(row)), file=out)
        files[hf_split] = str(path)
    return load_dataset("json", data_files=files)

"""Compare seeded weather conditions on one fixed, verified RDD validation subset."""

import argparse
import csv
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from PIL import Image

from core.data.rdd import NAMES, public, read
from core.detector import Detection, annotate, load, predict
from core.environment import CONDITIONS, transform
from core.metrics import class_metrics, latency


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select(root, limit, seed):
    selection = json.loads((root / "selection.json").read_text())
    verified = json.loads((root / "report.json").read_text())["sha256"]
    canonical = {r["file"]: public(r) for r in read() if r["split"] == "val"}
    rows = []
    seen = set()
    for row in selection["images"]:
        name = row["file"]
        if row["split"] != "val" or name not in verified:
            continue
        if row != canonical.get(name) or name in seen:
            raise ValueError(f"Invalid validation membership or annotations: {name}")
        seen.add(name)
        rows.append(row)
    random.Random(seed).shuffle(rows)
    if limit < 2 or limit > 256:
        raise ValueError("Choose 2–256 validation images")
    chosen = []
    for annotated in (True, False):
        first = next((r for r in rows if bool(r["boxes"]) == annotated), None)
        if first is None:
            raise ValueError(
                "Prepare a validation subset with both annotated and negative images"
            )
        chosen.append(first)
    reserved = {r["file"] for r in chosen}
    chosen.extend(r for r in rows if r["file"] not in reserved)
    manifest = []
    for row in chosen[:limit]:
        path = root / "images/val" / row["file"]
        if (
            path.resolve().parent != (root / "images/val").resolve()
            or digest(path) != verified[row["file"]]
        ):
            raise ValueError(f"Image identity mismatch: {row['file']}")
        with Image.open(path) as im:
            if (
                im.size != (row["width"], row["height"])
                or im.width * im.height > 16_000_000
            ):
                raise ValueError(f"Invalid dimensions: {row['file']}")
        manifest.append(dict(**row, sha256=verified[row["file"]]))
    return manifest


def annotations(row):
    w, h = row["width"], row["height"]
    return [
        Detection(
            c,
            1.0,
            ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h),
            "annotation",
        )
        for c, x, y, bw, bh in row["boxes"]
    ]


def summarize(per):
    labeled = [row for row in per if row["instances"]]
    total = sum(r["instances"] for r in labeled)
    fp, fn = (sum(r[k] for r in per) for k in ("fp", "fn"))
    tp = total - fn
    return dict(
        map50=mean(r["ap50"] for r in labeled) if labeled else None,
        map=mean(r["ap"] for r in labeled) if labeled else None,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / total if total else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        fp=fp,
        fn=fn,
    )


def differences(row, base):
    def delta(key):
        return row[key] - base[key] if None not in (row[key], base[key]) else None

    return dict(
        delta_map50=delta("map50"),
        delta_recall=delta("recall"),
        delta_fps=delta("processed_fps"),
        latency_change_pct=100 * delta("mean_ms") / base["mean_ms"]
        if base["mean_ms"]
        else None,
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def preview(image, boxes, path):
    rendered = Image.fromarray(annotate(image, boxes))
    rendered.thumbnail((720, 540))
    rendered.save(path)


def benchmark(a):
    if (
        not 0 <= a.strength <= 1
        or not 32 <= a.size <= 1280
        or a.size % 32
        or a.seed < 0
    ):
        raise ValueError(
            "Use strength 0–1, nonnegative seed and a multiple-of-32 size in [32,1280]"
        )
    if not a.weights.is_file():
        raise ValueError(f"Missing checkpoint: {a.weights}")
    rows = select(a.root, a.limit, a.seed)
    model = load(a.weights)
    conditions = list(dict.fromkeys(["daylight", *a.conditions]))
    truth = {r["file"]: annotations(r) for r in rows}
    run = a.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    figures = run / "figures"
    figures.mkdir()
    config = dict(
        seed=a.seed,
        strength=a.strength,
        conditions=conditions,
        sample_count=len(rows),
        root=str(a.root),
        selection_sha256=digest(a.root / "selection.json"),
        weights=str(a.weights),
        weights_sha256=digest(a.weights),
        device="cpu",
        platform=platform.platform(),
        machine=platform.machine(),
        evaluator_sha256=digest(Path(__file__)),
        metrics_sha256=digest(Path(__file__).resolve().parents[1] / "core/metrics.py"),
        python=platform.python_version(),
        versions={
            k: version(k)
            for k in ("torch", "ultralytics", "numpy", "opencv-python", "pillow")
        },
        size=a.size,
        score_floor=0.001,
        nms_iou=0.7,
        operating_score=0.25,
        matching_iou=0.5,
        class_names=NAMES,
        warmup_calls=1,
    )
    save(run / "config.json", config)
    save(run / "manifest.json", rows)
    predict(model, np.full((a.size, a.size, 3), 128, np.uint8), a.size, 0.001)
    summary, classes, samples, timings = [], [], [], []
    with (run / "predictions.jsonl").open("w") as stream:
        for condition in conditions:
            predictions, times, inference = {}, [], []
            for idx, row in enumerate(rows):
                name = row["file"]
                start = perf_counter()
                with Image.open(a.root / "images/val" / name) as im:
                    image = transform(
                        np.asarray(im.convert("RGB")), condition, a.strength, a.seed
                    )
                boxes, elapsed = predict(model, image, a.size, 0.001)
                times.append(perf_counter() - start)
                inference.append(elapsed)
                predictions[name] = boxes
                timings.append(
                    dict(
                        condition=condition,
                        file=name,
                        pipeline_s=times[-1],
                        detector_s=elapsed,
                    )
                )
                at_score = [b for b in boxes if b.score >= 0.25]
                sample = [
                    class_metrics({name: truth[name]}, {name: at_score}, c)
                    for c in range(len(NAMES))
                ]
                samples.append(
                    dict(
                        condition=condition,
                        file=name,
                        instances=len(truth[name]),
                        detections=len(at_score),
                        fp=sum(r["fp"] for r in sample),
                        fn=sum(r["fn"] for r in sample),
                    )
                )
                print(
                    json.dumps(
                        dict(
                            condition=condition,
                            file=name,
                            predictions=[b.record() for b in boxes],
                        )
                    ),
                    file=stream,
                )
                if idx < 2:
                    preview(image, [], figures / f"{idx}-{condition}-input.jpg")
                    preview(image, at_score, figures / f"{idx}-{condition}.jpg")
                    if condition == "daylight":
                        preview(image, truth[name], figures / f"{idx}-truth.jpg")
            per = [class_metrics(truth, predictions, c) for c in range(len(NAMES))]
            classes.extend(dict(condition=condition, **r) for r in per)
            metrics = dict(
                condition=condition,
                **summarize(per),
                mean_detector_ms=1000 * mean(inference),
                mean_ms=1000 * mean(times),
                **latency(times),
            )
            summary.append(
                metrics | differences(metrics, summary[0] if summary else metrics)
            )
    write_csv(run / "per_sample_metrics.csv", samples)
    write_csv(run / "timings.csv", timings)
    write_csv(run / "summary.csv", summary)
    save(
        run / "summary_metrics.json",
        dict(
            config=config,
            summary=summary,
            classes=classes,
            manifest=rows,
            preview_ids=[r["file"] for r in rows[:2]],
        ),
    )
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/processed/demo"))
    p.add_argument("--weights", type=Path, default=Path("models/rdd_best.pt"))
    p.add_argument("--out", type=Path, default=Path("runs/environment_benchmarks"))
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--strength", type=float, default=0.65)
    p.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS[:6]
    )
    try:
        print(benchmark(p.parse_args()))
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()

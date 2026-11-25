import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from datasets import Dataset
from PIL import Image

from core.data.rdd import NAMES
from core.detector import Detection, load
from core.environment import transform
from core.fusion import execute
from core.metrics import class_metrics, latency


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/rdd"))
    parser.add_argument("--weights", type=Path, default=Path("models/rdd_best.pt"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("runs/comparisons"))
    args = parser.parse_args()
    root = args.root
    verified = json.loads((root / "report.json").read_text())["sha256"]
    images = [
        r
        for r in json.loads((root / "selection.json").read_text())["images"]
        if r["split"] == "val" and r["file"] in verified
    ]
    if not images or args.limit < 1:
        parser.error("Need a positive count and prepared validation images")
    dataset = Dataset.from_list(images).shuffle(seed=args.seed)
    rows = list(dataset.select(range(min(args.limit, len(dataset)))))
    truth = {}
    manifest = []
    for r in rows:
        name = r["file"]
        path = root / "images" / "val" / name
        if digest(path) != verified[name]:
            raise ValueError(f"Image hash mismatch: {name}")
        truth[name] = []
        for cls, x, y, w, h in r["boxes"]:
            box = (
                (x - w / 2) * r["width"],
                (y - h / 2) * r["height"],
                (x + w / 2) * r["width"],
                (y + h / 2) * r["height"],
            )
            truth[name].append(Detection(int(cls), 1.0, box, "annotation"))
        manifest.append(
            dict(file=name, sha256=verified[name], instances=len(truth[name]))
        )
    model = load(args.weights)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out / run_id
    out.mkdir(parents=True)
    summary, classes, frames = [], [], []
    plans = [
        ("RGB", "daylight", "rgb_only", 640),
        ("Dark RGB", "darkness", "rgb_only", 640),
        ("Proxy", "daylight", "secondary_only", 640),
        ("Full fusion", "daylight", "late_fusion", 640),
        ("Reduced fusion", "daylight", "late_fusion", 320),
        ("Single best", "daylight", "best", 320),
        ("Dark fusion", "darkness", "late_fusion", 640),
    ]
    # Warm model once. Each wall-clock sample includes local image decode through final detections.
    execute(model, np.full((64, 64, 3), 128, np.uint8), "rgb_only")
    for pipeline, condition, mode, size in plans:
        predictions, times = {}, []
        for row in rows:
            name = row["file"]
            start = perf_counter()
            with Image.open(root / "images" / "val" / name) as img:
                rgb = transform(
                    np.asarray(img.convert("RGB")), condition, 0.65, args.seed
                )
            boxes, run = execute(model, rgb, mode, size=size, confidence=0.001)
            times.append(perf_counter() - start)
            predictions[name] = boxes
            frames.append(
                dict(
                    pipeline=pipeline,
                    file=name,
                    instances=len(truth[name]),
                    detections_at_025=sum(d.score >= 0.25 for d in boxes),
                    seconds=times[-1],
                    branches=run["branches"],
                )
            )
        per = [
            class_metrics(truth, predictions, cls) | dict(pipeline=pipeline)
            for cls in range(len(NAMES))
        ]
        classes.extend(per)
        labeled = [r for r in per if r["instances"]]
        summary.append(
            dict(
                pipeline=pipeline,
                condition=condition,
                mode=mode,
                size=size,
                map50=float(np.mean([r["ap50"] for r in labeled])),
                map=float(np.mean([r["ap"] for r in labeled])),
                fp=sum(r["fp"] for r in per),
                fn=sum(r["fn"] for r in per),
                **latency(times),
            )
        )
        print(pipeline, summary[-1], flush=True)
    cpu = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                l.split(":", 1)[1].strip()
                for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")
            ),
            cpu,
        )
    provenance = dict(
        evaluator_sha256=digest(Path(__file__)),
        metrics_sha256=digest(Path("core/metrics.py")),
        seed=args.seed,
        weights_sha256=digest(args.weights),
        manifest=manifest,
        device="cpu",
        cpu=cpu,
        environment={
            k: version(k)
            for k in ("ultralytics", "torch", "opencv-python", "datasets", "numpy")
        },
        runtime=sys.version,
        run_id=run_id,
        selection_sha256=digest(root / "selection.json"),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        score_floor=0.001,
        operating_score=0.25,
        synthetic_strength=0.65,
    )
    report = dict(
        summary=summary, classes=classes, frames=frames, provenance=provenance
    )
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for key in ("summary", "classes", "frames"):
        pd.DataFrame(report[key]).to_csv(out / (key + ".csv"), index=False)
    print(out)


if __name__ == "__main__":
    main()

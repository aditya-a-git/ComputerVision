"""Train the full-data CPU experiment, validating coverage before optimization."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from ultralytics import YOLO
from core.data.rdd import read
from core.data.yolo import check_layout
from core.detector import load
from scripts.prepare_rdd import labels
from scripts.prepare_rdd_full import ROOT


def verify_full(root):
    rows = read()
    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 30707, "val": 7676}:
        raise ValueError(f"Unexpected export split sizes: {counts}")
    for split in counts:
        actual = {p.name for p in (root / "images" / split).glob("*") if p.is_file()}
        expected = {row["file"] for row in rows if row["split"] == split}
        if actual != expected:
            raise ValueError(f"Incomplete or unexpected {split} images")
    for row in rows:
        name = Path(row["file"]).stem + ".txt"
        if (root / "labels" / row["split"] / name).read_text() != labels(row):
            raise ValueError(f"Annotation mismatch: {name}")
    check_layout(root / "data.yaml")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/full320"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = ROOT.resolve()
    counts = verify_full(root)
    out = args.out.resolve()
    weights = out / "rdd" / "weights"
    if args.resume:
        model = YOLO(weights / "last.pt")
        options = {"resume": True}
    else:
        if (out / "rdd").exists():
            parser.error("Run directory exists. Supply --resume or choose a new --out.")
        model = YOLO("yolo11n.pt")
        options = dict(
            data=str(root / "data.yaml"),
            epochs=8,
            batch=16,
            imgsz=320,
            device="cpu",
            workers=0,
            seed=42,
            amp=False,
            project=str(out),
            name="rdd",
            plots=False,
        )

    def configure(trainer):
        torch.set_num_threads(2)
        if (
            len(trainer.train_loader.dataset) != counts["train"]
            or trainer.train_loader.drop_last
        ):
            raise ValueError("Training loader cannot cover the complete split")

    model.add_callback("on_train_start", configure)
    model.train(**options)
    trained = load(str(model.trainer.best))
    metrics = trained.val(
        data=str(root / "data.yaml"),
        device="cpu",
        imgsz=320,
        batch=16,
        workers=0,
        plots=False,
        project=str(out),
        name="final-validation",
    )
    result = dict(
        train=counts["train"],
        val=counts["val"],
        epochs=8,
        size=320,
        seed=42,
        weights_sha256=hashlib.sha256(
            Path(model.trainer.best).read_bytes()
        ).hexdigest(),
        map50=float(metrics.box.map50),
        map5095=float(metrics.box.map),
        classes=metrics.summary(),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(model.trainer.best)


if __name__ == "__main__":
    main()

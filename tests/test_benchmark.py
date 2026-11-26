import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from core.detector import Detection
from scripts import environment_benchmark as bench


def test_paired_workflow(tmp_path, monkeypatch):
    rows, hashes = [], {}
    folder = tmp_path / "images/val"
    folder.mkdir(parents=True)
    for i in range(4):
        im = np.random.default_rng(i).integers(0, 255, (24, 24, 3), dtype=np.uint8)
        name = f"{i}.png"
        Image.fromarray(im).save(folder / name)
        hashes[name] = bench.digest(folder / name)
        rows.append(
            dict(
                file=name,
                width=24,
                height=24,
                boxes=[[0, 0.5, 0.5, 0.5, 0.5]] if i % 2 else [],
                split="val",
            )
        )
    (tmp_path / "selection.json").write_text(json.dumps(dict(images=rows)))
    (tmp_path / "report.json").write_text(json.dumps(dict(sha256=hashes)))
    monkeypatch.setattr(bench, "read", lambda: rows)
    assert bench.select(tmp_path, 2, 42) == bench.select(tmp_path, 2, 42)
    monkeypatch.setattr(bench, "load", lambda _: object())
    inputs = []

    def prediction(model, image, *args):
        inputs.append(image.copy())
        return [Detection(0, 0.8, (6, 6, 18, 18))], 0.01

    monkeypatch.setattr(bench, "predict", prediction)
    args = SimpleNamespace(
        root=tmp_path,
        weights=tmp_path / "report.json",
        limit=2,
        seed=42,
        out=tmp_path / "runs",
        size=32,
        strength=0.65,
        conditions=["noise"],
    )
    run = bench.benchmark(args)
    manifest = json.loads((run / "manifest.json").read_text())
    ids = [r["file"] for r in manifest]
    assert len(ids) == 2 and manifest[0]["boxes"] and not manifest[1]["boxes"]
    predictions = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text().splitlines()
    ]
    assert [r["file"] for r in predictions] == ids * 2
    assert not np.array_equal(inputs[1], inputs[3])
    report = json.loads((run / "summary_metrics.json").read_text())
    base, noise = report["summary"]
    assert base["map"] == 1 and base["recall"] == 1 and base["precision"] == 0.5
    assert base["f1"] == pytest.approx(2 / 3) and base["fp"] == 1
    assert noise["delta_map50"] == 0 and report["config"]["sample_count"] == 2
    for filename, count in [
        ("timings.csv", 4),
        ("summary.csv", 2),
        ("per_sample_metrics.csv", 4),
    ]:
        with (run / filename).open() as stream:
            assert len(list(csv.DictReader(stream))) == count
    (folder / ids[0]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="identity"):
        bench.select(tmp_path, 2, 42)


def test_empty_and_deltas():
    empty = bench.summarize([])
    assert empty["map50"] is None and empty["recall"] is None
    base = dict(map50=0.5, recall=0.75, processed_fps=10, mean_ms=100)
    row = dict(map50=0.25, recall=0.5, processed_fps=5, mean_ms=200)
    assert bench.differences(row, base) == dict(
        delta_map50=-0.25, delta_recall=-0.25, delta_fps=-5, latency_change_pct=100
    )

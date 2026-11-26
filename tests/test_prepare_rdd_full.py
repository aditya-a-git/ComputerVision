import json

from PIL import Image
import pytest

from core.data.rdd import NAMES
from scripts.prepare_rdd_full import prepare, target_size


def test_640_originals_and_resume(tmp_path, monkeypatch):
    originals = tmp_path / "originals"
    originals.mkdir()
    rows = []
    for split, size, color in (
        ("train", (1280, 640), (50, 70, 90)),
        ("val", (600, 900), (90, 20, 40)),
    ):
        filename = f"{split}.jpg"
        Image.new("RGB", size, color).save(originals / filename)
        rows.append(
            dict(
                file=filename,
                split=split,
                width=size[0],
                height=size[1],
                boxes=[] if split == "val" else [[3, 0.5, 0.5, 0.2, 0.2]],
            )
        )
    root = tmp_path / "full640"
    prepare(rows, root, workers=1, imgsz=640, local=originals)
    assert Image.open(root / "images/train/train.jpg").size == (640, 320)
    assert Image.open(root / "images/val/val.jpg").size == (427, 640)
    assert (root / "labels/train/train.txt").read_text() == "3 0.5 0.5 0.2 0.2\n"
    assert (root / "labels/val/val.txt").read_text() == ""
    assert json.loads((root / "preparation.json").read_text())["imgsz"] == 640
    assert json.loads((root / "errors.json").read_text()) == {}
    assert NAMES[3] == "pothole"
    originals.rename(tmp_path / "moved")
    prepare(rows, root, workers=1, imgsz=640, local=originals)  # Must reuse outputs.
    with pytest.raises(ValueError, match="another resolution"):
        prepare(rows, root, workers=1, imgsz=320, local=originals)


def test_320_cache_is_not_used_for_640(tmp_path, monkeypatch):
    import io

    original = Image.new("RGB", (1280, 640), (32, 65, 99))
    output = io.BytesIO()
    original.save(output, format="JPEG")
    raw = output.getvalue()
    row = dict(
        file="test.jpg", split="train", width=1280, height=640,
        boxes=[], url="https://cdn.ul.run/test.jpg",
    )
    monkeypatch.chdir(tmp_path)
    cached = tmp_path / "data/processed/demo/images/train/test.jpg"
    cached.parent.mkdir(parents=True)
    Image.new("RGB", (320, 160)).save(cached)
    requested = []

    def download(url, budget):
        requested.append(url)
        return raw

    monkeypatch.setattr("scripts.prepare_rdd_full.fetch", download)
    root = tmp_path / "data/processed/full640"
    prepare([row], root, workers=1, imgsz=640)
    assert requested == [row["url"]]
    assert Image.open(root / "images/train/test.jpg").size == (640, 320)


def test_tiny_source_is_never_upscaled():
    assert target_size(dict(width=200, height=100), 640) == (200, 100)
    assert target_size(dict(width=1200, height=600), 640) == (640, 320)

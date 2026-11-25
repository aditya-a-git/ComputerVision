import json
from types import SimpleNamespace

from PIL import Image
import pytest
from scripts.prepare_rdd import prepare, fetch


def test_local_layout_and_resume(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    root = tmp_path / "processed"
    root.mkdir()
    rows = []
    for split, boxes in (("train", [[3, 0.5, 0.5, 0.4, 0.2]]), ("val", [])):
        name = split + ".png"
        Image.new("RGB", (30, 24)).save(source / name)
        rows.append(dict(file=name, width=30, height=24, split=split, boxes=boxes))
    args = SimpleNamespace(seed=42, max_download_mb=0, local=source)
    prepare(root, rows, rows, None, args)
    assert (root / "labels/val/val.txt").read_text() == ""
    assert (root / "labels/train/train.txt").read_text() == "3 0.5 0.5 0.4 0.2\n"
    (source / "val.png").unlink()
    prepare(root, rows, rows, None, args)
    assert not json.loads((root / "report.json").read_text())["skipped"]
    with pytest.raises(SystemExit):
        prepare(root, rows[:1], rows[:1], None, args)


def test_http_denied(monkeypatch):
    from unittest.mock import MagicMock

    request = MagicMock()
    request.return_value.__enter__.return_value.status_code = 403
    monkeypatch.setattr("scripts.prepare_rdd.requests.get", request)
    with pytest.raises(ValueError, match="HTTP 403"):
        fetch("https://cdn.ul.run/image.jpg", [100])

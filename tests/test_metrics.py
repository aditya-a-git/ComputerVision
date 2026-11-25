import pytest
from core.detector import Detection
from core.metrics import class_metrics


def test_matching_and_empty_images():
    box = Detection(0, 0.9, (0, 0, 10, 10))
    truth = {"a": [box], "b": []}
    m = class_metrics(truth, {"a": [box, box], "b": [box]}, 0)
    assert m["fp"] == 2 and m["recall"] == 1 and m["precision"] == pytest.approx(1 / 3)
    assert m["ap"] == 1
    assert class_metrics(truth, {}, 0)["ap"] == 0
    assert class_metrics(truth, {}, 1)["ap"] is None

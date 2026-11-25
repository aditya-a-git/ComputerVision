import pytest
import numpy as np
from core import fusion
from core.detector import Detection

A = Detection(0, 0.8, (1, 1, 11, 11))
B = Detection(0, 0.6, (2, 2, 12, 12), "proxy")


def test_pair_and_disagreement():
    assert fusion.iou(A.box, A.box) == 1
    assert fusion.iou(A.box, (20, 20, 30, 30)) == 0
    merged = fusion.merge([A], [B], (1, 1))
    assert len(merged) == 1 and merged[0].score == pytest.approx(0.7)
    assert 1 < merged[0].box[0] < 2
    other = Detection(1, 0.6, B.box, "proxy")
    assert len(fusion.merge([A], [other], (1, 1))) == 2
    assert fusion.merge([A], [B], (0, 0)) == []
    assert fusion.merge([A], [B], (0, 1)) == [B]


def test_modes(monkeypatch):
    image = np.random.default_rng(5).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(fusion, "predict", lambda *args, **kw: ([A], 0.1))
    for health, count in [((1, 1), 2), ((1, 0), 1), ((0, 1), 1), ((0, 0), 0)]:
        _, run = fusion.execute(None, image, health=health)
        assert len(run["branches"]) == count
    monkeypatch.setattr(fusion, "proxy", lambda x: x[:1])
    with pytest.raises(ValueError):
        fusion.execute(None, image)

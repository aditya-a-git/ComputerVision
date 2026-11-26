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


def test_execution_provenance(monkeypatch):
    image = np.full((32, 32, 3), 128, np.uint8)
    extra = Detection(1, 0.35, (12, 12, 22, 22))
    calls = []

    def detect(model, image, size, confidence, source):
        calls.append(source)
        return ([A, extra] if source == "rgb" else [B]), 0.02

    monkeypatch.setattr(fusion, "predict", detect)
    for health, expected in [
        ((1, 1), ["rgb", "proxy"]),
        ((0, 1), ["proxy"]),
        ((1, 0), ["rgb"]),
        ((0, 0), []),
        ((1, 1), ["rgb", "proxy"]),
    ]:
        calls.clear()
        boxes, run = fusion.execute(None, image, health=health, explain=True)
        assert calls == expected == run["branches"]
        if len(calls) == 2:
            match, unmatched = run["trace"]
            assert (
                match["iou"] == fusion.iou(A.box, B.box)
                and match["rgb_index"] == match["proxy_index"] == 0
            )
            assert match["score"] == boxes[0].score and match["rgb_weight"] + match[
                "proxy_weight"
            ] == pytest.approx(1)
            assert (
                unmatched["decision"] == "below final confidence"
                and unmatched["rgb_index"] == 1
            )
            assert unmatched["proxy_index"] is None and len(boxes) == 1
        elif calls:
            assert all(
                r["kind"] == "single branch" and r["decision"] == "retained"
                for r in run["trace"]
            )
        else:
            assert not boxes and not run["trace"]

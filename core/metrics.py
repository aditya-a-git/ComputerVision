"""Confidence-ranked, same-class, per-image greedy matching and 101-point AP."""

import numpy as np
from core.data.rdd import NAMES
from core.fusion import iou


def class_metrics(truth, predictions, cls, operating=0.25):
    actual = {
        key: [d.box for d in boxes if d.cls == cls] for key, boxes in truth.items()
    }
    total = sum(map(len, actual.values()))
    ranked = sorted(
        (
            (d.score, key, d.box)
            for key, boxes in predictions.items()
            for d in boxes
            if d.cls == cls
        ),
        key=lambda item: -item[0],
    )
    overlaps = [[iou(box, b) for b in actual[key]] for _, key, box in ranked]
    aps = []
    for threshold in np.linspace(0.5, 0.95, 10):
        used = set()
        hits = []
        for (_, key, _), values in zip(ranked, overlaps):
            best = max(
                (i for i in range(len(values)) if (key, i) not in used),
                key=values.__getitem__,
                default=None,
            )
            match = best is not None and values[best] >= threshold
            if match:
                used.add((key, best))
            hits.append(match)
        cumulative = np.cumsum(hits)
        precision = cumulative / np.arange(1, len(hits) + 1)
        recall = cumulative / max(1, total)
        aps.append(
            float(
                np.mean(
                    [
                        precision[recall >= r].max(initial=0)
                        for r in np.linspace(0, 1, 101)
                    ]
                )
            )
        )
        if threshold == 0.5:
            count = sum(score >= operating for score, _, _ in ranked)
            tp = int(sum(hits[:count]))
            fp = count - tp
    p = tp / max(1, tp + fp)
    r = tp / max(1, total)
    return dict(
        name=NAMES[cls],
        instances=total,
        ap50=aps[0] if total else None,
        ap=float(np.mean(aps)) if total else None,
        precision=p,
        recall=r,
        f1=2 * p * r / (p + r) if p + r else 0,
        fp=fp,
        fn=total - tp,
    )


def latency(seconds):
    return dict(
        p50_ms=float(np.percentile(seconds, 50) * 1000),
        p95_ms=float(np.percentile(seconds, 95) * 1000),
        processed_fps=len(seconds) / sum(seconds),
    )

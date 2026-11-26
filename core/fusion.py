from dataclasses import replace
from time import perf_counter
from core.detector import Detection, predict
from core.reliability import quality
from core.sensor_simulator import proxy


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def merge(left, right, weights, threshold=0.5, *, trace=None):
    """Greedy one-to-one highest-IoU pairing. Penalize solitary evidence."""
    a, b = weights
    if not 0 < threshold <= 1 or min(weights) < 0:
        raise ValueError("Invalid fusion settings")
    if min(weights) == 0:
        return list(left if a else right if b else [])
    candidates = sorted(
        (
            (-iou(d.box, e.box), i, j)
            for i, d in enumerate(left)
            for j, e in enumerate(right)
            if d.cls == e.cls
        )
    )
    used_left, used_right = set(), set()
    out = []
    for neg_overlap, i, j in candidates:
        if -neg_overlap < threshold:
            break
        if i in used_left or j in used_right:
            continue
        d, e = left[i], right[j]
        w, v = a * d.score, b * e.score
        if w + v <= 0:
            continue
        box = tuple((w * x + v * y) / (w + v) for x, y in zip(d.box, e.box))
        out.append(Detection(d.cls, (w + v) / (a + b), box, "matched RGB + proxy"))
        if trace is not None:
            trace.append(
                decision(
                    out[-1],
                    i,
                    j,
                    d,
                    e,
                    -neg_overlap,
                    "matched",
                    "Highest available same-class IoU meets threshold",
                )
            )
        used_left.add(i)
        used_right.add(j)
    for name, items, used, weight in (
        ("rgb", left, used_left, a),
        ("proxy", right, used_right, b),
    ):
        for index, item in enumerate(items):
            if index in used:
                continue
            output = replace(
                item,
                score=item.score * weight / (a + b),
                source="unmatched " + item.source,
            )
            out.append(output)
            if trace is not None:
                opposite = right if name == "rgb" else left
                overlap = max(
                    (
                        iou(item.box, other.box)
                        for other in opposite
                        if other.cls == item.cls
                    ),
                    default=None,
                )
                trace.append(
                    decision(
                        output,
                        index if name == "rgb" else None,
                        index if name == "proxy" else None,
                        item if name == "rgb" else None,
                        item if name == "proxy" else None,
                        overlap,
                        "unmatched",
                        "No remaining eligible partner; scaled by view weight",
                    )
                )
    return sorted(out, key=lambda d: -d.score)


def decision(output, i, j, rgb, secondary, overlap, kind, reason):
    return dict(
        class_name=output.record()["class_name"],
        rgb_index=i,
        proxy_index=j,
        rgb_score=rgb.score if rgb else None,
        proxy_score=secondary.score if secondary else None,
        iou=overlap,
        score=output.score,
        box=output.box,
        kind=kind,
        reason=reason,
    )


def execute(
    model,
    rgb,
    mode="late_fusion",
    health=(1.0, 1.0),
    size=640,
    confidence=0.25,
    threshold=0.5,
    *,
    explain=False,
):
    start = perf_counter()
    if mode not in ("rgb_only", "secondary_only", "late_fusion", "best", "unavailable"):
        raise ValueError("Unknown mode")
    views = {}
    estimates = {}
    for i, name in enumerate(("rgb", "proxy")):
        if (
            mode == "unavailable"
            or health[i] <= 0
            or (mode == "rgb_only" and i == 1)
            or (mode == "secondary_only" and i == 0)
        ):
            continue
        image = rgb if i == 0 else proxy(rgb)
        if image.shape != rgb.shape:
            raise ValueError("Unaligned second view")
        estimates[name] = quality(image)
        weight = estimates[name] * health[i]
        if weight > 0:
            views[name] = (image, weight)
    if mode == "best" and views:
        name = max(views, key=lambda n: views[n][1])
        views = {name: views[name]}
    predictions, branch_seconds = {}, {}
    for name, (image, _) in views.items():
        predictions[name], branch_seconds[name] = predict(
            model, image, size, confidence, source=name
        )
    trace = [] if explain else None
    weights = {name: w for name, (_, w) in views.items()}
    if len(views) == 2:
        boxes = merge(
            predictions["rgb"],
            predictions["proxy"],
            (weights["rgb"], weights["proxy"]),
            threshold,
            trace=trace,
        )
    else:
        boxes = next(iter(predictions.values()), [])
        if explain and views:
            name = next(iter(views))
            for index, item in enumerate(boxes):
                is_rgb = name == "rgb"
                trace.append(
                    decision(
                        item,
                        index if is_rgb else None,
                        None if is_rgb else index,
                        item if is_rgb else None,
                        None if is_rgb else item,
                        None,
                        "single branch",
                        "Sole executed branch retains its score",
                    )
                )
    extra = {}
    if explain:
        shares = {name: w / sum(weights.values()) for name, w in weights.items()}
        for row in trace:
            row.update(
                rgb_weight=shares.get("rgb", 0),
                proxy_weight=shares.get("proxy", 0),
                decision="retained"
                if row["score"] >= confidence
                else "below final confidence",
            )
        extra = dict(
            trace=trace,
            predictions={k: [d.record() for d in v] for k, v in predictions.items()},
            quality=estimates,
            shares=shares,
            health=list(health),
            threshold=threshold,
            confidence=confidence,
        )
    return [d for d in boxes if d.score >= confidence], dict(
        branches=list(views),
        branch_seconds=branch_seconds,
        **extra,
        weights=weights,
        seconds=perf_counter() - start,
        size=size,
        mode=mode if views else "unavailable",
    )

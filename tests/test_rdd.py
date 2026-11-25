import json
import pytest
from core.data.rdd import SOURCE, read, sample, summary
from scripts.prepare_rdd import fetch, labels


def test_real_inventory():
    rows = read()
    s = summary(rows)
    assert (s["images"], s["negatives"], s["boxes"], s["splits"]) == (
        38383,
        12658,
        61075,
        {"train": 30707, "val": 7676},
    )
    selected = sample(rows, 25, 8, 4)
    assert selected == sample(rows, 25, 8, 4)
    assert sum(r["split"] == "val" for r in selected) == 8
    assert {b[0] for r in selected for b in r["boxes"]} == set(range(5))
    assert labels(rows[1]) == ""
    assert labels(rows[0]).startswith("2 ")


@pytest.mark.parametrize(
    "change",
    [
        {"file": "../escape.jpg"},
        {"annotations": {"boxes": [[1, 0.1, 0.1, 0.1, 0.1]]}},
        {"annotations": {"boxes": [[0, 0.2, 0.2, 0, 0.2]]}},
    ],
)
def test_invalid(tmp_path, change):
    with SOURCE.open() as source:
        meta, image = source.readline(), json.loads(source.readline())
    path = tmp_path / "invalid"
    path.write_text(meta + json.dumps(image | change))
    with pytest.raises(ValueError):
        read(path)


def test_expired():
    with pytest.raises(ValueError, match="expired"):
        fetch("https://cdn.ul.run/a?Expires=1", [100])

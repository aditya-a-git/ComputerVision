import hashlib
import json
import pytest
from core.data.local import original_image


def test_crosswalk_reconciliation(tmp_path):
    row = {"file": "export.jpg", "width": 10, "height": 5, "boxes": [], "split": "val"}
    data = b"original image identity"
    (tmp_path / "original.jpg").write_bytes(data)
    entry = dict(
        path="original.jpg",
        width=10,
        height=5,
        sha256=hashlib.sha256(data).hexdigest(),
        labels_sha256=hashlib.sha256(
            json.dumps(["val", []], separators=(",", ":")).encode()
        ).hexdigest(),
    )
    assert original_image(tmp_path, row, {"export.jpg": entry}) == data
    with pytest.raises(ValueError):
        original_image(tmp_path, row, {"export.jpg": entry | {"sha256": "wrong"}})
    with pytest.raises(ValueError):
        original_image(tmp_path, row | {"split": "train"}, {"export.jpg": entry})

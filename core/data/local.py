"""Import renamed official images only with an independently verified crosswalk."""

import hashlib
import json


def original_image(directory, row, crosswalk):
    entry = crosswalk.get(row["file"])
    if not entry:
        raise ValueError("No verified identity crosswalk entry")
    root = directory.resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path escapes source directory")
    if entry["width"] != row["width"] or entry["height"] != row["height"]:
        raise ValueError("Crosswalk dimensions mismatch")
    # Bind the verified image identity to this exact export annotation and split.
    label_digest = hashlib.sha256(
        json.dumps([row["split"], row["boxes"]], separators=(",", ":")).encode()
    ).hexdigest()
    if label_digest != entry["labels_sha256"]:
        raise ValueError("Crosswalk annotation identity mismatch")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("Crosswalk image identity mismatch")
    return data

import csv
import json
from pathlib import Path
from core.data.rdd import MAPPING, NAMES, normalize, read, summary


def main():
    rows = read()
    out = Path("data/metadata")
    dataset = normalize(rows, out)
    counts = summary(rows)
    expected = dict(images=38383, negatives=12658, boxes=61075, splits={"train":30707,"val":7676})
    counts["deviations"] = {k: [v, counts[k]] for k,v in expected.items() if counts[k] != v}
    for name, content in (("summary", counts), ("mapping", {"original_to_model": MAPPING, "names": NAMES})):
        (out / f"{name}.json").write_text(json.dumps(content, indent=2))
    with (out / "classes.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "boxes"])
        writer.writerows(counts["classes"].items())
    print(dataset, counts)


if __name__ == "__main__":
    main()

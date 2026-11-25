import argparse
import json
from pathlib import Path
from core.detector import load
from core.data.yolo import check_layout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/rdd_best.pt")
    p.add_argument("--data", type=Path, default=Path("data/processed/rdd/data.yaml"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()
    check_layout(a.data)
    model = load(a.weights)
    metrics = model.val(
        data=str(a.data.resolve()),
        split="val",
        device=a.device,
        imgsz=a.imgsz,
        workers=0,
        project="runs/evaluation",
    )
    result = dict(
        config=vars(a),
        map50=metrics.box.map50,
        map=metrics.box.map,
        classes=metrics.summary(),
    )
    (Path(metrics.save_dir) / "evaluation.json").write_text(
        json.dumps(result, default=str, indent=2)
    )


if __name__ == "__main__":
    main()

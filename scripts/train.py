import argparse
from pathlib import Path
from core.data.yolo import check_layout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed/rdd/data.yaml"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/train")
    p.add_argument("--base", default="yolo11n.pt")
    a = p.parse_args()
    check_layout(a.data)
    from ultralytics import YOLO

    model = YOLO(a.base)
    model.train(
        data=str(a.data.resolve()),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        seed=a.seed,
        project=str(Path(a.out).resolve()),
        name="rdd",
        workers=0,
        amp=False,
    )
    print("Trained checkpoint:", model.trainer.best)


if __name__ == "__main__":
    main()

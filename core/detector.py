from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
from core.data.rdd import NAMES


@dataclass(frozen=True)
class Detection:
    cls: int
    score: float
    box: tuple
    source: str = "rgb"

    def record(self):
        return asdict(self) | {"class_name": NAMES[self.cls]}


def load(path="models/rdd_best.pt"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Model not trained / weights missing. See README training instructions.")
    from ultralytics import YOLO
    model = YOLO(str(path))
    if model.task != "detect" or model.names != dict(enumerate(NAMES)):
        raise ValueError("Checkpoint class mapping differs from the RDD training configuration")
    return model


def predict(model, rgb, size=640, confidence=.25, device="cpu", source="rgb"):
    start = perf_counter()
    result = model.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR), imgsz=size, conf=confidence,
                           device=device, verbose=False)[0]
    detections = [Detection(int(cls),float(score),tuple(coords),source)
                  for *coords,score,cls in result.boxes.data.cpu().tolist()]
    return detections, perf_counter()-start


def annotate(rgb, detections):
    image = rgb.copy()
    for d in detections:
        x,y,x2,y2 = map(round,d.box)
        cv2.rectangle(image,(x,y),(x2,y2),(255,180,32),2)
        cv2.putText(image,f"{NAMES[d.cls]} {d.score:.2f}",(x,max(16,y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(255,180,32),1,cv2.LINE_AA)
    return image


def video_frames(path, limit=90):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Could not decode this video")
        for index in range(min(180,max(0,limit))):
            ok, frame = capture.read()
            if not ok:
                break
            yield index, cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    finally:
        capture.release()

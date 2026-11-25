# Recorded results

A small CPU YOLO11n experiment completed with 100 training images and 24 validation images. The 24 frames contained 37 annotated defects and nine negative images. The native Ultralytics evaluation at 320 pixels returned mAP@0.50 **0.005129** and mAP@0.50:0.95 **0.000935** (0–1 scale). Pothole AP was zero on three pothole instances. This checkpoint demonstrates inference and has poor detection quality.

At the dashboard cutoff of 0.25, the 640-pixel RGB pipeline missed all 37 annotated boxes and produced 18 false positives. For example, Japan_003139 had eight annotations and no detections. India_004905 had zero annotations and one detection. Lowering the cutoff increases the number of boxes displayed and does not establish accuracy.

| Execution on Intel Xeon Platinum 8573C CPU | p50 ms | p95 ms | Processed FPS |
|---|---:|---:|---:|
| RGB only, 640 | 111.3 | 298.2 | 6.28 |
| Both views, 640 | 437.9 | 807.6 | 2.05 |
| Both views, 320 | 114.3 | 410.7 | 5.58 |
| Best single view, 320 | 32.1 | 290.0 | 15.21 |

These are observations on the same 24 frames. Sampling variation, shared-host load, small classes and a weak detector limit conclusions. No general fusion improvement has been established. The one-batch smoke test is excluded. CARD frame retrieval required authentication; real CARD projection and cross-dataset accuracy remain unmeasured. Hardware telemetry, IR and vehicle tests were outside scope. [Paired results and per-frame provenance](measurements/tiny-rdd/report.json) · [Ultralytics class results](measurements/tiny-rdd/ultralytics.json)

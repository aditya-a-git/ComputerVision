# Implementation record
Baseline 3122e28ba48d81ba4feb60944e6f3aa2d327ae1c (see history for full stage hashes).
Stage 0 introduces a credential-free homepage. Smoke test: python -m pytest -q. Data, model and CARD access follow in subsequent stages.
Stage 0 9fbc573: server health returned ok. Initial AppTest resolved app.py relative to tests and failed. Corrected its absolute path in the following fix commit.
Stage 1: inspected all 38,383 images, HF train/validation partitions, five classes and 61,075 boxes. No expected count deviations. Dry-run selected 8 images. CLI uses serial requests, an aggregate byte budget, validated exact-name local imports and a signed-URL-free manifest. Run python -m scripts.inspect_rdd and python -m scripts.prepare_rdd --dry-run.
Stage 1 commit 58510bb omitted core/data because of an overly broad ignore pattern. The immediate follow-up tracks the ingestion module and scopes the generated-data exclusion to the repository root.
Stage 2: training, reserved validation, trusted checkpoint loading, bounded video reader, image annotation and Colab training. Run python -m scripts.train --help and python -m scripts.evaluate --help. Prepared all 12 smoke images. The one-epoch smoke run is isolated in runs/smoke.
Stage 3: seeded photometric transformations and interactive side-by-side image upload. Run python -m pytest tests/test_environment.py.
Stage 4: measured two-pass RGB ensemble with same-class matching, weighted coordinates, decision provenance and health-aware fallback. Both real image acquisition runs completed (12 smoke, 124 demonstration pairs). One CPU smoke epoch completed. Run python -m pytest tests/test_fusion.py.
Stage 5: public HF metadata confirmed gated CC BY 4.0 dataset and matching per-sequence paths. SDK camera_infos, distortion, labels and pts_cam formats inspected. Explorer includes metric projection on undistorted cam_1. Actual gated pixels unverified. Run python -m scripts.inspect_card.
Stage 6: configurable temperature state machine, dwell and hysteresis, immediate missing-stream fallback. Execution parameters map to the tested fusion pipeline. Tests cover escalation and recovery. Eight-epoch CPU demonstration completed on 100 train and 24 validation images at 320 pixels.
Stage 7: eight local dashboard views with bounded video replay, class filtering, downloadable outputs, health and temperature controls and measured run history. Run streamlit run app.py. AppTest traverses each view with missing weights.
Stage 8 evaluation implementation: 24 offline tests passed on fresh Python 3.11 CPU environment. Trained-model AppTest exercised all eight screens and a four-frame MP4 including pause, resume and export. The next commit pins the evaluator before the final validation report is captured.

## Verified delivery

The source files are unchanged. Local and remote main were compared after each push. Stage 0's test path and Stage 1's overly broad ignore rule were corrected by follow-up commits without rewriting history. Stage 8 has a separate evaluator commit so the saved results identify the exact implementation. The final documentation commit can be resolved with the command below, avoiding a self-referential commit ID.

| Stage | Commit(s) | Test suite at stage completion |
|---|---|---:|
| 0 – scaffold and smoke-test correction | 9fbc573, 2e0798b | 1 passing after correction |
| 1 – inspection and preparation | 58510bb, 25094e4 | 6 |
| 2 – training and inference | e833d59 | 7 |
| 3 – environment simulation | 48ef9b1 | 14 |
| 4 – RGB ensemble | 4417027 | 16 |
| 5 – CARD integration | d0586d8 | 17 |
| 6 – edge controller | 1758bd1 | 18 |
| 7 – dashboard and replay | 3a870e3 | 19 |
| 8 – evaluator, report and final quality gate | 7f9c3d3 and final documentation commit | 25 |

```bash
git log --reverse --format='%H %s' 3122e28..main
git ls-files
```

### Executed commands and observed outcomes

These commands ran from the repository root with the selected virtual environment on PATH, or its explicit Python executable. Preparation and training commands below use the flags actually used.

| Command | Result |
|---|---|
| git clone --branch main --single-branch https://github.com/aditya-a-git/ComputerVision.git | Clean main checkout, baseline 3122e28 |
| python -m scripts.inspect_rdd | Two HF partitions, 38,383 images, 61,075 boxes, zero expected-count deviations |
| python -m scripts.prepare_rdd --max-images 8 --dry-run | 8 selected, zero downloads |
| python -m scripts.prepare_rdd --train 8 --val 4 --max-download-mb 12 | 12 verified pairs |
| python -m scripts.prepare_rdd --train 100 --val 24 --max-download-mb 100 --out data/processed/demo | 124 verified pairs, zero skips |
| python -m scripts.train --epochs 1 --batch 8 --imgsz 160 --out runs/smoke | CPU smoke epoch completed, isolated checkpoint |
| python -m scripts.train --data data/processed/demo/data.yaml --epochs 8 --batch 8 --imgsz 320 --out runs/demo | Eight CPU epochs completed. Best checkpoint copied to models/rdd_best.pt |
| python -m scripts.evaluate --data data/processed/demo/data.yaml --imgsz 320 | Native mAP50 0.005129, mAP50–95 0.000935 |
| python -m scripts.compare --root data/processed/demo | Seven pipelines, 24 identical validation frames each. Committed-evaluator repeat captured in docs/measurements |
| python -m scripts.inspect_card | Listed one five-file group at pinned HF revision. Anonymous HTTP GET for its left image returned 401 |
| python -m pytest -q | Offline unit and AppTest suite; final count 25 |
| ruff check app.py core scripts tests --select E9,F | Passed |
| streamlit run app.py --server.headless true --server.port 8532 | Clean checkout and fresh Python 3.11 CPU environment returned ok at /_stcore/health |
| Streamlit AppTest with local trained weights | All eight views executed; actual four-frame MP4 play, pause, resume and export exercised |
| git push origin main and git ls-remote origin refs/heads/main | Main-only pushes verified |

The tiny checkpoint is local and downloadable from Home in the training workspace. In a fresh clone, follow README to reproduce it or supply your own trained five-class weights. CARD requires your account's acceptance and login. Real CARD labels and geometry were not experimentally validated, and the optional depth anomaly heuristic is omitted. The Colab notebook was generated but execution was verified locally on CPU only. Results, including missed annotations, are in RESULTS.md. No hardware or genuine thermal data were used.

### File tree

```text
app.py, README.md, requirements.txt, context.md
.github/workflows/test.yml, .streamlit/config.toml, .gitignore
datasets/rdd2022.ndjson
core/
  data/{rdd,local,yolo,card}.py
  detector.py, environment.py, sensor_simulator.py, reliability.py
  fusion.py, thermal_controller.py, replay.py, metrics.py
scripts/
  inspect_rdd.py, prepare_rdd.py, train.py, evaluate.py, compare.py
  inspect_card.py, download_card_sample.py
configs/
notebooks/train_yolo.ipynb
tests/test_{app,rdd,prepare,local,detector,environment,fusion,card,thermal,metrics}.py
docs/
  ACADEMIC_SCOPE.md, IMPLEMENTATION_STATUS.md, DATASET_PROVENANCE.md
  EXPERIMENTS.md, RESULTS.md, DEMO_SCRIPT.md
  measurements/tiny-rdd/{report.json,summary.csv,selection.json,training.csv,
                         training-provenance.json,ultralytics.json}
```

## Full-data training milestone

Published epoch 2 of the eight-epoch full RDD run. Every epoch covers all 30,707 training images. Validation on the 7,676 held-out images measured mAP@0.50 0.16703 and mAP@0.50:0.95 0.06761. The downloadable checkpoint is models/rdd_best.pt. Its SHA-256 is ec812673a50a9fcba8a0951caa892b3505ce0bdfa36ba85c4c5ebd21fb143dce. Training is continuing in the original workspace.

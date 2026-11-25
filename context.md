# Project Context — Edge Multimodal Road Infrastructure Diagnostics

> **Audience:** coding agents and engineers implementing a research prototype and eventually an edge-deployed system.
> **Source:** user-supplied article, *Multimodal Sensor Fusion in Edge-Deployed Computer Vision for Urban Infrastructure Diagnostics* (`Pasted text.txt`).
> **Status:** requirements synthesis and proposed implementation plan, **not** an existing repository specification or a validated scientific design.
> **Priority rule:** follow explicit repository/user instructions over recommendations in this document. Do not pretend that proposed modules, datasets, benchmarks, or APIs already exist.

## 1. Mission and desired outcome

Build a vehicle-mounted, on-device computer-vision pipeline to detect, localize, track, and report urban road-surface defects in challenging lighting, weather, and thermal conditions. The research emphasizes multimodal **RGB + thermal infrared** perception with **optional LiDAR/depth**, lightweight detection, intermediate feature-level fusion, INT8-oriented optimization, and sustained operation on constrained edge compute.

The practical outcome is a reproducible, experimentally evaluated prototype that can:

1. ingest timestamped sensor streams;
2. align, validate, and synchronize observations without assuming perfect registration;
3. detect road defects with confidence and provenance;
4. track/consolidate detections across frames and geo-reference them when localization is available;
5. remain operational when one modality becomes unreliable or the device heats up;
6. measure **end-to-end** latency, quality, resource use, and robustness; and
7. export privacy-conscious diagnostic records for review or municipal maintenance workflows.

**Do not conflate research hypotheses with observed performance.** The pasted source is a narrative review/proposal; it contains unverified performance numbers, broad categorical claims, and some mismatched references. Treat its architecture selections as hypotheses to test, not mandated scientific conclusions.

## 2. Research-derived technical direction

### Fusion design space

| Strategy | How it works | Why it matters | Initial implementation stance |
|---|---|---|---|
| Early/data fusion | Register and concatenate raw inputs (e.g. RGB + thermal) | Simple joint learning, but vulnerable to registration errors and bad modalities | Optional comparison baseline; not default |
| Late/decision fusion | Separate detection pipelines; merge predictions | Failure isolation and asynchronous inputs; potentially costly duplicated inference | **Mandatory robustness baseline** and fallback candidate |
| Intermediate/feature fusion | Modality-specific encoders feed a shared fusion block and detection head | Proposed balance of synergy and compute cost | **Primary research candidate** |
| BEV/3D fusion | Project camera/depth/LiDAR features into common ground/voxel space | Useful geometric representation; may be too costly for target devices | Defer until measured need and hardware budget |

For intermediate fusion, evaluate a small spatial/channel gate or quantization-friendly cross-modal mixer **before** attempting heavy attention. Use reliability/missing-modality masks and avoid assuming RGB and thermal pixels are perfectly aligned. The paper mentions frequency-guided approaches, FACA, CAST, and CrossWeaver as inspirations; it does not provide reproducible architectures or implementation details.

### Detection and model choices

- Start from **single-modality RGB** and **single-modality thermal** detection baselines, then build fusion variants.
- Candidate detector heads/backbones from the source: YOLOv10-nano (NMS-free design), YOLO11-nano (often written “YOLOv11”), and MobileNetV4-style lightweight encoders. **Check library availability, licenses, model/export support, and target runtime compatibility before choosing versions.**
- Road defects can be small/thin (particularly cracks). Preserve sufficiently high-resolution features; compare higher-resolution heads or segmentation against simple bounding-box detection. Do **not** assume bounding boxes can faithfully represent thin cracks.
- LayerCAM is an **offline explainability/analysis tool**, not a hard real-time inference dependency.
- Treat LiDAR as optional rather than inventing a LiDAR dataset/interface when none is supplied; leave a clean extension point.

### Optimization and adaptive behavior

- Measure PyTorch/ONNX/FP16 inference first; deploy TensorRT on compatible NVIDIA hardware if supported. Hailo requires its own compiler/runtime path and cannot use TensorRT as a drop-in replacement.
- Compare **post-training quantization (PTQ)** and **quantization-aware training (QAT)**; QAT is a hypothesis for attention-sensitive accuracy recovery, **not guaranteed** to maintain FP32 accuracy. Keep unsupported operations in higher precision if necessary and measure conversion overhead.
- Investigate thermal-aware **mode switching**: full multimodal fusion → lower-cost/one-modality mode → degraded-safe reporting, based on telemetry and measured modality reliability. Use hysteresis and recovery checks to prevent oscillation.
- Maintain temporal tracking to reduce duplicate alerts and assess consistency, but never assume tracking mathematically restores missed detections.

## 3. Initial scope and explicitly deferred work

**In scope (MVP):** offline replay of synchronized or partially synchronized RGB/thermal video; training/evaluation of single-modality and intermediate-fusion models; sensor-quality/missing-input handling; frame-level predictions; simple temporal association; JSONL export; latency + resource profiling; simulated thermal mode switching; reproducible configs and tests.

**Next phase:** live camera capture, on-device TensorRT engine, GPS/IMU-assisted geo-localization, production telemetry, robust mounting/calibration, sustained hot-environment tests, and operator review UI.

**Deferred until justified:** 3D BEV grid, fully general LiDAR fusion, autonomous vehicle control, automated repair prioritization, citywide cloud ingestion, unverified custom CUDA kernels, or large-scale geographic claims.

**Safety boundary:** the system is an **inspection aid**, not a navigation, braking, collision-avoidance, or autonomous driving safety system. Confidence scores do not imply structural risk severity.

## 4. Functional requirements

### 4.1 Sensor ingestion

- Support RGB frames and thermal frames with timestamps, sensor IDs, frame IDs, original dimensions, calibration revision, and acquisition status.
- Store thermal format/units explicitly (raw radiometric values, temperature units, or palette-rendered images); **do not treat colorized thermal images as measured temperatures**.
- Optional metadata: GPS coordinates, GPS uncertainty, IMU pose, camera exposure, sensor temperatures, vehicle speed, and LiDAR depth/point clouds.
- Buffer streams independently, align using timestamps with configurable tolerance, and emit missing/stale modality flags. Never silently reuse stale frames as current observations.
- Provide both offline dataset/replay adapters and a replaceable live-device adapter.

### 4.2 Calibration and spatial alignment

- Support per-sensor intrinsics, distortion, extrinsics, time offsets, and calibration version.
- Offer registered-coordinate transformations where calibration exists; otherwise use explicitly weakly aligned pairing or late fusion.
- Detect/flag invalid or drifted calibration (e.g. reprojection error thresholds), rather than asserting pixel-perfect registration.
- Preserve original source coordinates and define transforms from model-space boxes/masks back to sensor-space.

### 4.3 Perception

- Configurable class taxonomy: initial candidates `pothole`, `crack`, `speed_bump`, and `road_debris`; validate and map to the actual annotated dataset classes before training.
- Model modes: `rgb_only`, `thermal_only`, `late_fusion`, `mid_fusion`.
- Preserve per-detection class, confidence, geometry, model/version, modality/mode, timestamp, and quality flags.
- Work with absent/low-quality modalities using trained masks or explicit fallback; never feed arbitrary zeros at inference unless that exact missing-modality behavior was trained and tested.
- Use small-target evaluation; permit segmentation masks if label format supports them.

### 4.4 Temporal and geographic reporting

- Associate detections across frames using track IDs and configurable temporal confirmation.
- Avoid duplicate reports of the same defect during one pass.
- Emit road-relative or geo-referenced location only if geometry/GPS calibration supports it; include uncertainty and mark `location_status: unavailable` otherwise.
- Output both frame-level and consolidated event-level results.

### 4.5 System health and degradation

- Monitor device temperature, throttling status, power, memory, latency percentiles, dropped/stale frames, and modality health.
- Implement a configurable health state machine with `normal`, `reduced_compute`, `single_modality`, and `unavailable` states.
- Choose fallback modality based on measured quality/availability, not time of day alone.
- Hysteresis: separate enter/exit thresholds and minimum dwell times; log all transitions and reasons.
- If inference cannot meet the budget, emit an explicit degraded/stale health status rather than claiming uninterrupted coverage.

## 5. Nonfunctional goals and acceptance metrics

These are **targets and experiment definitions**, not source-validated guarantees:

| Dimension | Candidate target / measurement | Important qualification |
|---|---|---|
| Sustained inference throughput | Aim for **>= 30 processed FPS** on an explicitly named deployment profile | Measure sustained performance, not peak engine FPS; sensor rates may constrain achievable throughput |
| Frame processing latency | Aim for **p95 <= 33 ms** if 30 FPS serial per-frame processing is required | Throughput and latency differ with pipelining; measure capture-to-result p50/p95/p99 separately |
| Accuracy | Report per-class AP, mAP@0.5, mAP@0.5:0.95, precision, recall, F1 | Never invent minimum AP thresholds before baseline/data review |
| Small defects | Report AP/recall by object area, including <32x32 px, and crack geometry where relevant | Source cites tiny ~9x9 px objects; evaluate actual size distribution |
| Fusion benefit | Compare to RGB-only, thermal-only, and late fusion under identical splits | Report daytime/night/rain/water/occlusion slices where labels exist |
| Quantization | FP32/FP16 vs INT8: delta in AP, performance and memory | Recalibrate/validate per deployed engine and device |
| Reliability | Missing-camera tests, stale-frame tests, temporal drift and calibration perturbation tests | Fail gracefully with auditable quality/status |
| Thermals | Sustained 30–120 min profile, FPS/latency vs temperature and throttle events | Duration and safe limits should be adjusted to actual device constraints |
| Reproducibility | Fixed seeds, pinned env, versioned dataset splits, model artifacts, run manifests | Report run-to-run variance if relevant |

**Physical sanity check:** 50 km/h is ~13.9 m/s; a 100 ms interval corresponds to ~1.39 m of travel. That does **not** prove an object is missed; viewing geometry, camera frame rate, overlap, and tracking matter. The article's reported ~31 FPS FP16 and >58 FPS INT8 should be treated as **unverified example values**, not project acceptance evidence.

## 6. Proposed system architecture

```text
RGB sensor ── capture/decoder ── timestamped buffer ─┐
                                                    ├─ sync + calibration + quality checks
Thermal ───── capture/decoder ── timestamped buffer ─┘           │
                                                                ▼
                                            per-modality transforms / preprocessing
                                                                │
                ┌───────────────────────────────────────────────┤
                ▼                                               ▼
        RGB lightweight encoder                        Thermal lightweight encoder
                └──────────────────────┬────────────────────────┘
                                       ▼
                      reliability-aware feature fusion
                              (or fallback path)
                                       ▼
                      shared detector / optional mask head
                                       ▼
                             postprocess + inverse transforms
                                       ▼
                         temporal association / deduplication
                                       ▼
                       geolocation when available + uncertainty
                                       ▼
                             event export + monitoring

        Hardware telemetry ── policy/state machine ── mode selector
```

**Keep boundaries separable** so datasets, model variants, hardware accelerators, and fusion choices can be swapped without rewriting ingestion or output schemas.

### Suggested repository layout (create only when implementing)

```text
.
├── context.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── model_rgb.yaml
│   ├── model_mid_fusion.yaml
│   ├── train.yaml
│   ├── eval.yaml
│   └── edge_orin_nano.yaml
├── src/road_diag/
│   ├── schemas.py                 # versioned frame/detection/event contracts
│   ├── sensors/                   # file replay and live adapters
│   ├── sync/                      # timestamp pairing, buffering, stale policy
│   ├── calibration/               # intrinsics/extrinsics/transforms
│   ├── preprocessing/             # deterministic, modality-specific transforms
│   ├── data/                      # loaders, annotations, split/QA tools
│   ├── models/                    # single-modality baselines, fusion, heads
│   ├── training/                  # loss, loop, reproducibility, checkpoints
│   ├── inference/                 # runtime interface and postprocessing
│   ├── tracking/                  # temporal association and deduplication
│   ├── localization/              # GPS/pose integration, uncertainty
│   ├── optimization/              # ONNX, TensorRT, PTQ/QAT, validation
│   ├── telemetry/                 # device metrics and thermal mode policy
│   └── reporting/                 # event export and summaries
├── scripts/                       # explicit CLI entrypoints
├── tests/
│   ├── unit/
│   ├── integration/
│   └── hardware/                   # skip cleanly without target device
├── benchmarks/                    # protocols and machine-readable results
└── docs/                          # setup, datasets, calibration, limitations
```

This structure is a **proposal**. Follow existing repo layout and naming conventions if a codebase is provided.

## 7. Minimum versioned data contracts

Use typed structures (dataclasses/Pydantic/Protobuf as appropriate). Keep units and coordinate frames explicit. The following JSON is **illustrative, not a finalized API**:

```json
{
  "schema_version": "0.1.0",
  "frame_id": "drive7-rgb-000123",
  "sensor_id": "front_rgb",
  "modality": "rgb",
  "timestamp_ns": 1795600000000000000,
  "image_uri": "relative/path/to/frame.png",
  "width": 1280,
  "height": 720,
  "calibration_id": "rig_A_v2",
  "status": "valid"
}
```

A synchronized sample must contain `sample_id`, `target_timestamp_ns`, `rgb_frame_ref`, `thermal_frame_ref` (nullable), measured `time_delta_ms` (nullable), `alignment_status`, `quality_flags`, and optional `pose_ref`. Use stable sample IDs for reproducible evaluation.

```json
{
  "schema_version": "0.1.0",
  "event_id": "evt-00042",
  "track_id": "track-8",
  "class_name": "pothole",
  "confidence": 0.91,
  "timestamp_ns": 1795600000000000000,
  "bbox_xyxy_px": [220, 340, 288, 392],
  "bbox_frame": "front_rgb_native",
  "mode": "mid_fusion",
  "model_version": "experiment-003",
  "modality_quality": {"rgb": "ok", "thermal": "ok"},
  "location": null,
  "location_status": "unavailable",
  "calibration_id": "rig_A_v2"
}
```

Real implementation should define whether events contain representative geometry or aggregated per-frame observations, whether confidence is calibrated, how event IDs persist across runs, and how location uncertainty is encoded. Never mix normalized/absolute coordinates, image/vehicle/world frames, or Celsius/Kelvin silently.

## 8. Dataset and labeling requirements

1. **Inventory datasets before coding training logic.** The source mentions CARD and general road-topography data but supplies no machine-readable dataset, schema, license, URL guaranteed to be useful, or exact train/test split. Do not claim CARD is available or has any specific modality until verified.
2. Require explicit dataset provenance, download/license instructions, checksums, sample counts, modality distribution, sensor calibration, and class frequency.
3. Pair modalities by recording/session and time, not lexical filename similarity alone. Validate timestamps, registration and annotation transforms.
4. Split by **drive/route/time/location** (or other capture group) before augmentation to avoid near-duplicate leakage; preserve hard night/weather cases in held-out tests.
5. Normalize thermal imagery according to its sensor semantics; never apply RGB photometric augmentations indiscriminately to thermal raw values.
6. Add realistic modality dropouts, sensor noise, brightness/rain/blur, alignment perturbations and calibration drift to training **only when they model physically plausible conditions**.
7. Evaluate label quality, tiny-object density, occlusion, wet potholes, low contrast, and domain shift. Do not manufacture missing ground-truth labels.
8. If no paired RGB/thermal dataset with usable defect labels is available, implement RGB-only baseline first, then document the multimodal data blocker rather than generating unsupported fusion accuracy claims.

## 9. Training and inference experiment matrix

Run experiments in stages, with identical dataset splits and documented hyperparameters:

| ID | Model/pipeline | Precision | Purpose |
|---|---|---|---|
| B0 | RGB-only lightweight detector | FP32/FP16 | Main accuracy and speed baseline |
| B1 | Thermal-only detector | FP32/FP16 | Modality robustness baseline |
| B2 | RGB + thermal late fusion | FP32/FP16 | Independent-stream/failure baseline |
| F0 | Dual lightweight encoders + simple concat/gating + shared head | FP32/FP16 | Initial feature-fusion candidate |
| F1 | Reliability-aware spatial/channel or efficient attention fusion | FP32/FP16 | Test whether adaptive weighting helps |
| Q0 | Best feasible candidate, PTQ | INT8/mixed | Quantization impact baseline |
| Q1 | Same candidate, QAT | INT8/mixed | Test attention recovery and runtime cost |
| T0 | Runtime mode-switching under thermal/missing-modality stress | Target runtime | Test graceful degradation |

**Ablations:** each modality, fusion block, resolution, tiny-object head, modality-dropout training, misalignment tolerance, PTQ vs QAT, temporal confirmation, fallback policy. Report **parameter count, peak RAM, model artifact size, p50/p95/p99 latency, sustained FPS, watts if available, thermals, and metrics by data slice**. Explicitly record preprocessing and postprocessing time in end-to-end results.

## 10. Thermal-aware execution policy (implementation sketch)

```text
at each telemetry interval:
    read supported temperature / throttle / memory / power signals
    update smoothed values and sensor-quality scores
    if required sensor or model is unavailable:
        select tested compatible fallback; otherwise report unavailable
    else if above configured enter threshold for dwell period:
        select reduced-compute or tested single-modality path
    else if below recovery threshold for dwell period:
        restore full multimodal path if all inputs healthy
    log transition, reason, artifact version, timings and dropped frames
```

**Do not hardcode** 95°C or any generic threshold as a safe operating limit. Read actual SoC, carrier, sensor, enclosure, and power-mode specifications; make thermal thresholds configurable. Preload fallback engines if memory permits, measure switching overhead, and avoid untested model inputs/heads when changing modes. Ensure state remains observable when throughput targets are not met.

## 11. Implementation sequence for a coding agent

### Phase 0 — Repository and evidence audit

- Inspect repository, Python/runtime versions, device target, sensors, available paired data, class labels, constraints and licenses.
- Produce a short `docs/assumptions.md` and document missing prerequisites. Prefer extending existing code and tests over reorganizing.
- Do not download unknown datasets, install heavyweight frameworks, or lock the project to one platform without approval or established project convention.

**Done when:** environment and data inventory exist; uncertain assumptions are visible; a small replay sample is identified.

### Phase 1 — Contracts, sync and baseline

- Implement typed input/output schemas; offline replay; timestamp pairing; missing/stale handling; calibration interfaces.
- Build RGB-only baseline training/evaluation and end-to-end replay CLI; add thermal-only baseline if training data supports it.
- Add dataset QA, groupwise splitting, reproducible training config and smoke tests.

**Done when:** replay → detections → JSONL works on real or clearly labeled synthetic test fixtures and metrics are reproducible.

### Phase 2 — Fusion with robustness

- Add separate RGB/thermal encoders, minimal feature-fusion module, shared detection head, modality masks.
- Add late-fusion reference and simulated corruption/misalignment tests.
- Profile per-stage compute; optimize only after locating bottlenecks.

**Done when:** all configured modes execute; fusion comparisons and sensor-degradation tests run on the same validation splits.

### Phase 3 — Edge deployment and quantization

- Export supported models to ONNX, verify parity, build target-specific engines if hardware exists.
- Record FP16 baseline, PTQ calibration and accuracy comparison; trial QAT only if PTQ loss warrants it and support exists.
- Integrate telemetry and tested thermal policy; measure steady-state throughput, latency, memory and heat.

**Done when:** engine correctness vs reference is measured; cold and sustained results are reproducible; fallbacks are exercised.

### Phase 4 — Field-ready diagnostics

- Add calibrated temporal tracking, optional vehicle pose/location, event dedupe, local persistence and reviewer-friendly evidence.
- Assess dataset shift, privacy retention, operational logs and alerts.
- Create operator and recovery documentation; field-test without implying road-safety autonomy.

**Done when:** events are traceable to input frames, model and calibration versions, and limitations are explicit.

## 12. Testing requirements

**Unit tests:** timestamp ordering, missing/stale pairing, calibration projections and coordinate round trips, image/thermal normalization, box transforms, fusion masks, confidence thresholding, event serialization, policy hysteresis.

**Integration tests:** offline replay including one missing modality; model outputs invariant to batch/replay boundaries within defined tolerance; run exports contain expected provenance; export-engine parity when supported.

**Robustness tests:** darkness, thermal saturation, blur/mud, wet surfaces, artificial sensor disconnect, delayed frames, 2D alignment perturbation, GPS loss, memory pressure, simulated throttling. Label simulated perturbations clearly.

**Performance tests:** device warm-up, multiple power modes, sustained hot runs, per-stage and end-to-end timings, p95/p99 and FPS, dropped input frames, engine swap overhead, logging overhead and energy measurements when instrumentation exists.

**Evaluation hygiene:** separate validation/test, bootstrap uncertainty where practical, per-class confusion and PR curves, explicit IoU/area definitions, documented class mapping, representative difficult-case examples, no cherry-picked footage.

## 13. CLI and configuration recommendations

Prefer a small set of commands consistent with the repository's tooling, e.g.:

```bash
python -m road_diag.data.validate --config configs/data.yaml
python -m road_diag.training.train --config configs/model_rgb.yaml
python -m road_diag.evaluation.run --config configs/eval.yaml
python -m road_diag.inference.replay --config configs/edge_orin_nano.yaml
python -m road_diag.benchmarks.run --config configs/edge_orin_nano.yaml
```

These commands are **proposed interfaces**, not verified working commands. Use typed configuration with at least: sensor formats/resolutions/rates; pairing tolerance; calibration reference; taxonomy; encoder/fusion/head; confidence/IoU; runtime backend and precision; device/power mode; telemetry thresholds; buffer capacity/backpressure; output privacy/retention; random seed; experiment paths.

Every run should generate a manifest recording git commit, timestamp, device/runtime versions, sensor identifiers, model/config hashes, dataset/split versions, precision and evaluation code version. Do not overwrite experiment artifacts silently.

## 14. Coding-agent working instructions

- **Read the current repository and this document first.** Do not assume source code exists or invent available hardware/data.
- Work in small, testable vertical slices. Make the offline pipeline function before building novel research modules.
- Clearly separate **confirmed repository facts**, **source claims**, **assumptions**, and **new experimental observations** in notes and PR descriptions.
- Favor configurable, backend-neutral interfaces; keep NVIDIA-specific code isolated from generic training/evaluation.
- Use existing package manager, formatter, test runner and CI conventions; avoid unrequested dependency churn.
- Report evaluated metrics only when backed by a saved experiment log and exact data/device context. Never copy benchmark values from this context as measured project results.
- If an implementation path is blocked by unavailable paired data, calibration, hardware, licensing, or compatible export ops, record the blocker and deliver the next runnable slice.
- Treat timestamps, coordinate frames, input channel order, thermal units and quantization calibration as safety-critical correctness boundaries.
- Do not add cloud uploads or continuous raw video retention by default; build offline-first and minimize identifiable footage.
- For each change, supply: files modified, design decisions, commands run, tests and results, remaining limitations, and recommended next step.

## 15. Open questions that must be resolved from the actual project

1. Which edge target and power mode: Jetson Orin Nano (exact module/JetPack), Hailo-8, or another platform?
2. What RGB/thermal sensor models, timestamp sources, frame rates, intrinsics and extrinsics are actually available? Is LiDAR present?
3. What **paired road-defect dataset** exists, with what annotations, sensor calibration, licensing and country/road distribution?
4. Is the task bounding-box detection, segmentation, severity estimation, or some combination? How are cracks and speed bumps labeled?
5. Is 30 FPS an end-to-end processing requirement, a camera sampling target, or a best-effort goal? What vehicle speeds and field of view matter?
6. What thermal, power and memory budgets hold in the **real enclosure**, and how are safe thresholds defined?
7. What information is needed in reports: GPS coordinates, defect geometry, evidence images, severity, confidence and human review?
8. What retention, privacy, compliance, and data-governance requirements apply?
9. What inference frameworks/operators are supported and licensed on the precise target?
10. What precision/recall criteria are operationally sufficient, especially false alerts versus missed dangerous defects?

**Default if unanswered:** choose an offline RGB baseline, a clean RGB/thermal extension API, synthetic unit-test fixtures clearly marked as synthetic, and no invented accuracy/performance claims.

## 16. Research claims and citation quality — verify before reliance

The supplied article lists references [1]–[10], but most body claims repeat generic grouped citations, and at least references [2] and [3] concern **software vulnerability detection / Joern**, not pavement diagnostics. A coding agent should not assume those references substantiate vision or benchmark claims. In particular, independently validate:

- datasets and whether CARD actually contains the claimed paired RGB/thermal/defect labels;
- availability and details of FACA, CAST, CrossWeaver, BEVFusion and CenterPoint as relevant implementations;
- backbone compatibility between MobileNetV4 and a YOLO-style head, including head feature scales and exportability;
- claimed superiority of nano over medium variants for tiny defects (not a universal property);
- precision claims that QAT *fully preserves* cross-modal attention or that PTQ *always* catastrophically fails;
- exact device TOPS, power, RAM and operating temperature specifications for each hardware revision;
- reported 31 FPS FP16 and >58 FPS INT8 on Jetson Orin Nano and reported 1–2% mAP change;
- whether an INT8 path truly eliminates intermediate casts and has usable accelerator kernels;
- simplified assertions that any particular architecture is always “optimal,” “mandatory,” or “fail-operational.”

Source links supplied in the document (retain as research starting points, not endorsement):

1. https://thegrenze.com/abstract/journal/7407
2. https://www.researchgate.net/publication/403379558_Software_Vulnerability_Detection_Using_a_Lightweight_Graph_Neural_Network
3. https://joern.io/impact/
4. https://www.researchgate.net/publication/415508310_Fire-Aware_Cross-Modal_Alignment_and_Modality-Adaptive_Fusion_Framework_for_UAV-Based_RGB-Thermal_Fire_and_Smoke_Detection
5. https://pmc.ncbi.nlm.nih.gov/articles/PMC13259335/
6. https://arxiv.org/pdf/2506.21885
7. https://arxiv.org/html/2609.19445v1
8. https://opus.hs-offenburg.de/files/13670/applsci-16-09359-v2.pdf
9. https://www.mdpi.com/1424-8220/26/20/6490
10. https://arxiv.org/pdf/2608.29667

## 17. Definition of done for the first useful deliverable

A successful first increment includes a repository-compatible, documented **offline replay + RGB baseline** with a clearly defined RGB/thermal sample schema, meaningful handling of missing modalities, reproducible data split, saved detection outputs, metrics script, latency report, unit/integration tests, and a documented path to intermediate fusion. If paired RGB/thermal defect data is available, additionally demonstrate a minimal fusion model and compare it fairly against single-modality baselines; otherwise record precisely what is missing.

**Guiding principle:** prefer a measurable, robust edge pipeline with traceable limitations over an unverified complex architecture that merely appears accurate or fast.

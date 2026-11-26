# RoadFusion AI
Adaptive multimodal road defect diagnostics is a local Streamlit demonstrator. Use a road photograph or short MP4, simulate weather, compare RGB inference with a second contrast-enhanced RGB pass, and route inference using a simulated device temperature. The CARD explorer reads real camera and depth data separately.

## Run locally

From the repository root on Linux or macOS, create a Python 3.11 environment. Python 3.12 was also tested. On Linux, install the CPU PyTorch wheels first.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Linux CPU installation
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS use the default PyTorch wheel index. On Windows activate with `.venv\Scripts\activate`. The Home, simulation, recorded results and methods views work in a clean checkout. The Environment Lab includes a labeled synthetic calibration pattern. Network access is only needed for installation and explicit data or pretrained-weight retrieval.

## Prepare a small sample

```bash
python -m scripts.inspect_rdd
python -m scripts.prepare_rdd --dry-run
python -m scripts.prepare_rdd --train 100 --val 24 --max-download-mb 100 --out data/processed/demo
```

Inspection writes normalized JSONL, a Hugging Face DatasetDict cache, mapping JSON, counts and class histogram under local data/cache locations. Preparation preserves the export split, favors class coverage, retains genuine negative images and writes an immutable selection, integrity report, YOLO image/label pairs, preview montage and data.yaml. Default selection is 2,000 train and 400 validation with a 256 MB transfer budget. An exhausted budget can yield a partial selection. Inspect report.json for skips and rerun with the same selection to resume. Review preview.jpg before training. Choose a different output directory to change seed or selection. The split field determines membership even when filenames start with train__.

For existing export images, add `--local /path/to/export-images` to the same preparation command. Original RDD files with different names require an independently verified crosswalk. See the dataset provenance document for the schema and rules.

## Train and evaluate

```bash
python -m scripts.train --data data/processed/demo/data.yaml --epochs 8 --imgsz 320 --batch 8 --device cpu --seed 42 --out runs/demo
mkdir -p models
cp runs/demo/rdd/weights/best.pt models/rdd_pilot.pt
python -m scripts.evaluate --data data/processed/demo/data.yaml --weights models/rdd_pilot.pt --imgsz 320
python -m scripts.compare --root data/processed/demo --weights models/rdd_pilot.pt --limit 24
```

Use the checkpoint path printed by the trainer on repeat runs. The first training invocation may download the YOLO11n initialization. For offline training supply a local YOLO11n initialization with `--base`. Streamlit only loads an existing trusted road-trained checkpoint and validates its five class names. To select another local checkpoint set `ROADFUSION_WEIGHTS`. The repository includes the deployable five-class full-data checkpoint at `models/rdd_best.pt`. A fresh checkout can immediately run Detect on a local road image. The one-epoch smoke checkpoint is separate. For a larger experiment increase the prepared sample and epochs or open [the Colab notebook](notebooks/train_yolo.ipynb), upload a zip with images/, labels/ and data.yaml at its root, and save the returned checkpoint under models/.

## Full RDD training

The full-data checkpoint uses all 30,707 training images from the export. All 7,676 validation images are held out. Full acquisition is opt-in and reads the original signed image URLs. It retains a 320-pixel-long-edge JPEG per verified image for the CPU experiment.

```bash
python -m scripts.prepare_rdd_full --all-images --workers 16
python -m scripts.train_full --out runs/full320
```

The trainer verifies coverage, label files and split integrity, runs eight epochs, and writes final validation metrics to result.json. Recover an interrupted run from its latest completed epoch with `python -m scripts.train_full --out runs/full320 --resume`. [Checkpoint measurements](docs/measurements/full-rdd/)

## CARD access

Accept the Germany Batch 1 dataset conditions under your own Hugging Face account, then run:

```bash
hf auth login
python -m scripts.inspect_card
python -m scripts.download_card_sample --dry-run
python -m scripts.download_card_sample --max-files 5 --max-mb 32
```

The CLI resolves a pinned revision and requests one coherent file group. The public listing worked without credentials. Fetching the selected frame anonymously returned HTTP 401. The Explorer reads the exact local paths under data/card. CARD classes keep their source IDs. No CARD metric or RDD-to-CARD class mapping has been established.

## Evidence and operation

- [Demo script](docs/DEMO_SCRIPT.md) walks through the eight views.
- [Results](docs/RESULTS.md) gives the full-data validation measurements and the earlier pilot failure cases.
- [Experiment protocol](docs/EXPERIMENTS.md) defines metrics and runtime boundaries.
- [Dataset provenance and licenses](docs/DATASET_PROVENANCE.md) documents sources and access conditions.
- [Stage history, commands and file tree](docs/IMPLEMENTATION_STATUS.md) records verification and limitations.

Run `python -m pytest -q` for the offline suite. CI installs dependencies, checks imports and lint, then runs tests without model or dataset downloads.

## Run the paired environment benchmark

With the environment activated and the repository checkpoint in place, run the following from the repository root. Use a fresh preparation directory for this selection.

```bash
python -m scripts.prepare_rdd --train 0 --val 12 --seed 42 --out data/processed/weather12
python -m scripts.environment_benchmark --root data/processed/weather12 --weights models/rdd_best.pt --limit 12 --seed 42 --size 320 --strength 0.65 --conditions daylight darkness fog rain vibration noise --out runs/environment_benchmarks
streamlit run app.py
```

Open **Experiments / Results** for the recorded CPU comparison or a new local run. Each finished local run includes the verified validation manifest, predictions, timings, exports and two sample previews. The published summary also works offline. **Fusion Lab** shows branch detections, quality estimates and the decision behind each fused candidate. **Virtual Edge Device** records explicit controller steps from a manual temperature or a bounded automatic ramp. [Experiment definitions and timing](docs/EXPERIMENTS.md)
## Prepare full RDD images at 640 px (NVIDIA GPU)

Use this instead of the legacy full320 preparer when training at 640 px. Images
are retrieved or imported at their **original dimensions**, verified, then
downscaled only when their longest edge exceeds 640 px. This does **not**
upscale the previously prepared full320 images. The default without
`--imgsz` is still 320 px for backwards compatibility.

```powershell
python -m scripts.prepare_rdd_full --all-images --imgsz 640 --workers 8
python -m scripts.train --data data/processed/full640/data.yaml --base models/rdd_best.pt --epochs 30 --imgsz 640 --batch 8 --device 0 --out runs/full640
python -m scripts.evaluate --data data/processed/full640/data.yaml --weights runs/full640/rdd/weights/best.pt --imgsz 640 --device 0
```

The full preparer also accepts `--out data/processed/my_dataset` and
`--local D:path	oerifiedoriginals` (source files must use exact export
filenames and match the recorded image dimensions). An existing output folder
cannot silently switch resolutions; use a distinct folder for each image size.
The script records its size in `preparation.json`, errors in `errors.json`,
and writes `data.yaml` only after all images are prepared successfully.
Signed source image links may expire; if downloads fail, use independently
verified original files rather than substituting unrelated images.

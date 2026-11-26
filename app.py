import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.benchmark_panel import panel as environment_results
from core.edge_lab import panel as edge_panel
from core.data.card import explorer
from core.data.rdd import NAMES
from core.detector import Detection, annotate, load
from core.environment import CONDITIONS, transform
from core.fusion import execute
from core.replay import video_panel
from core.sensor_simulator import proxy


PAGES = (
    "Home",
    "Detect",
    "Environment Lab",
    "Fusion Lab",
    "Virtual Edge Device",
    "CARD Explorer",
    "Experiments / Results",
    "About / Methods / Limits",
)
WEIGHTS = Path(os.getenv("ROADFUSION_WEIGHTS", "models/rdd_best.pt"))


@st.cache_resource
def cached_model(path, modified):
    return load(path)


def get_model():
    if not WEIGHTS.is_file():
        st.info(
            "Model not trained / weights missing. Run python -m scripts.train after preparing data, then copy the trained best.pt to models/rdd_best.pt."
        )
        return None
    try:
        return cached_model(str(WEIGHTS), WEIGHTS.stat().st_mtime_ns)
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
        return None


def image_source(simulation=False):
    upload = st.file_uploader("Road image", type=["jpg", "jpeg", "png"])
    samples = sorted(Path("data/processed").glob("*/images/val/*"))
    source = upload
    if samples and upload is None:
        source = st.selectbox(
            "Prepared validation frame", samples, format_func=lambda p: p.name
        )
    if (
        source is None
        and simulation
        and st.checkbox("Use synthetic calibration pattern", value=True)
    ):
        x = np.tile(np.arange(384, dtype=np.uint16) * 255 // 383, (192, 1)).astype(
            np.uint8
        )
        x[20:172:12, :] = 230
        st.caption("Synthetic calibration pattern for transform inspection.")
        return np.stack((x, np.flip(x, 1), x), axis=-1)
    if source is None:
        st.info("Upload a JPG or PNG to inspect a road frame.")
        return None
    try:
        with Image.open(source) as im:
            if im.width * im.height > 16_000_000:
                raise ValueError("Image exceeds 16 megapixels")
            return np.asarray(im.convert("RGB"))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        st.error(str(exc))
        return None


def show_run(rgb, boxes, run, label):
    view = annotate(rgb, boxes)
    st.image(view, caption=f"{label} · {len(boxes)} detections")
    st.caption(
        f"Executed {', '.join(run['branches']) or 'no branches'} · {1000 * run['seconds']:.1f} ms · input size {run['size']}"
    )
    rows = [d.record() for d in boxes]
    st.dataframe(
        pd.DataFrame(rows, columns=["class_name", "score", "box", "source"]),
        hide_index=True,
    )
    st.download_button(
        "Detection CSV",
        pd.DataFrame(rows, columns=["class_name", "score", "box", "source"]).to_csv(
            index=False
        ),
        "detections.csv",
        key=label + "csv",
    )
    output = io.BytesIO()
    Image.fromarray(view).save(output, format="PNG")
    st.download_button(
        "Annotated PNG",
        output.getvalue(),
        "annotated.png",
        "image/png",
        key=label + "png",
    )
    st.download_button(
        "Detection JSON",
        json.dumps(dict(detections=rows, run=run)),
        "detections.json",
        key=label + "json",
    )


def fusion_panel(model, rgb, conf):
    st.session_state.setdefault("rgb_health", 1.0)
    st.session_state.setdefault("secondary_health", 1.0)
    health = [
        st.slider("RGB health", 0.0, 1.0, key="rgb_health"),
        st.slider("Proxy health", 0.0, 1.0, key="secondary_health"),
    ]
    scenario = st.radio(
        "Stream connections",
        ["Both available", "Disconnect RGB", "Disconnect secondary", "Disconnect both"],
        horizontal=True,
        key="fusion_connections",
    )

    def restore():
        st.session_state.fusion_connections = "Both available"
        st.session_state.rgb_health = 1.0
        st.session_state.secondary_health = 1.0

    st.button("Restore both streams", on_click=restore)
    connected = (
        scenario in ("Both available", "Disconnect secondary"),
        scenario in ("Both available", "Disconnect RGB"),
    )
    mode = st.selectbox(
        "Inference mode", ["late_fusion", "rgb_only", "secondary_only", "unavailable"]
    )
    threshold = st.slider("Minimum same-class overlap (IoU)", 0.1, 1.0, 0.5)
    st.caption(
        "Heuristic quality averages brightness balance (1 − |mean gray − 127.5| / 127.5), contrast (gray SD / 48) and sharpness (Laplacian variance / 120). Contrast and sharpness cap at 1. Manual health multiplies quality before normalization."
    )
    if rgb is None:
        return
    effective = tuple(h if active else 0.0 for h, active in zip(health, connected))
    if model is None:
        st.image(
            [rgb, proxy(rgb)],
            caption=["Original RGB", "Software-derived RGB view — not thermal."],
            width=275,
        )
        return
    key = (
        hashlib.sha256(rgb.tobytes()).digest(),
        rgb.shape,
        str(WEIGHTS),
        WEIGHTS.stat().st_mtime_ns,
        mode,
        effective,
        conf,
        threshold,
    )
    state = st.session_state
    if state.get("fusion_key") != key:
        state.fusion_output = execute(
            model,
            rgb,
            mode,
            effective,
            confidence=conf,
            threshold=threshold,
            explain=True,
        )
        state.fusion_key = key
    boxes, run = state.fusion_output
    columns = st.columns(3)
    for col, name, label, img in zip(
        columns,
        ("rgb", "proxy"),
        ("Original RGB", "Software-derived RGB view — not thermal."),
        (rgb, proxy(rgb)),
    ):
        rows = run["predictions"].get(name, [])
        detections = [Detection(r["cls"], r["score"], r["box"], name) for r in rows]
        col.image(annotate(img, detections), caption=label)
        col.caption(
            f"{len(rows)} detections" if name in run["branches"] else "Branch inactive"
        )
    with columns[2]:
        show_run(rgb, boxes, run, "Fused result")
    matches = sum(r["kind"] == "matched" for r in run["trace"])
    unmatched = sum(r["kind"] == "unmatched" for r in run["trace"])
    branches = len(run["branches"])
    explanation = [
        "No view was eligible for inference.",
        "Single-branch fallback retains the active view’s scores.",
        "Matched coordinates use reliability × score. Unmatched scores are discounted by the view’s weight share.",
    ][branches]
    st.write(
        f"{branches} branches executed · {matches} matched pairs · {unmatched} unmatched candidates · {len(boxes)} final detections. {explanation}"
    )
    st.dataframe(
        [
            dict(
                view=n,
                connected=c,
                manual_health=h,
                estimated_quality=run["quality"].get(n),
                effective_weight=run["shares"].get(n, 0),
            )
            for n, c, h in zip(("rgb", "proxy"), connected, health)
        ]
    )
    st.caption(
        "Indices identify detections in each branch's prediction order. Unmatched IoU is the best same-class overlap, when available. Only detections returned by YOLO enter this table."
    )
    st.dataframe(run["trace"])


def labs(page):
    model = get_model()
    conf = st.slider("Minimum detection score", 0.01, 1.0, 0.25)
    if page == "Detect":
        classes = st.multiselect("Classes", NAMES, default=NAMES)
        if st.radio("Input", ["Image", "Video"], horizontal=True) == "Video":
            upload = st.file_uploader("Short MP4 · up to 90 frames", type=["mp4"])
            if upload and model:
                video_panel(upload, model, conf, [NAMES.index(n) for n in classes])
            return
    rgb = image_source(page in ("Environment Lab", "Fusion Lab", "Virtual Edge Device"))
    if page == "Environment Lab":
        condition = st.selectbox("Synthetic condition", CONDITIONS)
        strength = st.slider("Intensity", 0.0, 1.0, 0.5)
        seed = st.number_input("Simulation seed", 0, 100000, 42)
        if rgb is not None:
            changed = transform(rgb, condition, strength, seed)
            a, b = st.columns(2)
            for col, img, label in (
                (a, rgb, "Original RGB"),
                (b, changed, "Simulated " + condition),
            ):
                with col:
                    if model:
                        boxes, run = execute(model, img, "rgb_only", confidence=conf)
                        show_run(img, boxes, run, label)
                    else:
                        st.image(img, caption=label)
        return
    if page == "Virtual Edge Device":
        edge_panel(model, rgb, conf, str(WEIGHTS), show_run)
        return
    if page == "Fusion Lab":
        fusion_panel(model, rgb, conf)
        return
    if rgb is None:
        return
    if model:
        boxes, run = execute(model, rgb, "rgb_only", confidence=conf)
        boxes = [d for d in boxes if NAMES[d.cls] in classes]
        show_run(rgb, boxes, run, page)
    elif page == "Detect":
        st.image(rgb, caption="Uploaded RGB")


def main():
    st.set_page_config(page_title="RoadFusion AI", layout="wide")
    st.title("RoadFusion AI")
    page = st.sidebar.radio("Workspace", PAGES)
    st.header("Adaptive multimodal road defect diagnostics" if page == "Home" else page)
    if page == "Home":
        st.write(
            "Prepare a labeled road sample, inspect a frame, then compare detection under simulated weather and sensor loss."
        )
        cols = st.columns(3)
        for col, title, value in zip(
            cols,
            ["RDD metadata", "Trained checkpoint", "CARD sample"],
            [
                "Available" if Path("datasets/rdd2022.ndjson").exists() else "Missing",
                "Available" if WEIGHTS.exists() else "Missing",
                "Local" if any(Path("data/card").rglob("*.jpg")) else "Unavailable",
            ],
        ):
            col.metric(title, value)
        st.code(
            "python -m scripts.inspect_rdd\npython -m scripts.prepare_rdd --dry-run\npython -m scripts.prepare_rdd\npython -m scripts.train"
        )
        st.write(
            "Use Detect for local images and video. Environment Lab applies reproducible perturbations. Fusion Lab runs the same detector on RGB and a derived view. CARD Explorer accepts real gated RGB and depth files."
        )
        summary = Path("data/metadata/summary.json")
        if summary.exists():
            st.json(json.loads(summary.read_text()), expanded=False)
        if WEIGHTS.is_file():
            st.download_button(
                "Download local RDD checkpoint", WEIGHTS.read_bytes(), "rdd_best.pt"
            )
        if not WEIGHTS.exists():
            get_model()
    elif page == "CARD Explorer":
        explorer()
    elif page == "About / Methods / Limits":
        st.markdown(Path("docs/ACADEMIC_SCOPE.md").read_text())
        st.table(
            pd.DataFrame(dict(original=[0, 2, 3, 4, 6], model=range(5), name=NAMES))
        )
        for file in ["DATASET_PROVENANCE", "RESULTS"]:
            path = Path("docs") / (file + ".md")
            if path.exists():
                st.markdown(path.read_text())
    elif page == "Experiments / Results":
        environment_results()
        reports = sorted(
            [
                *Path("runs").glob("comparisons/*/report.json"),
                *Path("docs/measurements").glob("*/report.json"),
            ],
            reverse=True,
        )
        if reports:
            path = st.selectbox("Recorded validation run", reports, format_func=str)
            report = json.loads(path.read_text())
            frame = pd.DataFrame(report["summary"])
            st.dataframe(frame, hide_index=True)
            st.bar_chart(frame.set_index("pipeline")[["map50", "map"]])
            st.dataframe(pd.DataFrame(report["classes"]), hide_index=True)
            st.json(report["provenance"], expanded=False)
            st.download_button("Summary CSV", frame.to_csv(index=False), "summary.csv")
            st.download_button("Full experiment JSON", path.read_bytes(), "report.json")
        else:
            st.info(
                "No paired validation run recorded. Run python -m scripts.compare with prepared images and trained weights."
            )
    else:
        labs(page)


if __name__ == "__main__":
    main()

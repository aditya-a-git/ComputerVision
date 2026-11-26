import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.environment import CONDITIONS


def panel():
    st.subheader("Environment benchmark")
    paths = sorted(
        Path("runs/environment_benchmarks").glob("*/summary_metrics.json"), reverse=True
    )
    snapshot = Path("docs/measurements/environment/summary_metrics.json")
    if snapshot.is_file():
        paths.append(snapshot)
    if not paths:
        st.info(
            "Prepare validation images and run python -m scripts.environment_benchmark to see paired results."
        )
        return
    path = st.selectbox("Environment run", paths, format_func=lambda p: str(p.parent))
    try:
        report = json.loads(path.read_text())
        config = report["config"]
        frame = pd.DataFrame(report["summary"]).set_index("condition")
        charts = ["map50", "recall", "p50_ms"]
        if (
            not set(charts).issubset(frame.columns)
            or list(frame.index) != config["conditions"]
            or not all(c in CONDITIONS for c in frame.index)
        ):
            raise ValueError("Invalid conditions or summary metrics")
        st.caption(
            f"{config['sample_count']} paired validation images · seed {config['seed']} · strength {config['strength']} · {config['size']} px · {config['weights']}"
        )
        st.dataframe(frame)
        for col, metric in zip(st.columns(3), charts):
            col.bar_chart(frame[[metric]], y_label=metric)
        st.caption(
            "Daylight is the baseline. Vibration simulates motion blur. Deltas are absolute except percentage change in mean pipeline latency. Small validation subsets and synthetic weather limit generalization."
        )
        with st.expander("Evaluation settings and per-class metrics"):
            st.json(config)
            st.dataframe(
                [
                    {
                        "file": r["file"],
                        "instances": len(r["boxes"]),
                        "sha256": r["sha256"],
                    }
                    for r in report.get("manifest", [])
                ]
            )
            st.dataframe(pd.DataFrame(report["classes"]))
        st.download_button(
            "Environment CSV", frame.to_csv(), "environment.csv", "text/csv"
        )
        st.download_button(
            "Environment JSON",
            json.dumps(report),
            "environment.json",
            "application/json",
        )
        if (path.parent / "figures").is_dir():
            ids = report["preview_ids"]
            selected = st.selectbox("Validation preview", ids)
            condition = st.selectbox("Preview condition", frame.index)
            if (
                path.parent / "figures" / f"{ids.index(selected)}-daylight-input.jpg"
            ).exists():
                for col, key in zip(st.columns(2), ["daylight", condition]):
                    image = (
                        path.parent
                        / "figures"
                        / f"{ids.index(selected)}-{key}-input.jpg"
                    )
                    col.image(image.read_bytes(), caption=key + " input")
            for col, key in zip(st.columns(3), ["truth", "daylight", condition]):
                image = path.parent / "figures" / f"{ids.index(selected)}-{key}.jpg"
                col.image(
                    image.read_bytes(),
                    caption="Ground truth" if key == "truth" else key + " predictions",
                )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        st.error(f"Cannot read environment benchmark {path}: {exc}")

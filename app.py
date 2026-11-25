from pathlib import Path
import streamlit as st


def main():
    st.set_page_config(page_title="RoadFusion AI", layout="wide")
    st.title("RoadFusion AI")
    st.subheader("Adaptive multimodal road defect diagnostics")
    st.write("Inspect road imagery, simulate weather, and compare RGB detection pipelines.")
    if not Path("models/rdd_best.pt").is_file():
        st.info("Model not trained / weights missing. Prepare the RDD sample and train YOLO11n.")


if __name__ == "__main__":
    main()

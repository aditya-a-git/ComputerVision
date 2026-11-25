from pathlib import Path
import streamlit as st


def main():
    st.set_page_config(page_title="RoadFusion AI", layout="wide")
    st.title("RoadFusion AI")
    st.subheader("Adaptive multimodal road defect diagnostics")
    st.write("Inspect road imagery, simulate weather, and compare RGB detection pipelines.")
    from core.environment import CONDITIONS, transform
    import numpy as np
    from PIL import Image
    upload = st.file_uploader("Environment image", type=["jpg","jpeg","png"])
    condition = st.selectbox("Synthetic condition", CONDITIONS)
    strength = st.slider("Intensity",0.,1.,.5)
    if upload:
        rgb = np.asarray(Image.open(upload).convert("RGB"))
        st.image([rgb, transform(rgb,condition,strength)],caption=["Original","Simulation"])
    if not Path("models/rdd_best.pt").is_file():
        st.info("Model not trained / weights missing. Prepare the RDD sample and train YOLO11n.")


if __name__ == "__main__":
    main()

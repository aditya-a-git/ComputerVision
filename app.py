import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.data.card import explorer
from core.data.rdd import NAMES
from core.detector import annotate,load
from core.environment import CONDITIONS,transform
from core.fusion import execute
from core.replay import video_panel
from core.sensor_simulator import proxy
from core.thermal_controller import Controller

PAGES=("Home","Detect","Environment Lab","Fusion Lab","Virtual Edge Device","CARD Explorer","Experiments / Results","About / Methods / Limits")
WEIGHTS=Path(os.getenv("ROADFUSION_WEIGHTS","models/rdd_best.pt"))


@st.cache_resource
def cached_model(path,modified):
    return load(path)


def get_model():
    if not WEIGHTS.is_file():
        st.info("Model not trained / weights missing. Run python -m scripts.train after preparing data, then copy the trained best.pt to models/rdd_best.pt.")
        return None
    try:
        return cached_model(str(WEIGHTS),WEIGHTS.stat().st_mtime_ns)
    except (ValueError,OSError,RuntimeError) as exc:
        st.error(str(exc))
        return None


def image_source():
    upload=st.file_uploader("Road image",type=["jpg","jpeg","png"])
    samples=sorted(Path("data/processed").glob("*/images/val/*"))
    source=upload
    if samples and upload is None:
        source=st.selectbox("Prepared validation frame",samples,format_func=lambda p:p.name)
    if source is None:
        st.info("Upload a JPG or PNG to inspect a road frame.")
        return None
    try:
        with Image.open(source) as im:
            if im.width*im.height>16_000_000:
                raise ValueError("Image exceeds 16 megapixels")
            return np.asarray(im.convert("RGB"))
    except (OSError,ValueError,Image.DecompressionBombError) as exc:
        st.error(str(exc))
        return None


def show_run(rgb,boxes,run,label):
    view=annotate(rgb,boxes)
    st.image(view,caption=f"{label} · {len(boxes)} detections")
    st.caption(f"Executed {', '.join(run['branches']) or 'no branches'} · {1000*run['seconds']:.1f} ms · input size {run['size']}")
    rows=[d.record() for d in boxes]
    st.dataframe(pd.DataFrame(rows,columns=["class_name","score","box","source"]),hide_index=True)
    output=io.BytesIO()
    Image.fromarray(view).save(output,format="PNG")
    st.download_button("Annotated PNG",output.getvalue(),"annotated.png","image/png",key=label+"png")
    st.download_button("Detection JSON",json.dumps(dict(detections=rows,run=run)),"detections.json",key=label+"json")


def labs(page):
    model=get_model()
    conf=st.slider("Minimum detection score",.01,1.,.25)
    if page=="Detect":
        classes=st.multiselect("Classes",NAMES,default=NAMES)
        if st.radio("Input",["Image","Video"],horizontal=True)=="Video":
            upload=st.file_uploader("Short MP4 · up to 90 frames",type=["mp4"])
            if upload and model:
                video_panel(upload,model,conf,[NAMES.index(n) for n in classes])
            return
    rgb=image_source()
    if page=="Environment Lab":
        condition=st.selectbox("Synthetic condition",CONDITIONS)
        strength=st.slider("Intensity",0.,1.,.5)
        seed=st.number_input("Simulation seed",0,100000,42)
        if rgb is not None:
            changed=transform(rgb,condition,strength,seed)
            a,b=st.columns(2)
            for col,img,label in ((a,rgb,"Original RGB"),(b,changed,"Simulated "+condition)):
                with col:
                    if model:
                        boxes,run=execute(model,img,"rgb_only",confidence=conf)
                        show_run(img,boxes,run,label)
                    else:
                        st.image(img,caption=label)
        return
    mode,size,health,threshold="rgb_only",640,(1.,1.),.5
    if page in ("Virtual Edge Device","Fusion Lab"):
        health=(st.slider("RGB health",0.,1.,1.),st.slider("Proxy health",0.,1.,1.))
        if page=="Fusion Lab":
            st.caption("Both passes use the same RGB-trained network. The proxy is contrast-enhanced grayscale. It contains no measured thermal information.")
            mode=st.selectbox("Inference mode",["late_fusion","rgb_only","secondary_only","unavailable"])
            threshold=st.slider("Minimum same-class overlap (IoU)",.1,1.,.5)
            if rgb is not None:
                st.image([rgb,proxy(rgb)],caption=["RGB","Derived proxy"],width=275)
        else:
            temp=st.slider("Simulated device temperature (°C)",20,100,50)
            state=st.session_state
            if "controller" not in state:
                state.controller=Controller()
                state.edge_history=[]
            event,mode,size=state.controller.step(temp,health)
            st.write(f"Requested route · **{event}** · {size} px")
            st.caption("Illustrative thresholds 70°C and 85°C, 5°C recovery margin and two controller updates minimum dwell.")
            if rgb is not None and model and st.button("Process frame"):
                boxes,run=execute(model,rgb,mode,health,size,conf)
                state.edge_history=(state.edge_history+[dict(temperature=temp,route=event,**run)])[-100:]
                show_run(rgb,boxes,run,event)
            if state.edge_history:
                history=pd.DataFrame(state.edge_history)
                st.dataframe(history)
                timed=history[history.branches.map(len)>0].seconds
                if len(timed):
                    st.write(f"p50 {timed.median()*1000:.1f} ms · p95 {timed.quantile(.95)*1000:.1f} ms · {1/timed.mean():.2f} processed FPS")
            return
    if rgb is None:
        return
    if model:
        boxes,run=execute(model,rgb,mode,health,size,conf,threshold)
        if page=="Detect":
            boxes=[d for d in boxes if NAMES[d.cls] in classes]
        if page=="Fusion Lab":
            st.write("View reliability",run["weights"])
            st.caption("Matched coordinates use reliability × detection score. Scores for unmatched boxes are scaled by their view’s share of total reliability.")
        show_run(rgb,boxes,run,page)
    elif page=="Detect":
        st.image(rgb,caption="Uploaded RGB")


def main():
    st.set_page_config(page_title="RoadFusion AI",layout="wide")
    st.title("RoadFusion AI")
    page=st.sidebar.radio("Workspace",PAGES)
    st.header("Adaptive multimodal road defect diagnostics" if page=="Home" else page)
    if page=="Home":
        st.write("Prepare a labeled road sample, inspect a frame, then compare detection under simulated weather and sensor loss.")
        cols=st.columns(3)
        for col,title,value in zip(cols,["RDD metadata","Trained checkpoint","CARD sample"],
            ["Available" if Path("datasets/rdd2022.ndjson").exists() else "Missing","Available" if WEIGHTS.exists() else "Missing","Local" if any(Path("data/card").rglob("*.jpg")) else "Unavailable"]):
            col.metric(title,value)
        st.code("python -m scripts.inspect_rdd\npython -m scripts.prepare_rdd --dry-run\npython -m scripts.prepare_rdd\npython -m scripts.train")
        st.write("Use Detect for local images and video. Environment Lab applies reproducible perturbations. Fusion Lab runs the same detector on RGB and a derived view. CARD Explorer accepts real gated RGB and depth files.")
        summary=Path("data/metadata/summary.json")
        if summary.exists():
            st.json(json.loads(summary.read_text()),expanded=False)
        if not WEIGHTS.exists():
            get_model()
    elif page=="CARD Explorer":
        explorer()
    elif page=="About / Methods / Limits":
        st.markdown(Path("docs/ACADEMIC_SCOPE.md").read_text())
        st.table(pd.DataFrame(dict(original=[0,2,3,4,6],model=range(5),name=NAMES)))
        for file in ["DATASET_PROVENANCE","RESULTS"]:
            path=Path("docs")/(file+".md")
            if path.exists(): st.markdown(path.read_text())
    elif page=="Experiments / Results":
        reports=sorted([*Path("runs").glob("comparisons/*/report.json"),*Path("docs/measurements").glob("*/report.json")],reverse=True)
        if reports:
            path=st.selectbox("Recorded validation run",reports,format_func=str)
            report=json.loads(path.read_text())
            frame=pd.DataFrame(report["summary"])
            st.dataframe(frame,hide_index=True)
            st.bar_chart(frame.set_index("pipeline")[["map50","map"]])
            st.dataframe(pd.DataFrame(report["classes"]),hide_index=True)
            st.json(report["provenance"],expanded=False)
            st.download_button("Summary CSV",frame.to_csv(index=False),"summary.csv")
            st.download_button("Full experiment JSON",path.read_bytes(),"report.json")
        else:
            st.info("No paired validation run recorded. Run python -m scripts.compare with prepared images and trained weights.")
    else:
        labs(page)


if __name__=="__main__":
    main()

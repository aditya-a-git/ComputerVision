"""Process video in short resumable chunks. Temporary codecs/files live per request."""
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import streamlit as st
from core.detector import annotate, predict


def video_panel(upload,model,confidence,classes):
    key=(upload.file_id,confidence,tuple(classes))
    if st.session_state.get("video_key")!=key:
        st.session_state.update(video_key=key,frames=[],frame_rows=[],playing=False,finished=False)
    state=st.session_state
    left,right=st.columns(2)
    if left.button("Play",disabled=state.finished):
        state.playing=True
    if right.button("Pause"):
        state.playing=False

    @st.fragment(run_every=.5 if state.playing else None)
    def playback():
        if state.playing and not state.finished:
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"input.mp4"
                path.write_bytes(upload.getbuffer())
                cap=cv2.VideoCapture(str(path))
                try:
                    if not cap.isOpened():
                        st.error("MP4 decoder could not open the upload.")
                        state.playing=False
                        return
                    state.video_fps=float(cap.get(cv2.CAP_PROP_FPS)) or 12.
                    cap.set(cv2.CAP_PROP_POS_FRAMES,len(state.frames))
                    for _ in range(3):
                        start=perf_counter()
                        ok,frame=cap.read()
                        if not ok or len(state.frames)>=90:
                            state.finished=True
                            state.playing=False
                            break
                        h,w=frame.shape[:2]
                        scale=min(1,640/max(h,w))
                        frame=cv2.resize(frame,(max(2,int(w*scale)//2*2),max(2,int(h*scale)//2*2)))
                        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        boxes,_=predict(model,rgb,confidence=confidence)
                        boxes=[d for d in boxes if d.cls in classes]
                        annotated=cv2.cvtColor(annotate(rgb,boxes),cv2.COLOR_RGB2BGR)
                        ok,encoded=cv2.imencode(".jpg",annotated)
                        if not ok:
                            raise ValueError("Frame encoder failed")
                        state.frame_rows.append(dict(frame=len(state.frames),seconds=perf_counter()-start,
                            width=frame.shape[1],height=frame.shape[0],detections=[d.record() for d in boxes]))
                        state.frames.append(encoded.tobytes())
                finally:
                    cap.release()
            if state.finished:
                st.rerun()
        if state.frames:
            st.image(state.frames[-1],caption=f"Processed {len(state.frames)} of up to 90 frames")
    playback()
    if state.frames and not state.playing:
        import json
        import pandas as pd
        import numpy as np
        st.download_button("Frame JSON",json.dumps(state.frame_rows),"frames.json")
        st.download_button("Frame CSV",pd.DataFrame(state.frame_rows).to_csv(index=False),"frames.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"annotated.mp4"
            row=state.frame_rows[0]
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),state.video_fps,(row["width"],row["height"]))
            if writer.isOpened():
                try:
                    for jpg in state.frames:
                        writer.write(cv2.imdecode(np.frombuffer(jpg,np.uint8),cv2.IMREAD_COLOR))
                finally:
                    writer.release()
                st.download_button("Annotated MP4",path.read_bytes(),"roadfusion.mp4","video/mp4")

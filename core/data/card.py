"""Sequence-local CARD associations and sparse optical-axis depth in metres."""

import json
import re
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

REPO = "CARD-Data/CARD-Germany-Batch1"


def groups(files):
    names = set(files)
    for left in sorted(names):
        match = re.fullmatch(r"(.+)/img/cam_1/cam_1_(\d+)\.jpg", left)
        if not match or ".." in PurePosixPath(left).parts or left.startswith("/"):
            continue
        seq, stamp = match.groups()
        paths = {
            "left": left,
            "right": f"{seq}/img/cam_0/cam_0_{stamp}.jpg",
            "labels": f"{seq}/labels/cam_1_{stamp}.txt",
            "depth": f"{seq}/agg_depth/cam_1/cam_1_{stamp}.npz",
            "calibration": f"{seq}/export/output.laz.trajectory.json",
        }
        yield {k: v for k, v in paths.items() if v in names}


def project(points, K, shape):
    h, w = shape
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected N by 3 camera-space points")
    p = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    uvw = p @ K.T
    xy = np.rint(uvw[:, :2] / uvw[:, 2:3])
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    x, y = xy[keep].astype(int).T
    depth = np.full((h, w), np.inf, np.float32)
    np.minimum.at(depth, (y, x), p[keep, 2])
    depth[~np.isfinite(depth)] = np.nan
    return depth


def overlay(root, group, rgb):
    ci = json.loads((root / group["calibration"]).read_text())["camera_infos"]["cam_1"]
    K = np.array(
        [[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]], np.float64
    )
    dist = np.asarray(ci["dist_coeffs"], dtype=np.float64)
    with np.load(root / group["depth"], allow_pickle=False) as archive:
        depth = project(archive["pts_cam"], K, rgb.shape[:2])
    aligned = cv2.undistort(rgb, K, dist, None, K)
    valid = np.isfinite(depth)
    if not valid.any():
        return aligned, depth
    indices = np.uint8(np.nan_to_num(depth / 50, nan=0).clip(0, 1) * 255)
    colors = cv2.cvtColor(
        cv2.applyColorMap(indices, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    aligned[valid] = colors[valid]
    return aligned, depth


def explorer():
    import streamlit as st
    from PIL import Image

    root = Path("data/card")
    frames = list(
        groups(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    )
    if not frames:
        st.info(
            "CARD frames unavailable. Accept the dataset conditions on Hugging Face, run hf auth login, then python -m scripts.download_card_sample."
        )
        return
    selected = st.selectbox("Left camera frame", [f["left"] for f in frames])
    group = next(f for f in frames if f["left"] == selected)
    try:
        rgb = np.array(Image.open(root / selected).convert("RGB"))
        st.image(rgb, caption="cam_1 · left RGB")
        if "right" in group:
            st.image(
                str(root / group["right"]),
                caption="cam_0 · right RGB, same sequence and filename timestamp",
            )
        if "labels" in group:
            st.text_area(
                "CARD labels · original IDs and normalized boxes",
                (root / group["labels"]).read_text(),
                disabled=True,
            )
        if {"depth", "calibration"} <= group.keys():
            view, depth = overlay(root, group, rgb)
            st.image(
                view,
                caption="Undistorted left RGB with sparse camera-axis depth · turbo scale 0–50 m",
            )
            st.caption(f"{np.isfinite(depth).sum():,} pixels have projected depth.")
        else:
            st.info("Matching depth or camera calibration missing.")
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        st.error(f"CARD frame could not be loaded: {exc}")

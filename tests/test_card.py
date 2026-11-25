import numpy as np
from core.data.card import groups, project


def test_exact_pairing_and_projection():
    files = [
        "a/img/cam_1/cam_1_12.jpg",
        "a/img/cam_0/cam_0_12.jpg",
        "a/agg_depth/cam_1/cam_1_13.npz",
        "b/labels/cam_1_12.txt",
    ]
    assert set(next(groups(files))) == {"left", "right"}
    pts = np.array([[0, 0, 5], [0, 0, 2], [0, 0, -2], [30, 2, 1], [np.nan, 0, 4]])
    grid = project(pts, np.eye(3), (3, 3))
    assert grid[0, 0] == 2 and np.isfinite(grid).sum() == 1


def test_undistorted_overlay(tmp_path):
    import json
    from core.data.card import overlay

    (tmp_path / "cal.json").write_text(
        json.dumps(
            {
                "camera_infos": {
                    "cam_1": dict(fx=10, fy=10, cx=2, cy=2, dist_coeffs=[0, 0, 0, 0, 0])
                }
            }
        )
    )
    np.savez(tmp_path / "points.npz", pts_cam=[[0, 0, 3], [0, 0, 8]])
    rgb = np.zeros((5, 5, 3), np.uint8)
    image, depth = overlay(
        tmp_path, {"depth": "points.npz", "calibration": "cal.json"}, rgb
    )
    assert depth[2, 2] == 3 and image[2, 2].any()

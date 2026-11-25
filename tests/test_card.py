import numpy as np
from core.data.card import groups, project


def test_exact_pairing_and_projection():
    files=["a/img/cam_1/cam_1_12.jpg","a/img/cam_0/cam_0_12.jpg","a/agg_depth/cam_1/cam_1_13.npz","b/labels/cam_1_12.txt"]
    assert set(next(groups(files)))=={"left","right"}
    pts=np.array([[0,0,5],[0,0,2],[0,0,-2],[30,2,1],[np.nan,0,4]])
    grid=project(pts,np.eye(3),(3,3))
    assert grid[0,0]==2 and np.isfinite(grid).sum()==1

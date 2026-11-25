from unittest.mock import Mock
import numpy as np
import pytest
from core.detector import annotate, load, predict


def test_mock_detection(tmp_path):
    with pytest.raises(FileNotFoundError): load(tmp_path/"missing.pt")
    result=Mock()
    result.boxes.data.cpu.return_value.tolist.return_value=[[1,2,16,18,.8,3]]
    model=Mock()
    model.predict.return_value=[result]
    rgb=np.zeros((20,20,3),np.uint8)
    boxes,seconds=predict(model,rgb)
    assert boxes[0].record()["class_name"]=="pothole" and seconds>=0
    assert annotate(rgb,boxes).any() and not rgb.any()

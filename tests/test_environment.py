import numpy as np
import pytest
from core.environment import CONDITIONS, transform


@pytest.mark.parametrize("condition", CONDITIONS)
def test_perturbation(condition):
    img = np.random.default_rng(1).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    result = transform(img, condition)
    assert result.dtype == img.dtype and result.shape == img.shape
    assert np.array_equal(result, transform(img, condition))
    assert np.array_equal(img, transform(img, condition, 0))
    assert np.array_equal(img, result) == (condition == "daylight")

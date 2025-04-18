import numpy as np
import pytest
from vcast.stat import compute_fss

def test_compute_fss_basic():
    fcst = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1]
    ])
    ref = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    result = compute_fss(fcst, ref, fcst_threshold=0.5, ref_threshold=0.5, window_size=2)
    assert 0 <= result <= 1

def test_compute_fss_no_events():
    arr = np.zeros((3, 3))
    result = compute_fss(arr, arr, fcst_threshold=1, ref_threshold=1, window_size=2)
    assert np.isnan(result)

def test_compute_fss_invalid_shape():
    with pytest.raises(ValueError):
        compute_fss(np.ones((3, 3)), np.ones((2, 2)), fcst_threshold=0.5, ref_threshold=0.5, window_size=2)

def test_compute_fss_invalid_window():
    with pytest.raises(ValueError):
        compute_fss(np.ones((3, 3)), np.ones((3, 3)), fcst_threshold=0.5, ref_threshold=0.5, window_size=0)


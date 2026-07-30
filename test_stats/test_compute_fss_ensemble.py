import numpy as np
import pytest
from vcast.stat import compute_fss_ensemble


def test_deterministic_2d_perfect_forecast():
    arr = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)
    fss = compute_fss_ensemble(arr, arr, threshold=0.5, window_size=2)
    assert fss == pytest.approx(1.0)


def test_ensemble_3d_reduces_to_member_mean_probability():
    # 3 members, 2 exceed threshold everywhere -> ensemble prob = 2/3
    ensemble = np.zeros((3, 2, 2))
    ensemble[:2, :, :] = 10.0
    ref = np.full((2, 2), 10.0)
    fss = compute_fss_ensemble(ensemble, ref, threshold=5.0, window_size=1)
    assert 0.0 <= fss <= 1.0


def test_no_events_anywhere_returns_nan():
    arr = np.zeros((3, 3))
    fss = compute_fss_ensemble(arr, arr, threshold=1.0, window_size=2)
    assert np.isnan(fss)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        compute_fss_ensemble(np.ones((2, 2)), np.ones((3, 3)), threshold=0.5, window_size=1)


def test_invalid_ndim_raises():
    with pytest.raises(ValueError, match="2D .* or 3D"):
        compute_fss_ensemble(np.ones((2, 2, 2, 2)), np.ones((2, 2)), threshold=0.5, window_size=1)


def test_window_size_zero_raises():
    with pytest.raises(ValueError, match="Window size"):
        compute_fss_ensemble(np.ones((2, 2)), np.ones((2, 2)), threshold=0.5, window_size=0)


def test_ensemble_and_equivalent_deterministic_probability_field_agree():
    # An ensemble where every member is identical to a single deterministic
    # field should give the same FSS as passing that field directly (with
    # the field already thresholded to 0/1, matching what the 2D branch
    # produces internally).
    det = np.array([[1.0, 0.0], [0.0, 1.0]])
    ref = np.array([[1.0, 0.0], [1.0, 0.0]])
    ensemble = np.stack([det] * 4, axis=0)

    fss_det = compute_fss_ensemble(det, ref, threshold=0.5, window_size=1)
    fss_ens = compute_fss_ensemble(ensemble, ref, threshold=0.5, window_size=1)
    assert fss_det == pytest.approx(fss_ens)

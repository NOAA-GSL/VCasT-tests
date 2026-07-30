import numpy as np
import pytest
from vcast.stat import compute_brier_score


def test_perfect_binary_forecast_gives_zero():
    fcst = np.array([[10.0, 0.0], [10.0, 0.0]])
    ref = np.array([[10.0, 0.0], [10.0, 0.0]])
    score = compute_brier_score(fcst, ref, threshold=5.0, window_size=1, probability_type="binary")
    assert score == pytest.approx(0.0)


def test_completely_wrong_binary_forecast_gives_one():
    fcst = np.array([[10.0, 0.0], [10.0, 0.0]])
    ref = np.array([[0.0, 10.0], [0.0, 10.0]])
    score = compute_brier_score(fcst, ref, threshold=5.0, window_size=1, probability_type="binary")
    assert score == pytest.approx(1.0)


def test_raw_probability_type_normalizes_forecast():
    fcst = np.array([[0.0, 5.0], [10.0, 5.0]])
    ref = np.array([[10.0, 0.0], [10.0, 0.0]])
    score = compute_brier_score(fcst, ref, threshold=5.0, window_size=1, probability_type="raw")
    assert 0.0 <= score <= 1.0


def test_raw_probability_constant_forecast_returns_nan():
    fcst = np.full((2, 2), 5.0)  # max == min -> division by zero guarded
    ref = np.array([[10.0, 0.0], [10.0, 0.0]])
    score = compute_brier_score(fcst, ref, threshold=5.0, window_size=1, probability_type="raw")
    assert np.isnan(score)


def test_invalid_probability_type_raises():
    fcst = np.ones((2, 2))
    ref = np.ones((2, 2))
    with pytest.raises(ValueError, match="Invalid probability_type"):
        compute_brier_score(fcst, ref, threshold=0.5, window_size=1, probability_type="bogus")


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        compute_brier_score(np.ones((2, 2)), np.ones((3, 3)), threshold=0.5, window_size=1)


def test_window_size_below_one_raises():
    with pytest.raises(ValueError, match="Window size"):
        compute_brier_score(np.ones((2, 2)), np.ones((2, 2)), threshold=0.5, window_size=0)


def test_sigmoid_probability_type_stays_in_unit_interval():
    fcst = np.array([[-5.0, 0.0], [5.0, 10.0]])
    ref = np.array([[0.0, 0.0], [10.0, 10.0]])
    score = compute_brier_score(fcst, ref, threshold=5.0, window_size=1, probability_type="sigmoid")
    assert 0.0 <= score <= 1.0

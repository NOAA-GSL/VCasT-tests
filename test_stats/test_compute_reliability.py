import numpy as np
import pytest
from vcast.stat import compute_reliability


def test_perfectly_calibrated_ensemble():
    # Every grid point where exactly half the members exceed the threshold
    # is observed as an event exactly half the time -> perfect calibration
    # at the 0.5 probability bin.
    n_members = 10
    ensemble = np.zeros((n_members, 4, 1))
    ensemble[:5, :, :] = 100.0  # 5 of 10 members exceed threshold everywhere
    obs = np.array([[100.0], [100.0], [0.0], [0.0]])  # half the points are events

    bin_centers, observed_freqs, counts = compute_reliability(ensemble, obs, threshold=50.0, n_bins=10)

    # forecast_prob is 0.5 at all 4 points -> falls in the bin covering [0.5, 0.6)
    bin_idx = np.searchsorted(np.linspace(0, 1, 11), 0.5, side="right") - 1
    assert counts[bin_idx] == 4
    assert observed_freqs[bin_idx] == pytest.approx(0.5)


def test_empty_bins_are_nan_not_zero():
    ensemble = np.zeros((4, 2, 2))  # no member ever exceeds threshold -> prob 0 everywhere
    obs = np.zeros((2, 2))
    bin_centers, observed_freqs, counts = compute_reliability(ensemble, obs, threshold=10.0, n_bins=5)

    # Only the bin containing probability 0 should have any counts.
    assert counts[0] == 4
    assert np.all(counts[1:] == 0)
    assert np.all(np.isnan(observed_freqs[1:]))


def test_bin_centers_span_unit_interval():
    ensemble = np.zeros((4, 2, 2))
    obs = np.zeros((2, 2))
    bin_centers, _, _ = compute_reliability(ensemble, obs, threshold=10.0, n_bins=4)
    assert bin_centers == pytest.approx([0.125, 0.375, 0.625, 0.875])


def test_high_but_not_certain_probability_lands_in_last_bin():
    ensemble = np.full((10, 2, 2), 0.0)
    ensemble[:9, :, :] = 100.0  # 9 of 10 members exceed threshold -> prob 0.9
    obs = np.full((2, 2), 100.0)
    _, observed_freqs, counts = compute_reliability(ensemble, obs, threshold=50.0, n_bins=10)
    assert counts[-1] == 4
    assert observed_freqs[-1] == pytest.approx(1.0)


def test_forecast_probability_exactly_one_is_excluded_from_every_bin():
    # Binning uses a half-open interval [bins[i], bins[i+1]) for every bin,
    # including the last -- so a forecast probability of exactly 1.0 never
    # satisfies `< bins[-1]` (which is 1.0) and silently isn't counted in
    # any bin. This documents that existing edge-case behavior rather than
    # asserting what "should" happen, since fixing it is a separate,
    # deliberate decision outside the scope of this test.
    ensemble = np.full((6, 2, 2), 100.0)  # all members exceed threshold -> prob 1.0 everywhere
    obs = np.full((2, 2), 100.0)
    _, observed_freqs, counts = compute_reliability(ensemble, obs, threshold=50.0, n_bins=10)
    assert counts.sum() == 0
    assert np.all(np.isnan(observed_freqs))

import numpy as np
import pytest
from vcast.stat import compute_scores


def test_compute_scores_local_grid_point():
    fcst = np.array([[1, 0], [1, 1]])
    ref = np.array([[1, 0], [0, 1]])
    hits, misses, false_alarms, correct_rejections, total = compute_scores(
        fcst, ref, fcst_threshold=0.5, ref_threshold=0.5
    )
    # (0,0): fcst=1>=.5, ref=1>=.5 -> hit
    # (0,1): fcst=0, ref=0 -> correct rejection
    # (1,0): fcst=1, ref=0 -> false alarm
    # (1,1): fcst=1, ref=1 -> hit
    assert hits == 2
    assert misses == 0
    assert false_alarms == 1
    assert correct_rejections == 1
    assert total == 4


def test_compute_scores_all_misses():
    fcst = np.zeros((2, 2))
    ref = np.ones((2, 2))
    hits, misses, false_alarms, correct_rejections, total = compute_scores(
        fcst, ref, fcst_threshold=0.5, ref_threshold=0.5
    )
    assert hits == 0
    assert misses == 4
    assert false_alarms == 0
    assert correct_rejections == 0


def test_compute_scores_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_scores(np.ones((2, 2)), np.ones((3, 3)), fcst_threshold=0.5, ref_threshold=0.5)


def test_compute_scores_radius_matches_local_when_radius_zero():
    fcst = np.array([[1, 0], [1, 1]])
    ref = np.array([[1, 0], [0, 1]])
    local = compute_scores(fcst, ref, fcst_threshold=0.5, ref_threshold=0.5, radius=0)
    default = compute_scores(fcst, ref, fcst_threshold=0.5, ref_threshold=0.5, radius=None)
    assert local == default


def test_compute_scores_radius_of_influence_counts_neighborhood_hits():
    # A single event at (1,1) in the reference; forecast has an event one
    # cell away at (0,1). With radius=0 that's a miss (no fcst event at the
    # exact ref location) *and* a false alarm (fcst event with no ref event
    # there); with radius=1 the neighborhood check turns the miss into a hit.
    fcst = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    ref = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    hits_r0, misses_r0, _, _, _ = compute_scores(fcst, ref, 0.5, 0.5, radius=0)
    hits_r1, misses_r1, _, _, _ = compute_scores(fcst, ref, 0.5, 0.5, radius=1)

    assert hits_r0 == 0 and misses_r0 == 1
    assert hits_r1 == 1 and misses_r1 == 0

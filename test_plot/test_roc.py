"""Unit tests for vcast.plot.roc.Roc.

_roc_points() is the pure computation at the heart of the ROC plot: given
one aggregated PCT row, it derives (POFD, PODY, AUC) from the oy_i/on_i
bin counts. These tests exercise it directly against hand-computable
cases, plus add_lines() end-to-end for bin-count/config-driven behavior
(mirrors the reliability plot's coverage in test_plot/test_reliability.py).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from vcast.plot.roc import Roc
from _helpers import make_plot_config, make_var_entry


def make_row(bins):
    """bins: dict of {index: (oy, on)}. Returns a pandas Series like one row
    of an aggregated PCT file."""
    data = {}
    for i, (oy, on) in bins.items():
        data[f"oy_{i}"] = oy
        data[f"on_{i}"] = on
    return pd.Series(data)


class TestRocPoints:
    def test_perfect_classifier_auc_near_one(self):
        # All obs-yes cases land in the highest bin, all obs-no in the lowest.
        row = make_row({1: (0, 100), 2: (0, 0), 3: (100, 0)})
        pofd, pody, auc = Roc(make_plot_config([]))._roc_points(row)
        assert auc == pytest.approx(1.0, abs=1e-9)
        assert pody[0] == 1.0 and pofd[0] == 1.0  # k=0: forecast yes always
        assert pody[-1] == 0.0 and pofd[-1] == 0.0  # k=n: forecast yes never

    def test_no_skill_auc_near_half(self):
        # Same proportion of obs-yes and obs-no in every bin -> diagonal ROC.
        row = make_row({1: (10, 10), 2: (10, 10), 3: (10, 10)})
        _, _, auc = Roc(make_plot_config([]))._roc_points(row)
        assert auc == pytest.approx(0.5, abs=1e-9)

    def test_known_two_bin_auc(self):
        # 2 bins: bin1 has all the on's, bin2 has all the oy's -- POFD/PODY
        # sweep is (1,1) -> (0,1) -> (0,0), enclosing exactly half the box
        # on one side: trapezoid area = 0.5*1*(1+0) + 0.5*1*(0+... let's just
        # hand-verify via the trapezoid formula directly.
        row = make_row({1: (0, 50), 2: (50, 0)})
        pofd, pody, auc = Roc(make_plot_config([]))._roc_points(row)
        # k=0: (1,1); k=1: (0,1); k=2: (0,0)
        assert list(pofd) == pytest.approx([1.0, 0.0, 0.0])
        assert list(pody) == pytest.approx([1.0, 1.0, 0.0])
        # Cross-check against numpy's own trapezoidal integration (an
        # independent computation from the one _roc_points does internally)
        # rather than any specific function name, since np.trapz is
        # deprecated in favour of np.trapezoid as of numpy 2.x.
        trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        expected_auc = trapz_fn(pody[::-1], pofd[::-1])
        assert auc == pytest.approx(expected_auc)

    def test_variable_bin_count_30_bins(self):
        bins = {i: (10 + i, 20 - (i % 5)) for i in range(1, 31)}
        row = make_row(bins)
        pofd, pody, auc = Roc(make_plot_config([]))._roc_points(row)
        assert len(pofd) == 31  # n+1 threshold-sweep points for 30 bins
        assert 0.0 <= auc <= 1.0

    def test_non_uniform_bin_indices_still_work(self):
        # Bin numbering doesn't need to be contiguous from 1 -- bin_columns()
        # just needs the oy_/on_ columns present, sorted by index.
        row = make_row({2: (5, 20), 7: (15, 5)})
        pofd, pody, auc = Roc(make_plot_config([]))._roc_points(row)
        assert len(pofd) == 3  # 2 bins -> 3 sweep points

    def test_zero_total_oy_raises(self):
        row = make_row({1: (0, 10), 2: (0, 20)})
        with pytest.raises(Exception, match="zero obs-yes or obs-no"):
            Roc(make_plot_config([]))._roc_points(row)

    def test_missing_columns_raises(self):
        row = pd.Series({"thresh_1": 0.1, "thresh_2": 0.5})  # no oy_/on_ at all
        with pytest.raises(Exception, match="No 'oy_'/'on_' columns"):
            Roc(make_plot_config([]))._roc_points(row)


class TestAddLines:
    def _write_agg_file(self, tmp_path, lead, bins, fcst_var="REFC_ge20"):
        row = {"fcst_lead": lead, "fcst_var": fcst_var}
        for i, (oy, on) in bins.items():
            row[f"oy_{i}"] = oy
            row[f"on_{i}"] = on
        path = tmp_path / f"agg_{lead}.tsv"
        pd.DataFrame([row]).to_csv(path, sep="\t", index=False)
        return str(path)

    def test_show_auc_appends_to_label(self, tmp_path):
        f = self._write_agg_file(tmp_path, 30000, {1: (10, 90), 2: (90, 10)})
        cfg = make_plot_config([make_var_entry(30000, f)], fcst_var="REFC_ge20",
                                labels=["03"], show_auc=True)
        roc = Roc(cfg)
        roc.setup_plot()
        roc.add_lines()
        legend_label = roc.ax.get_lines()[0].get_label()
        assert legend_label.startswith("03 (AUC=")

    def test_show_auc_false_keeps_plain_label(self, tmp_path):
        f = self._write_agg_file(tmp_path, 30000, {1: (10, 90), 2: (90, 10)})
        cfg = make_plot_config([make_var_entry(30000, f)], fcst_var="REFC_ge20",
                                labels=["03"], show_auc=False)
        roc = Roc(cfg)
        roc.setup_plot()
        roc.add_lines()
        assert roc.ax.get_lines()[0].get_label() == "03"

    def test_missing_fcst_lead_raises(self, tmp_path):
        f = self._write_agg_file(tmp_path, 30000, {1: (10, 90), 2: (90, 10)})
        cfg = make_plot_config([make_var_entry(99999, f)], fcst_var="REFC_ge20", labels=["x"])
        roc = Roc(cfg)
        roc.setup_plot()
        with pytest.raises(Exception, match="No data found for fcst_lead"):
            roc.add_lines()

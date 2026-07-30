"""Unit tests for vcast.plot.reliability.Reliability.

These persist the verification done ad hoc while building the dynamic
bin-count support: the plot must read however many thresh_/oy_/on_ bin
columns are actually present in the data (not assume a fixed count of
10), plot each bin's forecast probability as the raw thresh_k value with
no derived math (confirmed against METviewer/METplotpy's own reliability
diagram convention, which plots PSTD_CALIBRATION directly against
thresh_i), and handle non-uniform bin spacing correctly since thresh_k is
read as-is regardless of spacing.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from vcast.plot.reliability import Reliability
from _helpers import make_plot_config, make_var_entry


def write_agg_row(tmp_path, lead, row, filename=None):
    row = {**row, "fcst_lead": lead}
    path = tmp_path / (filename or f"agg_{lead}.tsv")
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False)
    return str(path)


class TestAddLines:
    def test_uniform_bins_plot_thresh_k_directly(self, tmp_path):
        row = {}
        for i in range(1, 11):
            row[f"thresh_{i}"] = i / 10
            row[f"oy_{i}"] = 10 + i
            row[f"on_{i}"] = 20 - i
        f = write_agg_row(tmp_path, 30000, row)

        cfg = make_plot_config([make_var_entry(30000, f)])
        rel = Reliability(cfg)
        rel.setup_plot()
        rel.add_lines()

        xdata = list(rel.ax.lines[0].get_xdata())
        assert xdata == pytest.approx([i / 10 for i in range(1, 11)])

    def test_non_uniform_bins_no_midpoint_math(self, tmp_path):
        # Bin edges 0.0, 0.05, 0.2, 0.5, 1.0 (not evenly spaced). thresh_k
        # must be read straight from the file -- no averaging with the
        # next edge, no assuming uniform step size.
        row = {
            "thresh_1": 0.0, "thresh_2": 0.05, "thresh_3": 0.2, "thresh_4": 0.5, "thresh_n": 1.0,
            "oy_1": 5, "on_1": 45, "oy_2": 10, "on_2": 40, "oy_3": 20, "on_3": 20, "oy_4": 40, "on_4": 5,
        }
        f = write_agg_row(tmp_path, 30000, row)
        cfg = make_plot_config([make_var_entry(30000, f)])
        rel = Reliability(cfg)
        rel.setup_plot()
        rel.add_lines()

        xdata = list(rel.ax.lines[0].get_xdata())
        assert xdata == pytest.approx([0.0, 0.05, 0.2, 0.5])

    @pytest.mark.parametrize("n_bins", [2, 15, 30])
    def test_variable_bin_counts(self, tmp_path, n_bins):
        row = {}
        for i in range(1, n_bins + 1):
            row[f"thresh_{i}"] = i / (n_bins + 1)
            row[f"oy_{i}"] = 10 + i
            row[f"on_{i}"] = 20 - (i % 5)
        f = write_agg_row(tmp_path, 30000, row, filename=f"agg_{n_bins}.tsv")

        cfg = make_plot_config([make_var_entry(30000, f)])
        rel = Reliability(cfg)
        rel.setup_plot()
        rel.add_lines()

        assert len(rel.ax.lines[0].get_xdata()) == n_bins

    def test_ob_freq_is_oy_over_oy_plus_on(self, tmp_path):
        row = {"thresh_1": 0.5, "oy_1": 30, "on_1": 10}
        f = write_agg_row(tmp_path, 30000, row)
        cfg = make_plot_config([make_var_entry(30000, f)])
        rel = Reliability(cfg)
        rel.setup_plot()
        rel.add_lines()

        ydata = list(rel.ax.lines[0].get_ydata())
        assert ydata == pytest.approx([30 / (30 + 10)])

    def test_bin_with_zero_count_does_not_crash(self, tmp_path):
        # oy=on=0 for a bin -> 0/0 -> NaN, should not raise or warn-crash.
        row = {"thresh_1": 0.5, "oy_1": 0, "on_1": 0}
        f = write_agg_row(tmp_path, 30000, row)
        cfg = make_plot_config([make_var_entry(30000, f)])
        rel = Reliability(cfg)
        rel.setup_plot()
        rel.add_lines()  # should not raise
        ydata = list(rel.ax.lines[0].get_ydata())
        assert ydata[0] != ydata[0]  # NaN != NaN

    def test_missing_fcst_lead_raises(self, tmp_path):
        row = {"thresh_1": 0.5, "oy_1": 5, "on_1": 5}
        f = write_agg_row(tmp_path, 30000, row)
        cfg = make_plot_config([make_var_entry(99999, f)])
        rel = Reliability(cfg)
        rel.setup_plot()
        with pytest.raises(Exception, match="No data found for fcst_lead"):
            rel.add_lines()

    def test_no_bin_columns_raises(self, tmp_path):
        f = write_agg_row(tmp_path, 30000, {"rmse": 1.0})
        cfg = make_plot_config([make_var_entry(30000, f)])
        rel = Reliability(cfg)
        rel.setup_plot()
        with pytest.raises(Exception, match="thresh_.*oy_.*on_"):
            rel.add_lines()

    def test_multiple_series_each_get_own_line(self, tmp_path):
        row1 = {"thresh_1": 0.5, "oy_1": 5, "on_1": 5}
        row2 = {"thresh_1": 0.5, "oy_1": 8, "on_1": 2}
        f1 = write_agg_row(tmp_path, 30000, row1, filename="s1.tsv")
        f2 = write_agg_row(tmp_path, 60000, row2, filename="s2.tsv")

        cfg = make_plot_config([make_var_entry(30000, f1), make_var_entry(60000, f2)],
                                labels=["03", "06"])
        rel = Reliability(cfg)
        rel.setup_plot()
        rel.add_lines()

        assert len(rel.ax.lines) == 2
        assert rel.ax.lines[0].get_ydata()[0] == pytest.approx(0.5)
        assert rel.ax.lines[1].get_ydata()[0] == pytest.approx(0.8)

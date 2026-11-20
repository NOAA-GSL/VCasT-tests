"""Unit tests for the ``dodge`` option of vcast.plot.line_plot.LinePlot.

When ``dodge`` is enabled each plotted series (its line, markers, confidence
intervals and significance markers) is shifted slightly along the x-axis so
that overlapping lines and CIs from several series stay readable. The true x
ticks and limits are left untouched.
"""
import matplotlib
matplotlib.use("Agg")  # headless backend for tests

import numpy as np
import pandas as pd
import pytest

from vcast.plot.line_plot import LinePlot
from _helpers import make_plot_config, make_var_entry


def _make_line_plot(n_series, dodge=False, dodge_width=0.6, unique=None):
    """Build a LinePlot with a minimal config for ``n_series`` series."""
    vars_list = [make_var_entry(f"v{i}", f"/tmp/unused{i}.data") for i in range(n_series)]
    config = make_plot_config(
        vars_list,
        unique=unique,
        dodge=dodge,
        dodge_width=dodge_width,
    )
    return LinePlot(config)


class TestCountSeries:
    def test_counts_vars_without_unique(self):
        assert _make_line_plot(3)._count_series() == 3

    def test_counts_vars_plus_unique_grouping(self):
        # add_lines indexes grouped series as i + j, so the max index is
        # len(vars) + len(unique) - 1.
        unique = [make_var_entry("model", "A"), make_var_entry("model", "B")]
        plot = _make_line_plot(3, unique=unique)
        assert plot._count_series() == 3 + 2 - 1


class TestDodgeOffset:
    def test_disabled_returns_zero(self):
        plot = _make_line_plot(3, dodge=False)
        assert plot._dodge_offset(0, [0, 6, 12]) == 0.0
        assert plot._dodge_offset(2, [0, 6, 12]) == 0.0

    def test_single_series_returns_zero(self):
        plot = _make_line_plot(1, dodge=True)
        assert plot._dodge_offset(0, [0, 6, 12]) == 0.0

    def test_offsets_are_symmetric_about_zero(self):
        plot = _make_line_plot(3, dodge=True)
        offsets = [plot._dodge_offset(i, [0, 6, 12]) for i in range(3)]
        assert offsets[1] == pytest.approx(0.0)          # centre series unshifted
        assert offsets[0] == pytest.approx(-offsets[2])  # outer series mirror each other
        assert sum(offsets) == pytest.approx(0.0)

    def test_offsets_are_ordered_and_distinct(self):
        plot = _make_line_plot(4, dodge=True)
        offsets = [plot._dodge_offset(i, [0, 6, 12]) for i in range(4)]
        assert offsets == sorted(offsets)
        assert len(set(offsets)) == 4

    def test_span_scales_with_dodge_width_and_x_spacing(self):
        # Total spread between the outermost series == dodge_width * median dx.
        plot = _make_line_plot(2, dodge=True, dodge_width=0.6)
        x = [0, 6, 12]  # median spacing dx = 6
        span = plot._dodge_offset(1, x) - plot._dodge_offset(0, x)
        assert span == pytest.approx(0.6 * 6)

    def test_offset_adapts_to_date_like_spacing(self):
        plot = _make_line_plot(2, dodge=True, dodge_width=0.5)
        x = [100.0, 101.0, 102.0]  # e.g. matplotlib date numbers, dx = 1
        span = plot._dodge_offset(1, x) - plot._dodge_offset(0, x)
        assert span == pytest.approx(0.5 * 1.0)

    def test_handles_nan_and_unsorted_x(self):
        plot = _make_line_plot(2, dodge=True, dodge_width=0.6)
        x = [12, np.nan, 0, 6]  # dx still resolves to 6 after dropna/sort/unique
        span = plot._dodge_offset(1, x) - plot._dodge_offset(0, x)
        assert span == pytest.approx(0.6 * 6)


class TestDodgeAppliedToPlot:
    def _write_series(self, tmp_path, name, leads, values):
        path = tmp_path / f"{name}.data"
        pd.DataFrame({"fcst_lead": leads, name: values}).to_csv(path, sep="\t", index=False)
        return path

    def _config(self, tmp_path, dodge, dodge_width=0.6):
        leads = [0, 6, 12]
        f0 = self._write_series(tmp_path, "v0", leads, [1.0, 2.0, 3.0])
        f1 = self._write_series(tmp_path, "v1", leads, [1.5, 2.5, 3.5])
        vars_list = [make_var_entry("v0", str(f0)), make_var_entry("v1", str(f1))]
        return make_plot_config(
            vars_list,
            unique=None,
            fcst_var=None,
            start_date="2024-01-01_00:00:00",
            end_date="2024-01-02_00:00:00",
            interval_hours="6",
            x_label="lead",
            y_label="val",
            ylim=None,
            yticks=None,
            xlim=None,
            xticks=None,
            dodge=dodge,
            dodge_width=dodge_width,
        )

    def test_series_are_shifted_when_dodge_enabled(self, tmp_path):
        plot = LinePlot(self._config(tmp_path, dodge=True))
        plot.setup_plot()
        plot.add_lines()

        true_x = np.array([0, 6, 12], dtype=float)
        # dodge_width 0.6, dx 6, 2 series -> outer series at +/- 1.8
        line0_x = plot.ax.lines[0].get_xdata()
        line1_x = plot.ax.lines[1].get_xdata()
        np.testing.assert_allclose(line0_x, true_x - 1.8)
        np.testing.assert_allclose(line1_x, true_x + 1.8)

    def test_series_unchanged_when_dodge_disabled(self, tmp_path):
        plot = LinePlot(self._config(tmp_path, dodge=False))
        plot.setup_plot()
        plot.add_lines()

        true_x = np.array([0, 6, 12], dtype=float)
        np.testing.assert_allclose(plot.ax.lines[0].get_xdata(), true_x)
        np.testing.assert_allclose(plot.ax.lines[1].get_xdata(), true_x)

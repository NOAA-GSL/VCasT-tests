"""Unit tests for vcast.plot.rank_histogram.RankHistogram.

The plot sums the rank_i counts of every selected row per series and shows
relative frequencies, with a flat 1/N reference line. Rows are selected by
fcst_var, `filters` (numeric columns compared as numbers, so "060000"
matches 60000) and `unique` (one series per item).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from vcast.plot.rank_histogram import RankHistogram
from _helpers import make_plot_config, make_var_entry


def write_rhist(tmp_path, rows, name="agg.tsv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return str(path)


def row(mask, lead, counts, var="TMP"):
    r = {"model": "HRRRCast", "fcst_var": var, "vx_mask": mask, "fcst_lead": lead,
         "total": sum(counts)}
    r.update({f"rank_{i}": c for i, c in enumerate(counts, start=1)})
    return r


@pytest.fixture
def agg_file(tmp_path):
    return write_rhist(tmp_path, [
        row("CONUS", 60000, [3, 1, 1, 3]),
        row("CONUS", 120000, [1, 1, 1, 1]),
        row("WEST", 60000, [0, 2, 2, 0]),
        row("CONUS", 60000, [9, 9, 9, 9], var="DPT"),
    ])


def test_sums_rows_and_normalises(agg_file):
    cfg = make_plot_config([make_var_entry("rank", agg_file)], fcst_var="TMP",
                           unique=[SimpleNamespace(vx_mask="CONUS")])
    ranks, freq = RankHistogram(cfg).series_frequencies()[0]
    assert ranks.tolist() == [1, 2, 3, 4]
    # CONUS TMP rows summed over both leads: [4, 2, 2, 4] / 12
    assert freq == pytest.approx(np.array([4, 2, 2, 4]) / 12)


def test_filters_match_zero_padded_leads(agg_file):
    cfg = make_plot_config([make_var_entry("rank", agg_file)], fcst_var="TMP",
                           unique=[SimpleNamespace(vx_mask="CONUS")],
                           filters={"fcst_lead": ["060000"]})
    _, freq = RankHistogram(cfg).series_frequencies()[0]
    assert freq == pytest.approx(np.array([3, 1, 1, 3]) / 8)


def test_one_series_per_unique_item(agg_file):
    cfg = make_plot_config([make_var_entry("rank", agg_file)], fcst_var="TMP",
                           unique=[SimpleNamespace(vx_mask="CONUS"), SimpleNamespace(vx_mask="WEST")],
                           line_color=["black", "blue"], labels=["CONUS", "WEST"],
                           filters={"fcst_lead": [60000]})
    series = RankHistogram(cfg).series_frequencies()
    assert len(series) == 2
    assert series[1][1] == pytest.approx([0, 0.5, 0.5, 0])


def test_bars_and_uniform_line(agg_file):
    cfg = make_plot_config([make_var_entry("rank", agg_file)], fcst_var="TMP",
                           unique=[SimpleNamespace(vx_mask="CONUS"), SimpleNamespace(vx_mask="WEST")],
                           line_color=["black", "blue"], labels=["CONUS", "WEST"])
    rh = RankHistogram(cfg)
    rh.setup_plot()
    rh.add_bars()
    assert len(rh.ax.patches) == 2 * 4                     # 4 ranks x 2 series
    flat = [l for l in rh.ax.lines if l.get_label() == "Perfectly calibrated"]
    assert flat and flat[0].get_ydata()[0] == pytest.approx(0.25)


def test_step_style(agg_file):
    cfg = make_plot_config([make_var_entry("rank", agg_file)], fcst_var="TMP", style="step",
                           unique=[SimpleNamespace(vx_mask="CONUS")])
    rh = RankHistogram(cfg)
    rh.setup_plot()
    rh.add_bars()
    assert len(rh.ax.patches) == 0
    assert any(l.get_label() == "series0" for l in rh.ax.lines)


def test_missing_rank_columns_raise(tmp_path):
    f = write_rhist(tmp_path, [{"fcst_var": "TMP", "rmse": 1.0}])
    cfg = make_plot_config([make_var_entry("rank", f)])
    with pytest.raises(Exception, match="rank_"):
        RankHistogram(cfg).series_frequencies()


def test_zero_counts_raise(tmp_path):
    f = write_rhist(tmp_path, [row("CONUS", 60000, [0, 0, 0])])
    cfg = make_plot_config([make_var_entry("rank", f)])
    with pytest.raises(Exception, match="zero"):
        RankHistogram(cfg).series_frequencies()


def test_end_to_end_saves_png(agg_file, tmp_path):
    out = tmp_path / "rh.png"
    cfg = make_plot_config([make_var_entry("rank", agg_file)], fcst_var="TMP",
                           unique=[SimpleNamespace(vx_mask="CONUS")], output_filename=str(out),
                           legend=True)
    RankHistogram(cfg).plot()
    assert out.exists() and out.stat().st_size > 0

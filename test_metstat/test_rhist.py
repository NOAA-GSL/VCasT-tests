"""Unit tests for RHIST (rank histogram) support in ReadStat and Aggregation.

MET's RHIST line carries TOTAL, N_RANK and then one count per rank
(N_RANK = ensemble size + 1). ReadStat must turn those into rank_1 ...
rank_<N_RANK> columns (like the dynamic PCT bins), the "all_ranks" stat_vars
keyword must expand to all of them, and aggregation must SUM the counts per
group (averaging counts would be wrong) unless agg_method says otherwise.
"""
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vcast.metstat.stat_handler import ReadStat
from vcast.metstat.constants import FULL_HEADER, AVAILABLE_LINE_TYPES
from vcast.agg import Aggregation


def make_rhist_line(counts, fcst_lead="120000", date="20230101_120000",
                    fcst_var="TMP", vx_mask="CONUS", model="HRRRCast"):
    base = {h: "NA" for h in FULL_HEADER}
    base.update({
        "version": "V12.0.1", "model": model, "desc": "NA",
        "fcst_lead": fcst_lead, "fcst_valid_beg": date, "fcst_valid_end": date,
        "obs_lead": "000000", "obs_valid_beg": date, "obs_valid_end": date,
        "fcst_var": fcst_var, "vx_mask": vx_mask, "line_type": "RHIST",
    })
    row = [base[h] for h in FULL_HEADER]
    row += [str(sum(counts)), str(len(counts))] + [str(c) for c in counts]
    return " ".join(row)


def write_stat_file(tmpdir, filename, lines):
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        f.write("HEADER_LINE_PLACEHOLDER\n")
        f.write("\n".join(lines) + "\n")
    return path


def make_config(input_stat_folder, **overrides):
    defaults = dict(
        line_type="rhist",
        input_stat_folder=input_stat_folder,
        date_column="fcst_valid_beg",
        start_date="2023-01-01_00:00:00",
        end_date="2023-01-02_00:00:00",
        thresholds=None,
        string_filters={"vx_mask": ["CONUS", "WEST"]},
        columns_to_keep=None,
        reformat_file=False,
        output_reformat_file=None,
        stat_vars=["total", "all_ranks"],
        output_file=False,
        output_plot_file=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_rhist_is_a_known_line_type():
    assert "rhist" in AVAILABLE_LINE_TYPES


def test_process_file_creates_rank_columns():
    counts = list(range(1, 20))                      # 18 members -> 19 ranks
    with tempfile.TemporaryDirectory() as tmp:
        path = write_stat_file(tmp, "a.stat", [make_rhist_line(counts)])
        rs = ReadStat(make_config(tmp))
        df = rs.process_file(path, "rhist")
    assert df.shape[0] == 1
    assert [f"rank_{i}" for i in range(1, 20)] == [c for c in df.columns if c.startswith("rank_")]
    assert int(df["n_rank"].iloc[0]) == 19
    assert int(df["rank_19"].iloc[0]) == 19


def test_other_line_types_in_file_are_ignored():
    counts = [5] * 19
    ecnt = make_rhist_line(counts).replace(" RHIST ", " ECNT ")
    with tempfile.TemporaryDirectory() as tmp:
        path = write_stat_file(tmp, "a.stat", [make_rhist_line(counts), ecnt])
        df = ReadStat(make_config(tmp)).process_file(path, "rhist")
    assert df.shape[0] == 1


def test_run_all_expands_all_ranks_and_keeps_total():
    with tempfile.TemporaryDirectory() as tmp:
        write_stat_file(tmp, "a.stat", [make_rhist_line([1] * 19, vx_mask="CONUS"),
                                        make_rhist_line([2] * 19, vx_mask="WEST")])
        df, add_columns, svars = ReadStat(make_config(tmp)).run_all()
    assert svars[0] == "total"
    assert svars[1:] == [f"rank_{i}" for i in range(1, 20)]
    assert "all_ranks" not in svars
    assert sorted(df["rank_1"].tolist()) == [1, 2]
    assert pd.api.types.is_numeric_dtype(df["rank_5"])


class TestAggregationMethod:
    def _df(self):
        return pd.DataFrame({
            "fcst_lead": [60000, 60000, 120000],
            "total": [10.0, 30.0, 5.0],
            "rank_1": [4.0, 6.0, 1.0],
            "rank_2": [6.0, 24.0, 4.0],
        })

    def test_rhist_defaults_to_sum(self):
        cfg = SimpleNamespace(group_by=["fcst_lead"], output_agg_file=None, line_type="rhist", ci=True)
        out = Aggregation(cfg, self._df(), ["total", "rank_1", "rank_2"]).run()
        r6 = out[out["fcst_lead"] == 60000].iloc[0]
        assert r6["total"] == 40 and r6["rank_1"] == 10 and r6["rank_2"] == 30
        assert r6["count"] == 2
        assert "rank_1_bcl" not in out.columns            # no CI on summed counts

    def test_other_line_types_still_average(self):
        cfg = SimpleNamespace(group_by=["fcst_lead"], output_agg_file=None, line_type="ecnt", ci=False)
        out = Aggregation(cfg, self._df(), ["total"]).run()
        assert out[out["fcst_lead"] == 60000].iloc[0]["total"] == pytest.approx(20.0)

    def test_explicit_agg_method_wins(self):
        cfg = SimpleNamespace(group_by=["fcst_lead"], output_agg_file=None, line_type="rhist",
                              agg_method="mean", ci=False)
        out = Aggregation(cfg, self._df(), ["total"]).run()
        assert out[out["fcst_lead"] == 60000].iloc[0]["total"] == pytest.approx(20.0)

    def test_bad_agg_method_raises(self):
        cfg = SimpleNamespace(group_by=["fcst_lead"], output_agg_file=None, agg_method="median")
        with pytest.raises(ValueError):
            Aggregation(cfg, self._df(), ["total"])

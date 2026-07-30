"""Unit tests for vcast.metstat.stat_handler.ReadStat.

These focus on the PCT/PSTD bin-column handling: MET's probability
contingency table (PCT) line type carries a variable number of
probability thresholds (thresh_i/oy_i/on_i), and different rows or
files in the same run can legitimately have different bin counts (e.g.
different dates, forecast variables, or lead times configured with a
different number of thresholds). ReadStat.process_file()/update_headers()
must keep every bin seen anywhere in the run, not just whichever row
happened to be parsed last -- see the "mixed bin counts" tests below,
which are regression tests for exactly that bug.
"""
import os
import tempfile
from types import SimpleNamespace

import pandas as pd
import pytest

from vcast.metstat.stat_handler import ReadStat
from vcast.metstat.constants import FULL_HEADER


def make_pct_line(n_thresh, fcst_lead="030000", date="20240401_000000",
                   fcst_var="APCP", model="HRRR"):
    """Build one raw MET .stat PCT line with `n_thresh` threshold edges
    (n_thresh - 1 probability bins), matching the column layout ReadStat
    expects: FULL_HEADER fields, then TOTAL, N_THRESH, then
    (thresh_i, oy_i, on_i) for i in 1..n_thresh-1, then a trailing
    thresh_n column holding the final (upper) threshold edge.
    """
    base = {h: "NA" for h in FULL_HEADER}
    base.update({
        "version": "V11.0", "model": model, "desc": "NA",
        "fcst_lead": fcst_lead, "fcst_valid_beg": date, "fcst_valid_end": date,
        "obs_lead": "0", "obs_valid_beg": date, "obs_valid_end": date,
        "fcst_var": fcst_var, "line_type": "PCT",
    })
    row = [base[h] for h in FULL_HEADER]
    row += ["1000", str(n_thresh)]
    for i in range(1, n_thresh):
        row += [f"{i / n_thresh:.4f}", str(10 + i), str(20 + i)]
    row += ["1.0000"]
    return " ".join(row)


def write_stat_file(tmpdir, filename, lines):
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        f.write("HEADER_LINE_PLACEHOLDER\n")
        f.write("\n".join(lines) + "\n")
    return path


def make_config(input_stat_folder, **overrides):
    defaults = dict(
        line_type="pct",
        input_stat_folder=input_stat_folder,
        date_column="fcst_valid_beg",
        start_date="2024-04-01_00:00:00",
        end_date="2024-04-02_00:00:00",
        thresholds=None,
        string_filters={},
        columns_to_keep=None,
        reformat_file=False,
        output_reformat_file=None,
        stat_vars=["all_thresh"],
        output_file=False,
        output_plot_file=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBinColumnSortKey:
    def test_numeric_order_not_lexicographic(self):
        cols = ["thresh_10", "thresh_2", "oy_1", "on_1", "thresh_1", "thresh_n", "oy_10"]
        ordered = sorted(cols, key=ReadStat._bin_column_sort_key)
        # Grouped by bin index (1, 2, 10, ...) not lexicographically (which
        # would put thresh_10 right after thresh_1), and within an index,
        # thresh_i sorts before oy_i/on_i.
        assert ordered == ["thresh_1", "oy_1", "on_1", "thresh_2", "thresh_10", "oy_10", "thresh_n"]

    def test_thresh_n_sorts_last(self):
        cols = ["thresh_n", "thresh_5", "on_5"]
        ordered = sorted(cols, key=ReadStat._bin_column_sort_key)
        assert ordered[-1] == "thresh_n"


class TestUpdateHeadersAccumulation:
    def test_accumulates_union_across_calls(self):
        rs = ReadStat(make_config("/tmp"))
        headers = rs.all_columns("pct")

        row_10bins = ["x"] * (len(FULL_HEADER)) + ["1000", "11"]
        rs.update_headers(headers, row_10bins, "pct")
        assert "thresh_10" in rs.column_specific
        assert "thresh_11" not in rs.column_specific  # only 10 bins (n_thresh=11 -> i in 1..10)

        row_30bins = ["x"] * (len(FULL_HEADER)) + ["1000", "31"]
        rs.update_headers(headers, row_30bins, "pct")

        # Union of both: bins 1-10 (from the first row) AND 1-30 (from the second)
        for i in list(range(1, 11)) + list(range(1, 31)):
            assert f"thresh_{i}" in rs.column_specific
            assert f"oy_{i}" in rs.column_specific
            assert f"on_{i}" in rs.column_specific
        assert "thresh_n" in rs.column_specific

    def test_column_specific_stays_numerically_sorted(self):
        rs = ReadStat(make_config("/tmp"))
        headers = rs.all_columns("pct")
        row = ["x"] * len(FULL_HEADER) + ["1000", "13"]
        rs.update_headers(headers, row, "pct")

        thresh_cols = [c for c in rs.column_specific if c.startswith("thresh_") and c != "thresh_n"]
        indices = [int(c.split("_")[1]) for c in thresh_cols]
        assert indices == sorted(indices)


class TestProcessFileMixedBinCounts:
    """Regression tests for the bug where a later-processed row with fewer
    bins silently dropped higher-numbered bins seen in an earlier row."""

    def test_more_bins_first_then_fewer_bins(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                make_pct_line(31, fcst_lead="030000", date="20240401_000000"),  # 30 bins
                make_pct_line(11, fcst_lead="030000", date="20240401_010000"),  # 10 bins, processed last
            ]
            path = write_stat_file(tmp, "test.stat", lines)

            rs = ReadStat(make_config(tmp))
            df = rs.process_file(path, "pct")

            assert "thresh_25" in df.columns
            assert "oy_25" in df.columns
            assert "on_25" in df.columns
            # Row 0 (31 thresholds) should have real values for bin 25
            assert df.loc[0, "oy_25"] == "35"
            # Row 1 (11 thresholds) never had bin 25 -- it just wasn't collected
            assert pd.isna(df.loc[1, "oy_25"]) or df.loc[1, "oy_25"] is None

    def test_fewer_bins_first_then_more_bins(self):
        """Same scenario, reversed processing order -- must still keep every bin."""
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                make_pct_line(11, fcst_lead="030000", date="20240401_000000"),  # 10 bins, processed first
                make_pct_line(31, fcst_lead="030000", date="20240401_010000"),  # 30 bins, processed last
            ]
            path = write_stat_file(tmp, "test.stat", lines)

            rs = ReadStat(make_config(tmp))
            df = rs.process_file(path, "pct")

            assert "thresh_25" in df.columns
            assert df.loc[1, "oy_25"] == "35"

    def test_mixed_bins_across_two_files(self):
        """The same bug, but the mismatched rows are in separate files
        combined by run_all() rather than lines within a single file."""
        with tempfile.TemporaryDirectory() as tmp:
            write_stat_file(tmp, "a_first.stat", [make_pct_line(31, date="20240401_000000")])
            write_stat_file(tmp, "b_second.stat", [make_pct_line(11, date="20240401_010000")])

            cfg = make_config(
                tmp,
                string_filters={"fcst_lead": ["030000"]},
                end_date="2024-04-02_00:00:00",
            )
            rs = ReadStat(cfg)
            df, add_columns, svars = rs.run_all()

            assert "thresh_25" in svars
            assert "oy_25" in svars
            assert "thresh_25" in df.columns
            # One row (from a_first.stat) actually has bin 25 data
            assert df["oy_25"].notna().sum() == 1


class TestProcessFileUniformBins:
    def test_single_row_all_bins_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_stat_file(tmp, "test.stat", [make_pct_line(11)])
            rs = ReadStat(make_config(tmp))
            df = rs.process_file(path, "pct")

            for i in range(1, 11):
                assert f"thresh_{i}" in df.columns
                assert f"oy_{i}" in df.columns
                assert f"on_{i}" in df.columns
            assert "thresh_11" not in df.columns  # 11 thresholds -> bins 1..10 only
            assert "thresh_n" in df.columns
            assert df.loc[0, "thresh_n"] == "1.0000"


class TestFilterHelpers:
    def _sample_df(self):
        return pd.DataFrame({
            "fcst_valid_beg": ["20240401_000000", "20240401_060000", "20240402_120000"],
            "model": ["A", "B", "A"],
            "rmse": ["1.0", "2.5", "9.9"],
        })

    def test_filter_by_date_keeps_only_in_range(self):
        rs = ReadStat(make_config("/tmp"))
        df = self._sample_df()
        out = rs.filter_by_date(df, "fcst_valid_beg", "2024-04-01_00:00:00", "2024-04-01_12:00:00")
        assert len(out) == 2

    def test_filter_by_date_empty_raises(self):
        rs = ReadStat(make_config("/tmp"))
        df = self._sample_df()
        with pytest.raises(RuntimeError):
            rs.filter_by_date(df, "fcst_valid_beg", "2030-01-01_00:00:00", "2030-01-02_00:00:00")

    def test_filter_by_string_keeps_allowed_values(self):
        rs = ReadStat(make_config("/tmp"))
        df = self._sample_df()
        out = rs.filter_by_string(df, {"model": ["A"]})
        assert set(out["model"]) == {"A"}
        assert len(out) == 2

    def test_filter_by_string_missing_column_is_skipped_not_fatal(self):
        rs = ReadStat(make_config("/tmp"))
        df = self._sample_df()
        # "nonexistent" isn't a column; should log+skip rather than crash,
        # and other filters still apply.
        out = rs.filter_by_string(df, {"nonexistent": ["x"], "model": ["A"]})
        assert set(out["model"]) == {"A"}

    def test_filter_by_threshold_range(self):
        rs = ReadStat(make_config("/tmp"))
        df = self._sample_df()
        out = rs.filter_by_threshold(df, {"rmse": (0.5, 3.0)})
        assert len(out) == 2
        assert "9.9" not in out["rmse"].values

    def test_filter_by_columns_keeps_only_requested(self):
        rs = ReadStat(make_config("/tmp"))
        df = self._sample_df()
        out = rs.filter_by_columns(df, ["model", "rmse"])
        assert list(out.columns) == ["model", "rmse"]

    def test_filter_by_columns_none_requested_returns_unchanged(self):
        rs = ReadStat(make_config("/tmp"))
        df = self._sample_df()
        out = rs.filter_by_columns(df, [])
        assert list(out.columns) == list(df.columns)

    def test_filter_by_columns_no_matches_raises(self):
        rs = ReadStat(make_config("/tmp"))
        df = self._sample_df()
        with pytest.raises(RuntimeError):
            rs.filter_by_columns(df, ["nope_not_a_column"])

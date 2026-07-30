"""Unit tests for vcast.agg.agg.Aggregation.

Focused on the nanmean/CI behavior: when a group mixes rows that do and
don't have a given column (e.g. PCT bin columns that only exist for some
dates because of a non-uniform number of thresholds), the aggregate for
that column must be computed over the rows that actually have it, not
collapse to NaN just because *some* row in the group is missing it.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from vcast.agg.agg import Aggregation


def make_config(**overrides):
    defaults = dict(group_by=["fcst_lead"], stats=["rmse"], output_agg_file="/tmp/out.data", ci=False)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestMeanWithPartialColumns:
    def test_ignores_nan_from_missing_bins(self):
        df = pd.DataFrame({
            "fcst_lead": ["030000", "030000", "030000"],
            "oy_25": [10.0, np.nan, 20.0],  # bin 25 missing for the middle row
        })
        agg = Aggregation(make_config(stats=["oy_25"]), df=df)
        out = agg.run()

        assert out.loc[0, "oy_25"] == pytest.approx(15.0)  # mean of 10 and 20, not NaN-poisoned

    def test_all_missing_in_group_gives_nan_not_error(self):
        df = pd.DataFrame({
            "fcst_lead": ["030000", "030000"],
            "oy_30": [np.nan, np.nan],
        })
        agg = Aggregation(make_config(stats=["oy_30"]), df=df)
        out = agg.run()

        assert np.isnan(out.loc[0, "oy_30"])

    def test_group_with_no_missing_values_unaffected(self):
        df = pd.DataFrame({
            "fcst_lead": ["030000", "030000", "060000", "060000"],
            "rmse": [1.0, 3.0, 2.0, 4.0],
        })
        agg = Aggregation(make_config(stats=["rmse"]), df=df)
        out = agg.run()

        row_03 = out[out["fcst_lead"] == "030000"].iloc[0]
        row_06 = out[out["fcst_lead"] == "060000"].iloc[0]
        assert row_03["rmse"] == pytest.approx(2.0)
        assert row_06["rmse"] == pytest.approx(3.0)


class TestConfidenceIntervals:
    def test_ci_disabled_by_default_no_bcl_bcu_columns(self):
        df = pd.DataFrame({"fcst_lead": ["030000", "030000"], "rmse": [1.0, 2.0]})
        agg = Aggregation(make_config(stats=["rmse"], ci=False), df=df)
        out = agg.run()
        assert "rmse_bcl" not in out.columns
        assert "rmse_bcu" not in out.columns

    def test_ci_uses_valid_count_not_group_size(self):
        # 3 rows in the group, but only 2 have a value for this column --
        # the CI's degrees of freedom must come from the 2 valid values,
        # not the group's total row count of 3.
        df = pd.DataFrame({
            "fcst_lead": ["030000", "030000", "030000"],
            "oy_25": [10.0, np.nan, 20.0],
        })
        agg = Aggregation(make_config(stats=["oy_25"], ci=True), df=df)
        out = agg.run()

        valid = np.array([10.0, 20.0])
        mean_val = valid.mean()
        s = valid.std(ddof=1)
        t_crit = student_t.ppf(1 - 0.05 / 2, df=1)  # n_valid=2 -> df=1
        half_width = t_crit * s / np.sqrt(2)

        assert out.loc[0, "oy_25"] == pytest.approx(mean_val)
        assert out.loc[0, "oy_25_bcl"] == pytest.approx(mean_val - half_width)
        assert out.loc[0, "oy_25_bcu"] == pytest.approx(mean_val + half_width)

    def test_ci_single_valid_value_gives_zero_width(self):
        df = pd.DataFrame({
            "fcst_lead": ["030000", "030000"],
            "oy_25": [10.0, np.nan],
        })
        agg = Aggregation(make_config(stats=["oy_25"], ci=True), df=df)
        out = agg.run()

        assert out.loc[0, "oy_25"] == pytest.approx(10.0)
        assert out.loc[0, "oy_25_bcl"] == pytest.approx(10.0)
        assert out.loc[0, "oy_25_bcu"] == pytest.approx(10.0)


class TestGroupingAndPlumbing:
    def test_multiple_group_by_columns(self):
        df = pd.DataFrame({
            "model": ["A", "A", "B", "B"],
            "fcst_lead": ["030000", "030000", "030000", "030000"],
            "rmse": [1.0, 3.0, 5.0, 7.0],
        })
        agg = Aggregation(make_config(group_by=["model", "fcst_lead"], stats=["rmse"]), df=df)
        out = agg.run()

        assert set(out["model"]) == {"A", "B"}
        a_row = out[out["model"] == "A"].iloc[0]
        b_row = out[out["model"] == "B"].iloc[0]
        assert a_row["rmse"] == pytest.approx(2.0)
        assert b_row["rmse"] == pytest.approx(6.0)

    def test_count_column_reflects_group_size(self):
        df = pd.DataFrame({
            "fcst_lead": ["030000", "030000", "030000"],
            "rmse": [1.0, 2.0, 3.0],
        })
        agg = Aggregation(make_config(stats=["rmse"]), df=df)
        out = agg.run()
        assert out.loc[0, "count"] == 3

    def test_explicit_df_and_stats_override_config(self):
        # This mirrors how cli.py's handle_conversion calls Aggregation:
        # passing df/stats directly rather than config.input_file/config.stats.
        df = pd.DataFrame({"fcst_lead": ["030000"], "custom_col": [42.0]})
        cfg = make_config(stats=["rmse"])  # config.stats deliberately wrong/unused here
        agg = Aggregation(cfg, df=df, stats=["custom_col"])
        out = agg.run()
        assert "custom_col" in out.columns
        assert out.loc[0, "custom_col"] == pytest.approx(42.0)

    def test_save_output_writes_tsv(self, tmp_path):
        df = pd.DataFrame({"fcst_lead": ["030000"], "rmse": [1.0]})
        out_file = tmp_path / "agg_out.data"
        agg = Aggregation(make_config(stats=["rmse"], output_agg_file=str(out_file)), df=df)
        agg.run()
        agg.save_output()

        assert out_file.exists()
        reloaded = pd.read_csv(out_file, sep="\t", index_col=0)
        assert reloaded.loc[0, "rmse"] == pytest.approx(1.0)

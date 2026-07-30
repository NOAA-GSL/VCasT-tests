"""Unit tests for vcast.plot.base_plot.BasePlot.bin_columns, the helper
Reliability and Roc both use to discover PCT bin columns dynamically."""
from vcast.plot.base_plot import BasePlot


class TestBinColumns:
    def test_finds_matching_columns_in_numeric_order(self):
        columns = ["oy_10", "oy_2", "oy_1", "date", "fcst_lead"]
        assert BasePlot.bin_columns(columns, "oy_") == ["oy_1", "oy_2", "oy_10"]

    def test_excludes_non_numeric_suffix(self):
        # thresh_n is a real column in PCT output (the final upper edge)
        # but must not be picked up as a numbered bin.
        columns = ["thresh_1", "thresh_2", "thresh_n"]
        assert BasePlot.bin_columns(columns, "thresh_") == ["thresh_1", "thresh_2"]

    def test_excludes_columns_with_different_prefix(self):
        columns = ["oy_1", "on_1", "thresh_1"]
        assert BasePlot.bin_columns(columns, "oy_") == ["oy_1"]

    def test_no_matches_returns_empty_list(self):
        assert BasePlot.bin_columns(["date", "fcst_lead"], "oy_") == []

    def test_works_on_pandas_index(self):
        import pandas as pd
        idx = pd.Index(["oy_3", "oy_1", "oy_2"])
        assert BasePlot.bin_columns(idx, "oy_") == ["oy_1", "oy_2", "oy_3"]

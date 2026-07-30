"""Shared helpers for vcast.plot unit tests (not collected by pytest itself)."""
from types import SimpleNamespace

from vcast.io.config_loader import ConfigObject


def make_var_entry(lead, file_path):
    """Build one `vars:` list entry the way ConfigLoader actually would --
    a lead-time key (often an int, e.g. `30000: "./agg.data"` in YAML)
    mapped to a file path. Real config YAML lead keys are frequently
    integers, and vcast.io.ConfigObject stores them via `__dict__.update()`
    (bypassing normal attribute-name rules), which `SimpleNamespace(**kwargs)`
    can't do since it requires string keyword names."""
    return ConfigObject({lead: file_path})


def make_plot_config(vars_list, **overrides):
    """Build a minimal config object satisfying what Reliability/Roc/BasePlot
    read off `self.config`. `vars_list` is the actual list of SimpleNamespace
    entries (one per plotted series, each mapping a lead-time key to a tsv
    file path) -- everything else gets a same-length default that callers
    can override."""
    n_series = len(vars_list)
    defaults = dict(
        vars=vars_list,
        fcst_var=None,
        unique=None,
        line_color=["blue"] * n_series,
        line_marker=["o"] * n_series,
        line_type=["-"] * n_series,
        line_width=[1] * n_series,
        labels=[f"series{i}" for i in range(n_series)],
        plot_title="test",
        grid=False,
        legend=False,
        legend_title="",
        show_auc=True,
        output_filename="/tmp/unused.png",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)

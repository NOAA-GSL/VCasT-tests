"""Unit tests for vcast.cli.detect_yaml_config, the pure function that
routes every vcast invocation to convert/plot/stats/agg/sig based on
which required keys a YAML config file has. A bug here silently
misroutes every command, so it's worth covering directly even though
it's trivial."""
import yaml
import pytest

from vcast.cli import detect_yaml_config


def write_yaml(tmp_path, name, content):
    path = tmp_path / name
    with open(path, "w") as f:
        yaml.safe_dump(content, f)
    return str(path)


def test_convert_config_detected(tmp_path):
    f = write_yaml(tmp_path, "c.yaml", {
        "input_stat_folder": "./stats", "line_type": "pct",
        "date_column": "fcst_valid_beg", "output_file": True,
    })
    assert detect_yaml_config(f) == "convert"


def test_plot_config_detected(tmp_path):
    f = write_yaml(tmp_path, "p.yaml", {
        "plot_type": "reliability", "vars": [], "output_filename": "out.png",
    })
    assert detect_yaml_config(f) == "plot"


def test_stats_config_detected(tmp_path):
    f = write_yaml(tmp_path, "s.yaml", {
        "stat_name": ["rmse"], "fcst_file_template": "f.grib2", "ref_file_template": "r.grib2",
    })
    assert detect_yaml_config(f) == "stats"


def test_agg_config_detected(tmp_path):
    f = write_yaml(tmp_path, "a.yaml", {
        "input_file": "in.data", "group_by": ["fcst_lead"], "output_agg_file": "out.data",
    })
    assert detect_yaml_config(f) == "agg"


def test_sig_config_detected(tmp_path):
    f = write_yaml(tmp_path, "sig.yaml", {
        "input_model_A": "a.data", "input_model_B": "b.data", "output_file": "out.data",
    })
    assert detect_yaml_config(f) == "sig"


def test_unrecognized_config_returns_none(tmp_path):
    f = write_yaml(tmp_path, "bad.yaml", {"some_key": "some_value"})
    assert detect_yaml_config(f) is None


def test_missing_required_key_falls_back_to_none(tmp_path):
    # Has plot_type and vars but not output_filename -- should not be
    # mistaken for a plot config.
    f = write_yaml(tmp_path, "incomplete.yaml", {"plot_type": "reliability", "vars": []})
    assert detect_yaml_config(f) is None


def test_nonexistent_file_returns_none_not_raise(tmp_path):
    assert detect_yaml_config(str(tmp_path / "does_not_exist.yaml")) is None


def test_non_dict_yaml_returns_none(tmp_path):
    path = tmp_path / "list.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(["just", "a", "list"], f)
    assert detect_yaml_config(str(path)) is None

import pandas as pd
import numpy as np
import tempfile
import os
import pytest
from vcast.processing import StatiscalSignificance

class DummyConfig:
    def __init__(self, model_a, model_b, metric, output_file):
        self.input_model_A = model_a
        self.input_model_B = model_b
        self.metric = metric
        self.output_file = output_file

@pytest.fixture
def sample_config():
    data_a = pd.DataFrame({
        "fcst_lead": [0, 1, 2],
        "rmse": [0.5, 0.6, 0.7],
        "bias": [0.1, 0.2, 0.3]
    })
    data_b = pd.DataFrame({
        "fcst_lead": [0, 1, 2],
        "rmse": [0.4, 0.5, 0.6],
        "bias": [0.05, 0.15, 0.25]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = os.path.join(tmpdir, "model_a.tsv")
        path_b = os.path.join(tmpdir, "model_b.tsv")
        output_path = os.path.join(tmpdir, "result.tsv")

        data_a.to_csv(path_a, sep="\t", index=False)
        data_b.to_csv(path_b, sep="\t", index=False)

        yield DummyConfig(path_a, path_b, "rmse", output_path)

def test_significance_output_structure_and_validation(sample_config):
    StatiscalSignificance(sample_config)
    df = pd.read_csv(sample_config.output_file, sep="\t")

    assert not df.empty
    required_cols = {
        "fcst_lead", "observed_diff", "p_value",
        "ci_bcl", "ci_bcu", "better_model", "significant"
    }
    assert required_cols.issubset(df.columns)

    for _, row in df.iterrows():
        assert 0 <= row["p_value"] <= 1
        assert row["ci_bcl"] <= row["observed_diff"] <= row["ci_bcu"]
        assert row["better_model"] in ["Model A", "Model B"]
        if row["significant"]:
            assert row["p_value"] < 0.05


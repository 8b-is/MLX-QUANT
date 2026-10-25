import pytest

from examples.vaimshuk_pipeline import run_vaimshuk_pipeline


def test_vaimshuk_e2e_pipeline():
    result = run_vaimshuk_pipeline(verbose=False)
    assert 0.0 <= result["sparsity"] <= 1.0
    assert result["energy_dissipated"] >= 0.0
    assert result["snapshot"].version == 1
    assert result["metrics_count"] > 0
    assert "stripes_mode_ratio" in result["spectrum"]

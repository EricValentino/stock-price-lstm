"""Tests that need PyTorch (run in CI; skipped if torch isn't installed)."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import train  # noqa: E402
from model import PredictionModel  # noqa: E402


def test_model_output_shape():
    model = PredictionModel(input_dim=1, hidden_dim=8, num_layers=2, output_dim=1)
    assert model(torch.zeros(4, 30, 1)).shape == (4, 1)


def test_model_rejects_wrong_rank():
    model = PredictionModel()
    with pytest.raises(ValueError):
        model(torch.zeros(4, 30, 1, 1))


def test_training_smoke(tmp_path, monkeypatch):
    days = pd.bdate_range("2022-01-03", periods=260)
    prices = pd.Series(100 + 10 * np.sin(np.arange(260) / 10), index=days)
    monkeypatch.setattr(train, "download_close_prices", lambda *_args: prices)

    metrics = train.main(
        [
            "--ticker",
            "TEST",
            "--epochs",
            "3",
            "--window",
            "10",
            "--hidden",
            "8",
            "--out-dir",
            str(tmp_path / "models"),
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert metrics["test_days"] == 52
    assert np.isfinite(metrics["test_rmse_lstm"])
    assert (tmp_path / "models" / "test_lstm.pt").exists()
    assert (tmp_path / "reports" / "test_test.png").exists()
    saved = json.loads((tmp_path / "reports" / "test_metrics.json").read_text())
    assert saved["test_rmse_naive"] == pytest.approx(metrics["test_rmse_naive"])

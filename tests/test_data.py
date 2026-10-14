import numpy as np
import pandas as pd
import pytest

from data import (
    Scaler,
    chronological_split,
    extract_close,
    make_windows,
    naive_last_value,
    rmse,
)


def test_make_windows_shapes_and_alignment():
    X, y = make_windows(np.arange(10), window=3)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7, 1)
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
    assert y[0, 0] == 3
    np.testing.assert_array_equal(X[-1, :, 0], [6, 7, 8])
    assert y[-1, 0] == 9


def test_make_windows_rejects_short_series():
    with pytest.raises(ValueError):
        make_windows(np.arange(3), window=3)


def test_extract_close_handles_yfinance_multiindex():
    idx = pd.date_range("2024-01-01", periods=5)
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["MSFT"]], names=["Price", "Ticker"])
    frame = pd.DataFrame(np.arange(10, dtype=float).reshape(5, 2), index=idx, columns=cols)
    close = extract_close(frame)
    assert isinstance(close, pd.Series)
    assert close.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]
    # Regression: a DataFrame here used to produce 4-D tensors and crash the LSTM.
    X, _ = make_windows(close.to_numpy(), window=2)
    assert X.ndim == 3


def test_extract_close_flat_columns():
    frame = pd.DataFrame({"Close": [1.0, None, 3.0]})
    assert extract_close(frame).tolist() == [1.0, 3.0]


def test_scaler_round_trip_and_constant_series():
    values = np.array([10.0, 12.0, 14.0])
    scaler = Scaler.fit(values)
    np.testing.assert_allclose(scaler.inverse(scaler.transform(values)), values, rtol=1e-6)
    assert Scaler.fit(np.ones(5)).std == 1.0


def test_split_fits_scaler_on_training_data_only():
    prices = np.concatenate([np.full(80, 100.0), np.full(20, 200.0)])
    splits = chronological_split(prices, window=5, train_frac=0.8)
    # The jump to 200 happens in the test period; it must not shift the scaler.
    assert splits.scaler.mean == pytest.approx(100.0)


def test_split_sizes_and_test_windows_are_chronological():
    prices = np.arange(100, dtype=float)
    splits = chronological_split(prices, window=10, train_frac=0.8)
    assert len(splits.X_train) == 80 - 10
    assert len(splits.X_test) == 20
    assert splits.test_start == 80
    first_test_target = splits.scaler.inverse(splits.y_test[0])[0]
    last_train_target = splits.scaler.inverse(splits.y_train[-1])[0]
    assert first_test_target == pytest.approx(80.0)
    assert last_train_target == pytest.approx(79.0)


def test_naive_baseline_and_rmse():
    X, y = make_windows(np.array([1.0, 2.0, 3.0, 4.0]), window=2)
    baseline = naive_last_value(X)
    np.testing.assert_array_equal(baseline.reshape(-1), [2.0, 3.0])
    assert rmse(baseline, y) == pytest.approx(1.0)

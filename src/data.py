"""Data loading and preprocessing for next-day closing-price forecasting.

Everything here is plain NumPy/pandas so it can be unit-tested without a network
connection or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def download_close_prices(ticker: str, start: str, end: str | None = None) -> pd.Series:
    """Daily closing prices from Yahoo Finance as a 1-D float Series indexed by date."""
    import yfinance as yf  # imported lazily so tests don't need it

    frame = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if frame is None or frame.empty:
        raise ValueError(f"No price data returned for {ticker!r} starting {start}")
    return extract_close(frame)


def extract_close(frame: pd.DataFrame) -> pd.Series:
    """Return the Close column as a Series.

    Recent yfinance versions return MultiIndex columns like ``("Close", "MSFT")``
    even for a single ticker. Selecting ``frame["Close"]`` then yields a one-column
    DataFrame, which silently turns 2-D windows into 3-D ones. Normalize here.
    """
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError("Expected data for exactly one ticker")
        close = close.iloc[:, 0]
    return close.astype("float64").dropna()


def make_windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 1-D series into (inputs, targets) for one-step-ahead prediction.

    Each input is ``window`` consecutive values and its target is the next value.
    Shapes: X is ``(n, window, 1)``, y is ``(n, 1)``, with ``n = len(values) - window``.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if window < 1:
        raise ValueError("window must be at least 1")
    n = len(values) - window
    if n < 1:
        raise ValueError(f"Need more than {window} values to build a window, got {len(values)}")
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    X = values[idx][..., None]
    y = values[window:][:, None]
    return X, y


@dataclass
class Scaler:
    """Standardization fitted on the training split only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> Scaler:
        std = float(np.std(values))
        return cls(mean=float(np.mean(values)), std=std if std > 0 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Scaler
    test_start: int  # index into the original series of the first test target


def chronological_split(prices: np.ndarray, window: int, train_frac: float = 0.8) -> Splits:
    """Split by time, fit the scaler on training prices only, then window each part.

    Fitting the scaler on the full series (as the first version of this project did)
    leaks the test period's mean and variance into training. The test windows here
    may *look back* into the last ``window`` training prices, which is legitimate:
    that history is known at prediction time.
    """
    prices = np.asarray(prices, dtype=np.float32).reshape(-1)
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    split = int(len(prices) * train_frac)
    if split <= window or len(prices) - split < 1:
        raise ValueError("Series too short for this window and split")

    scaler = Scaler.fit(prices[:split])
    scaled = scaler.transform(prices)

    X_train, y_train = make_windows(scaled[:split], window)
    X_test, y_test = make_windows(scaled[split - window :], window)
    return Splits(X_train, y_train, X_test, y_test, scaler, test_start=split)


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def naive_last_value(X: np.ndarray) -> np.ndarray:
    """Persistence baseline: predict that tomorrow's price equals today's."""
    return X[:, -1, :]

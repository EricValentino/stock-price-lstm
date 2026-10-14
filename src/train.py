"""Train the LSTM on one ticker and report test RMSE against a naive baseline.

Example:
    python src/train.py --ticker MSFT --start 2020-01-01 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import chronological_split, download_close_prices, naive_last_value, rmse
from model import PredictionModel

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ticker", default="MSFT")
    p.add_argument("--start", default="2020-01-01", help="first date of history (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="last date of history (default: today)")
    p.add_argument("--window", type=int, default=30, help="days of history per prediction")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ROOT / "models")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, X_train, y_train, epochs, batch_size, lr, device, log_every=25):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,  # windows are independent samples; shuffling doesn't leak the future
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}  train MSE (scaled) {total / len(loader.dataset):.5f}")
    return model


@torch.no_grad()
def predict(model, X, device) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(X).to(device)).cpu().numpy()


def save_plot(dates, actual, predicted, baseline, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, actual, label="Actual close", linewidth=1.6)
    ax.plot(dates, predicted, label="LSTM prediction", linewidth=1.2)
    ax.plot(dates, baseline, label="Naive (yesterday's close)", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    close = download_close_prices(args.ticker, args.start, args.end)
    splits = chronological_split(close.to_numpy(), args.window, args.train_frac)
    print(
        f"{args.ticker}: {len(close)} days, {len(splits.X_train)} train windows, "
        f"{len(splits.X_test)} test windows, device={device}"
    )

    model = PredictionModel(1, args.hidden, args.layers, 1, args.dropout).to(device)
    train_model(
        model, splits.X_train, splits.y_train, args.epochs, args.batch_size, args.lr, device
    )

    inv = splits.scaler.inverse
    actual = inv(splits.y_test).reshape(-1)
    predicted = inv(predict(model, splits.X_test, device)).reshape(-1)
    baseline = inv(naive_last_value(splits.X_test)).reshape(-1)

    metrics = {
        "ticker": args.ticker,
        "start": args.start,
        "end": args.end,
        "test_days": int(len(actual)),
        "test_rmse_lstm": rmse(predicted, actual),
        "test_rmse_naive": rmse(baseline, actual),
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k not in {"out_dir", "report_dir"}
        },
    }
    print(
        f"Test RMSE  LSTM: ${metrics['test_rmse_lstm']:.2f}   "
        f"naive baseline: ${metrics['test_rmse_naive']:.2f}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler": {"mean": splits.scaler.mean, "std": splits.scaler.std},
            "hparams": {"hidden": args.hidden, "layers": args.layers, "window": args.window},
            "ticker": args.ticker,
        },
        args.out_dir / f"{args.ticker.lower()}_lstm.pt",
    )
    (args.report_dir / f"{args.ticker.lower()}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    if not args.no_plot:
        dates = close.index[splits.test_start : splits.test_start + len(actual)]
        save_plot(
            dates,
            actual,
            predicted,
            baseline,
            args.report_dir / f"{args.ticker.lower()}_test.png",
            f"{args.ticker}: next-day close on held-out data",
        )
    print(f"Saved model to {args.out_dir} and report to {args.report_dir}")
    return metrics


if __name__ == "__main__":
    main()

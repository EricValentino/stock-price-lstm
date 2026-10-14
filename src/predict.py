"""Predict the next trading day's close with a model saved by train.py.

Example:
    python src/predict.py --ticker MSFT
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from data import Scaler, download_close_prices
from model import PredictionModel

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> float:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ticker", default="MSFT")
    p.add_argument("--model", type=Path, default=None, help="checkpoint path (default: models/)")
    args = p.parse_args(argv)

    path = args.model or ROOT / "models" / f"{args.ticker.lower()}_lstm.pt"
    if not path.exists():
        raise SystemExit(
            f"No checkpoint at {path}. Run: python src/train.py --ticker {args.ticker}"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    hp = checkpoint["hparams"]
    scaler = Scaler(**checkpoint["scaler"])
    model = PredictionModel(1, hp["hidden"], hp["layers"], 1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Pull a generous calendar range so weekends/holidays still leave enough trading days.
    start = (date.today() - timedelta(days=hp["window"] * 3)).isoformat()
    close = download_close_prices(args.ticker, start)
    if len(close) < hp["window"]:
        raise SystemExit(f"Only {len(close)} trading days available; need {hp['window']}")

    window = scaler.transform(close.to_numpy()[-hp["window"] :]).reshape(1, -1, 1)
    with torch.no_grad():
        scaled = model(torch.from_numpy(window.astype(np.float32))).item()
    prediction = float(scaler.inverse(np.array([scaled]))[0])

    print(f"{args.ticker} last close ({close.index[-1].date()}): ${close.iloc[-1]:.2f}")
    print(f"{args.ticker} predicted next close: ${prediction:.2f}")
    return prediction


if __name__ == "__main__":
    main()

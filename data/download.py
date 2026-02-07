"""Download datasets: ETTh1, Electricity, BTC-USD."""

import numpy as np
from gluonts.dataset.repository import get_dataset


def get_etth1():
    """Download and return the ETTh1 dataset.

    Returns the GluonTS dataset object for "ett_small_1h" which contains
    hourly Electricity Transformer Temperature data (~17,000 timesteps,
    7 features). The oil temperature (OT) column is used as target.
    """
    dataset = get_dataset("ett_small_1h")
    return dataset


def get_electricity():
    """Download and return the Electricity dataset via GluonTS.

    Returns the GluonTS dataset object for "electricity" which contains
    hourly electricity consumption for 370 clients (~26,304 timesteps each).
    """
    dataset = get_dataset("electricity")
    return dataset


def get_btc(start="2014-09-17", end="2026-02-05"):
    """Download BTC-USD daily close prices and return train/val/test splits.

    Uses yfinance to download data, converts to log prices.
    Split: 80% train, 10% val, 10% test.

    Returns:
        dict with keys: train_vals, val_vals, test_vals (numpy float32 arrays)
    """
    import yfinance as yf

    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(start=start, end=end)
    prices = df["Close"].values.astype(np.float32)
    log_prices = np.log(prices)

    n = len(log_prices)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    return {
        "train_vals": log_prices[:train_end],
        "val_vals": log_prices[train_end:val_end],
        "test_vals": log_prices[val_end:],
    }


if __name__ == "__main__":
    import sys

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "etth1"

    if dataset_name == "etth1":
        dataset = get_etth1()
        print("=== ETTh1 Dataset Stats ===")
        print(f"Frequency: {dataset.metadata.freq}")
        print(f"Prediction length (default): {dataset.metadata.prediction_length}")
        test_entries = list(dataset.test)
        print(f"\nTest set:")
        print(f"  Number of time series: {len(test_entries)}")
        for i, entry in enumerate(test_entries):
            ts = entry["target"]
            print(f"  Series {i}: length={len(ts)}, start={entry['start']}")
        train_entries = list(dataset.train)
        print(f"\nTrain set:")
        print(f"  Number of time series: {len(train_entries)}")
        for i, entry in enumerate(train_entries):
            ts = entry["target"]
            print(f"  Series {i}: length={len(ts)}, start={entry['start']}")

    elif dataset_name == "electricity":
        dataset = get_electricity()
        print("=== Electricity Dataset Stats ===")
        print(f"Frequency: {dataset.metadata.freq}")
        test_entries = list(dataset.test)
        print(f"Test set: {len(test_entries)} time series")
        if test_entries:
            print(f"  Series 0: length={len(test_entries[0]['target'])}")

    elif dataset_name == "btc":
        btc = get_btc()
        print("=== BTC-USD Dataset Stats ===")
        print(f"Train: {len(btc['train_vals'])} days")
        print(f"Val:   {len(btc['val_vals'])} days")
        print(f"Test:  {len(btc['test_vals'])} days")
        print(f"Total: {len(btc['train_vals']) + len(btc['val_vals']) + len(btc['test_vals'])} days")
        print(f"Test log price range: [{btc['test_vals'].min():.4f}, {btc['test_vals'].max():.4f}]")

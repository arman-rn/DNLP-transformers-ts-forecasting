"""Download ETTh1 dataset using GluonTS."""

from gluonts.dataset.repository import get_dataset


def get_etth1():
    """Download and return the ETTh1 dataset.

    Returns the GluonTS dataset object for "ett_small_1h" which contains
    hourly Electricity Transformer Temperature data (~17,000 timesteps,
    7 features). The oil temperature (OT) column is used as target.
    """
    dataset = get_dataset("ett_small_1h")
    return dataset


if __name__ == "__main__":
    dataset = get_etth1()

    print("=== ETTh1 Dataset Stats ===")
    print(f"Frequency: {dataset.metadata.freq}")
    print(f"Prediction length (default): {dataset.metadata.prediction_length}")

    # Test set stats
    test_entries = list(dataset.test)
    print(f"\nTest set:")
    print(f"  Number of time series: {len(test_entries)}")
    for i, entry in enumerate(test_entries):
        ts = entry["target"]
        print(f"  Series {i}: length={len(ts)}, start={entry['start']}")

    # Train set stats
    train_entries = list(dataset.train)
    print(f"\nTrain set:")
    print(f"  Number of time series: {len(train_entries)}")
    for i, entry in enumerate(train_entries):
        ts = entry["target"]
        print(f"  Series {i}: length={len(ts)}, start={entry['start']}")

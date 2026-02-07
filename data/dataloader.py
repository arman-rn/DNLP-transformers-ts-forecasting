"""Convert datasets to tensors suitable for Chronos-2."""

import torch
import numpy as np

from data.download import get_etth1


def prepare_etth1_data(dataset, context_length=512, prediction_length=96):
    """Create sliding window samples from a GluonTS dataset test set.

    Takes the test split and generates (context, target) pairs using a sliding window.

    Args:
        dataset: GluonTS dataset object (has .test attribute).
        context_length: Number of past timesteps for context.
        prediction_length: Number of future timesteps to predict.

    Returns:
        contexts: List of torch.Tensor, each shape (context_length,).
        targets: List of torch.Tensor, each shape (prediction_length,).
    """
    contexts = []
    targets = []

    window_size = context_length + prediction_length

    for entry in dataset.test:
        ts = np.array(entry["target"], dtype=np.float32)

        if len(ts) < window_size:
            continue

        # Slide over the time series with non-overlapping steps of prediction_length
        for start in range(0, len(ts) - window_size + 1, prediction_length):
            context = torch.tensor(ts[start : start + context_length])
            target = torch.tensor(ts[start + context_length : start + window_size])
            contexts.append(context)
            targets.append(target)

    return contexts, targets


# Keep old name as alias for backward compatibility
prepare_data = prepare_etth1_data


def prepare_electricity_data(dataset, context_length=512, prediction_length=96, num_samples=100):
    """Create sliding window samples from the Electricity dataset test set.

    The Electricity dataset has 370 time series. We take samples across
    multiple series to get diversity.

    Args:
        dataset: GluonTS dataset object from get_electricity().
        context_length: Number of past timesteps for context.
        prediction_length: Number of future timesteps to predict.
        num_samples: Max number of samples to generate (0 = all).

    Returns:
        contexts: List of torch.Tensor, each shape (context_length,).
        targets: List of torch.Tensor, each shape (prediction_length,).
    """
    contexts = []
    targets = []

    window_size = context_length + prediction_length

    for entry in dataset.test:
        ts = np.array(entry["target"], dtype=np.float32)

        if len(ts) < window_size:
            continue

        # Take one sample from each series (from the end, where test data is)
        start = len(ts) - window_size
        context = torch.tensor(ts[start : start + context_length])
        target = torch.tensor(ts[start + context_length : start + window_size])
        contexts.append(context)
        targets.append(target)

        if num_samples > 0 and len(contexts) >= num_samples:
            break

    return contexts, targets


def prepare_btc_data(btc_dict, context_length=512, prediction_length=96, num_samples=0):
    """Create sliding window samples from BTC-USD data.

    Context can draw from train/val data, but prediction targets must fall
    within the test period. This handles the case where test_set alone is
    shorter than context_length + prediction_length.

    Args:
        btc_dict: Dict with train_vals, val_vals, test_vals (numpy arrays).
        context_length: Number of past timesteps for context.
        prediction_length: Number of future timesteps to predict.
        num_samples: Max number of samples (0 = all).

    Returns:
        contexts: List of torch.Tensor, each shape (context_length,).
        targets: List of torch.Tensor, each shape (prediction_length,).
    """
    # Concatenate all data; targets must start within test region
    all_data = np.concatenate([btc_dict["train_vals"], btc_dict["val_vals"], btc_dict["test_vals"]])
    test_start_idx = len(btc_dict["train_vals"]) + len(btc_dict["val_vals"])

    contexts = []
    targets = []
    window_size = context_length + prediction_length

    # First window: target starts at test_start_idx
    first_start = test_start_idx - context_length
    if first_start < 0:
        raise ValueError(
            f"Not enough data before test set for context_length={context_length}"
        )

    # Step = 1 for maximum samples from short daily data (caller caps via num_samples)
    step = 1
    for start in range(first_start, len(all_data) - window_size + 1, step):
        context = torch.tensor(all_data[start : start + context_length])
        target = torch.tensor(all_data[start + context_length : start + window_size])
        contexts.append(context)
        targets.append(target)

        if num_samples > 0 and len(contexts) >= num_samples:
            break

    return contexts, targets


if __name__ == "__main__":
    import sys
    from data.download import get_electricity, get_btc

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "etth1"

    if dataset_name == "etth1":
        dataset = get_etth1()
        contexts, targets = prepare_etth1_data(dataset, context_length=512, prediction_length=96)
        print("=== ETTh1 DataLoader Stats ===")
    elif dataset_name == "electricity":
        dataset = get_electricity()
        contexts, targets = prepare_electricity_data(dataset, context_length=512, prediction_length=96)
        print("=== Electricity DataLoader Stats ===")
    elif dataset_name == "btc":
        btc = get_btc()
        contexts, targets = prepare_btc_data(btc, context_length=512, prediction_length=96)
        print("=== BTC-USD DataLoader Stats ===")
    else:
        print(f"Unknown dataset: {dataset_name}")
        sys.exit(1)

    print(f"Number of samples: {len(contexts)}")
    if contexts:
        print(f"Context shape: {contexts[0].shape}")
        print(f"Target shape:  {targets[0].shape}")
        print(f"Context dtype: {contexts[0].dtype}")
        print(f"\nFirst context - min: {contexts[0].min():.4f}, max: {contexts[0].max():.4f}")
        print(f"First target  - min: {targets[0].min():.4f}, max: {targets[0].max():.4f}")

"""Convert GluonTS dataset to tensors suitable for Chronos-2."""

import torch
import numpy as np

from data.download import get_etth1


def prepare_data(dataset, context_length=512, prediction_length=96):
    """Create sliding window samples from the dataset test set.

    Takes the test split of a GluonTS dataset and generates
    (context, target) pairs using a sliding window.

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


if __name__ == "__main__":
    dataset = get_etth1()
    contexts, targets = prepare_data(dataset, context_length=512, prediction_length=96)

    print("=== ETTh1 DataLoader Stats ===")
    print(f"Number of samples: {len(contexts)}")
    if contexts:
        print(f"Context shape: {contexts[0].shape}")
        print(f"Target shape:  {targets[0].shape}")
        print(f"Context dtype: {contexts[0].dtype}")
        print(f"\nFirst context - min: {contexts[0].min():.4f}, max: {contexts[0].max():.4f}")
        print(f"First target  - min: {targets[0].min():.4f}, max: {targets[0].max():.4f}")

"""Evaluation metrics for time series forecasting."""

import torch
import numpy as np


def _to_point_forecast(pred):
    """Convert predictions to point forecast.

    If pred has 3 dimensions (num_samples, batch, pred_len), take the median
    across samples to get a single point forecast. This handles Chronos-2's
    probabilistic output which returns multiple forecast samples.
    """
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)

    if pred.ndim == 3:
        # (num_samples, batch, pred_len) -> median over samples
        pred = pred.median(dim=0).values

    return pred.float()


def compute_mse(pred, target):
    """Mean Squared Error between prediction and target.

    Args:
        pred: Predictions, shape (batch, pred_len) or (num_samples, batch, pred_len).
        target: Ground truth, shape (batch, pred_len).

    Returns:
        Scalar MSE value.
    """
    pred = _to_point_forecast(pred)
    if isinstance(target, np.ndarray):
        target = torch.tensor(target)
    return ((pred - target.float()) ** 2).mean().item()


def compute_mae(pred, target):
    """Mean Absolute Error between prediction and target.

    Args:
        pred: Predictions, shape (batch, pred_len) or (num_samples, batch, pred_len).
        target: Ground truth, shape (batch, pred_len).

    Returns:
        Scalar MAE value.
    """
    pred = _to_point_forecast(pred)
    if isinstance(target, np.ndarray):
        target = torch.tensor(target)
    return (pred - target.float()).abs().mean().item()


def compute_metrics(pred, target):
    """Compute all metrics.

    Args:
        pred: Predictions, shape (batch, pred_len) or (num_samples, batch, pred_len).
        target: Ground truth, shape (batch, pred_len).

    Returns:
        Dict with "mse" and "mae" keys.
    """
    pred = _to_point_forecast(pred)
    if isinstance(target, np.ndarray):
        target = torch.tensor(target)
    target = target.float()

    mse = ((pred - target) ** 2).mean().item()
    mae = (pred - target).abs().mean().item()
    return {"mse": mse, "mae": mae}

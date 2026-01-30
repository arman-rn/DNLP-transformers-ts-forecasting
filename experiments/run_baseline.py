"""Run vanilla Chronos-2 baseline on ETTh1."""

import argparse
import json
from pathlib import Path

import torch
import wandb

from chronos import Chronos2Pipeline
from data.download import get_etth1
from data.dataloader import prepare_data
from evaluation.metrics import compute_metrics
from src.config import load_config

MEDIAN_QUANTILE_IDX = 10  # Chronos-2 outputs 21 quantiles [0.01, 0.05, ..., 0.5, ..., 0.99]; index 10 = 0.5


def main():
    parser = argparse.ArgumentParser(description="Chronos-2 baseline on ETTh1")
    parser.add_argument("--config", default="configs/ttt_config.yaml")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of test samples to evaluate (0 = all)")
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = config["model"]["name"]
    device = config["model"]["device"]
    context_length = config["data"]["context_length"]
    prediction_length = config["data"]["prediction_length"]
    seed = config["experiment"]["seed"]

    torch.manual_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    # Load model
    print(f"Loading {model_name}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    # Load data
    print("Loading ETTh1 dataset...")
    dataset = get_etth1()
    contexts, targets = prepare_data(dataset, context_length, prediction_length)

    n = len(contexts) if args.num_samples == 0 else min(args.num_samples, len(contexts))
    contexts = contexts[:n]
    targets = targets[:n]
    print(f"Evaluating on {n} samples (context_length={context_length}, "
          f"prediction_length={prediction_length})")

    # Initialize wandb
    wandb.init(
        project=config["experiment"]["wandb_project"],
        name="baseline",
        config={
            "method": "baseline",
            "model": model_name,
            "dataset": config["data"]["dataset"],
            "context_length": context_length,
            "prediction_length": prediction_length,
            "num_samples": n,
            "seed": seed,
        },
    )

    # Run predictions (batch call - pipeline handles internal batching)
    print("Running predictions...")
    forecasts = pipeline.predict(contexts, prediction_length=prediction_length)
    # forecasts: list of n tensors, each shape (1, 21, prediction_length)

    # Compute per-sample metrics
    all_mse = []
    all_mae = []

    for i, (forecast, tgt) in enumerate(zip(forecasts, targets)):
        # forecast shape: (1, n_quantiles, pred_len) for univariate
        point_forecast = forecast[0, MEDIAN_QUANTILE_IDX, :]  # (pred_len,)
        m = compute_metrics(point_forecast.unsqueeze(0), tgt.unsqueeze(0))
        all_mse.append(m["mse"])
        all_mae.append(m["mae"])
        wandb.log({"sample_mse": m["mse"], "sample_mae": m["mae"], "sample_idx": i})

    avg_mse = sum(all_mse) / len(all_mse)
    avg_mae = sum(all_mae) / len(all_mae)

    print(f"\n=== Baseline Results ({n} samples) ===")
    print(f"MSE: {avg_mse:.6f}")
    print(f"MAE: {avg_mae:.6f}")

    wandb.log({"avg_mse": avg_mse, "avg_mae": avg_mae})

    # Save results
    results_dir = Path("results/baseline")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "method": "baseline",
        "model": model_name,
        "dataset": config["data"]["dataset"],
        "context_length": context_length,
        "prediction_length": prediction_length,
        "num_samples": n,
        "seed": seed,
        "avg_mse": avg_mse,
        "avg_mae": avg_mae,
        "per_sample_mse": all_mse,
        "per_sample_mae": all_mae,
    }

    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

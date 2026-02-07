"""Run TTT (Test-Time Training) experiment.

Loads Chronos-2 with TTT adaptation and evaluates on the chosen dataset.
Results are logged to wandb and saved as JSON for comparison with the baseline.

Run from project root:
    python experiments/run_ttt.py
    python experiments/run_ttt.py --dataset btc --target output --n_mask 16 --lr 1e-4 --optimizer sgd
    python experiments/run_ttt.py --num_samples 20   # quick test
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch
import wandb
from tqdm import tqdm

from chronos import Chronos2Pipeline
from experiments.run_baseline import load_dataset
from evaluation.metrics import compute_metrics
from src.config import load_config
from src.ttt import TTTChronos

MEDIAN_QUANTILE_IDX = 10  # Chronos-2 outputs 21 quantiles [0.01, 0.05, ..., 0.5, ..., 0.99]; index 10 = 0.5


def main():
    parser = argparse.ArgumentParser(description="TTT experiment")
    parser.add_argument("--config", default="configs/ttt_config.yaml")
    parser.add_argument("--dataset", default="etth1",
                        choices=["etth1", "electricity", "btc"],
                        help="Dataset to evaluate on (default: etth1)")
    parser.add_argument("--context_length", type=int, default=None,
                        help="Context length override (default: from config)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of test samples to evaluate (0 = all)")
    # TTT hyperparameter overrides
    parser.add_argument("--n_mask", type=int, default=None)
    parser.add_argument("--ttt_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--target", default="output",
                        choices=["input", "output", "both"],
                        help="Which layer(s) to adapt (default: output)")
    parser.add_argument("--optimizer", default="adam",
                        choices=["adam", "sgd"],
                        help="Optimizer for TTT updates (default: adam)")
    args = parser.parse_args()

    config = load_config(args.config)

    n_mask = args.n_mask if args.n_mask is not None else config["ttt"]["n_mask"]
    ttt_steps = args.ttt_steps if args.ttt_steps is not None else config["ttt"]["ttt_steps"]
    lr = args.lr if args.lr is not None else config["ttt"]["lr"]
    target = args.target
    optim = args.optimizer

    model_name = config["model"]["name"]
    device = config["model"]["device"]
    context_length = args.context_length if args.context_length is not None else config["data"]["context_length"]
    prediction_length = config["data"]["prediction_length"]
    seed = config["experiment"]["seed"]
    dataset_name = args.dataset

    torch.manual_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    # ---- Load model ----
    print(f"Loading {model_name}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    ttt_model = TTTChronos(
        pipeline, n_mask=n_mask, ttt_steps=ttt_steps, lr=lr, target=target,
        optimizer=optim
    )

    # ---- Load data ----
    contexts, targets = load_dataset(dataset_name, context_length, prediction_length, args.num_samples)

    n = len(contexts) if args.num_samples == 0 else min(args.num_samples, len(contexts))
    contexts = contexts[:n]
    targets = targets[:n]
    print(f"Evaluating on {n} samples (dataset={dataset_name}, context_length={context_length}, "
          f"prediction_length={prediction_length})")
    print(f"TTT config: target={target}, n_mask={n_mask}, ttt_steps={ttt_steps}, lr={lr}, optimizer={optim}")

    # ---- Initialize wandb ----
    wandb.init(
        project=config["experiment"]["wandb_project"],
        name=f"ttt_{dataset_name}_{target}_{optim}_mask{n_mask}_steps{ttt_steps}",
        config={
            "method": "ttt",
            "target": target,
            "optimizer": optim,
            "model": model_name,
            "dataset": dataset_name,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "n_mask": n_mask,
            "ttt_steps": ttt_steps,
            "lr": lr,
            "num_samples": n,
            "seed": seed,
        },
    )

    # ---- Run TTT predictions per sample ----
    all_mse = []
    all_mae = []

    pbar = tqdm(
        enumerate(zip(contexts, targets)),
        total=n,
        desc=f"TTT ({target}, mask={n_mask}, steps={ttt_steps})",
    )
    for i, (ctx, tgt) in pbar:
        with contextlib.redirect_stdout(io.StringIO()):
            forecast = ttt_model.predict(ctx, prediction_length=prediction_length)

        point_forecast = forecast[0][0, MEDIAN_QUANTILE_IDX, :]
        m = compute_metrics(point_forecast.unsqueeze(0), tgt.unsqueeze(0))
        all_mse.append(m["mse"])
        all_mae.append(m["mae"])

        wandb.log({"sample_mse": m["mse"], "sample_mae": m["mae"], "sample_idx": i})

        running_mse = sum(all_mse) / len(all_mse)
        running_mae = sum(all_mae) / len(all_mae)
        pbar.set_postfix(mse=f"{running_mse:.4f}", mae=f"{running_mae:.4f}")

    # ---- Final metrics ----
    avg_mse = sum(all_mse) / len(all_mse)
    avg_mae = sum(all_mae) / len(all_mae)

    print(f"\n=== TTT Results ({dataset_name}, {n} samples) ===")
    print(f"  target={target}, n_mask={n_mask}, ttt_steps={ttt_steps}, lr={lr}")
    print(f"  MSE: {avg_mse:.6f}")
    print(f"  MAE: {avg_mae:.6f}")

    wandb.log({"avg_mse": avg_mse, "avg_mae": avg_mae})

    # ---- Compare with baseline ----
    baseline_path = Path(f"results/baseline/{dataset_name}/results.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

        b_mse = baseline["avg_mse"]
        b_mae = baseline["avg_mae"]

        mse_diff = avg_mse - b_mse
        mae_diff = avg_mae - b_mae
        mse_pct = mse_diff / b_mse * 100
        mae_pct = mae_diff / b_mae * 100

        print(f"\n=== Comparison with Baseline ===")
        print(f"  {'Metric':<6} {'Baseline':>12} {'TTT':>12} {'Diff':>12} {'Change':>8}")
        print(f"  {'MSE':<6} {b_mse:>12.6f} {avg_mse:>12.6f} {mse_diff:>+12.6f} {mse_pct:>+7.1f}%")
        print(f"  {'MAE':<6} {b_mae:>12.6f} {avg_mae:>12.6f} {mae_diff:>+12.6f} {mae_pct:>+7.1f}%")

        wandb.log({
            "baseline_mse": b_mse,
            "baseline_mae": b_mae,
            "mse_change_pct": mse_pct,
            "mae_change_pct": mae_pct,
        })
    else:
        print(f"\n(No baseline results found at {baseline_path}. "
              "Run experiments/run_baseline.py first for comparison.)")

    # ---- Save results ----
    results_dir = Path(f"results/ttt/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "method": "ttt",
        "target": target,
        "optimizer": optim,
        "model": model_name,
        "dataset": dataset_name,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "n_mask": n_mask,
        "ttt_steps": ttt_steps,
        "lr": lr,
        "num_samples": n,
        "seed": seed,
        "avg_mse": avg_mse,
        "avg_mae": avg_mae,
        "per_sample_mse": all_mse,
        "per_sample_mae": all_mae,
    }

    results_path = results_dir / f"results_{target}_{optim}_{n_mask}_{ttt_steps}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

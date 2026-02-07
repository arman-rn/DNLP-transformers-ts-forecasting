"""Run TTT hyperparameter ablation sweep.

Loads Chronos-2 and data ONCE, then runs all hyperparameter combinations.
Results are logged to wandb and saved as JSON.

Run from project root:
    python experiments/run_ablations.py
    python experiments/run_ablations.py --dataset btc
    python experiments/run_ablations.py --num_samples 20   # quick test
    python experiments/run_ablations.py --num_samples 0    # all samples
"""

import argparse
import contextlib
import io
import json
from itertools import product
from pathlib import Path

import torch
import wandb
from tqdm import tqdm

from chronos import Chronos2Pipeline
from experiments.run_baseline import load_dataset
from evaluation.metrics import compute_metrics
from src.config import load_config
from src.ttt import TTTChronos

MEDIAN_QUANTILE_IDX = 10  # 21 quantiles; index 10 = 0.5 median

# ---- Hyperparameter grids per dataset ----
GRIDS = {
    "etth1": {
        "target": ["output"],
        "n_mask": [16, 32],
        "ttt_steps": [1, 2, 5],
        "lr": [1e-5, 5e-5, 1e-4, 5e-4],
        "optimizer": ["adam", "sgd"],
    },
    "electricity": {
        "target": ["output"],
        "n_mask": [16, 32, 48],
        "ttt_steps": [1, 3, 5, 10],
        "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "optimizer": ["sgd"],
    },
    "btc": {
        "target": ["output"],
        "n_mask": [16, 32],
        "ttt_steps": [1, 3, 5],
        "lr": [1e-5, 5e-5, 1e-4, 5e-4],
        "optimizer": ["sgd"],
    },
}


def grid_configs(grid):
    """Expand grid dict into list of config dicts."""
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    for combo in product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def run_single_config(pipeline, contexts, targets, prediction_length, config):
    """Run TTT with a single hyperparameter config. Returns avg metrics."""
    ttt_model = TTTChronos(
        pipeline,
        n_mask=config["n_mask"],
        ttt_steps=config["ttt_steps"],
        lr=config["lr"],
        target=config["target"],
        optimizer=config["optimizer"],
    )

    all_mse = []
    all_mae = []

    for ctx, tgt in zip(contexts, targets):
        with contextlib.redirect_stdout(io.StringIO()):
            forecast = ttt_model.predict(ctx, prediction_length=prediction_length)

        point_forecast = forecast[0][0, MEDIAN_QUANTILE_IDX, :]
        m = compute_metrics(point_forecast.unsqueeze(0), tgt.unsqueeze(0))
        all_mse.append(m["mse"])
        all_mae.append(m["mae"])

    avg_mse = sum(all_mse) / len(all_mse)
    avg_mae = sum(all_mae) / len(all_mae)

    return {
        "avg_mse": avg_mse,
        "avg_mae": avg_mae,
        "per_sample_mse": all_mse,
        "per_sample_mae": all_mae,
    }


def main():
    parser = argparse.ArgumentParser(description="TTT ablation sweep")
    parser.add_argument("--config", default="configs/ttt_config.yaml")
    parser.add_argument("--dataset", default="etth1",
                        choices=["etth1", "electricity", "btc"],
                        help="Dataset to evaluate on (default: etth1)")
    parser.add_argument("--context_length", type=int, default=None,
                        help="Context length override (default: from config)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of test samples to evaluate (0 = all)")
    args = parser.parse_args()

    config = load_config(args.config)

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

    # ---- Load model ONCE ----
    print(f"Loading {model_name}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    # ---- Load data ONCE ----
    contexts, targets = load_dataset(dataset_name, context_length, prediction_length, args.num_samples)

    n = len(contexts) if args.num_samples == 0 else min(args.num_samples, len(contexts))
    contexts = contexts[:n]
    targets = targets[:n]

    # ---- Generate all configs ----
    GRID = GRIDS.get(dataset_name, GRIDS["etth1"])
    all_configs = grid_configs(GRID)
    total = len(all_configs)
    print(f"\nAblation sweep: {total} configurations x {n} samples")
    print(f"Grid: {GRID}\n")

    # ---- Load baseline for comparison ----
    baseline_mse = None
    baseline_mae = None
    baseline_path = Path(f"results/baseline/{dataset_name}/results.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_mse = baseline["avg_mse"]
        baseline_mae = baseline["avg_mae"]
        print(f"Baseline: MSE={baseline_mse:.6f}, MAE={baseline_mae:.6f}\n")

    # ---- Initialize wandb for the sweep ----
    wandb.init(
        project=config["experiment"]["wandb_project"],
        name=f"ablation_sweep_{dataset_name}",
        config={
            "method": "ablation",
            "grid": GRID,
            "model": model_name,
            "dataset": dataset_name,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "num_samples": n,
            "seed": seed,
        },
    )

    # ---- Run sweep ----
    all_results = []

    pbar = tqdm(all_configs, desc="Ablation sweep", total=total)
    for i, cfg in enumerate(pbar):
        label = (f"target={cfg['target']}, n_mask={cfg['n_mask']}, "
                 f"steps={cfg['ttt_steps']}, lr={cfg['lr']}, opt={cfg['optimizer']}")
        pbar.set_description(f"[{i+1}/{total}] {label}")

        metrics = run_single_config(
            pipeline, contexts, targets, prediction_length, cfg
        )

        result = {**cfg, **metrics}

        # Add baseline comparison
        if baseline_mse is not None:
            result["mse_change_pct"] = (metrics["avg_mse"] - baseline_mse) / baseline_mse * 100
            result["mae_change_pct"] = (metrics["avg_mae"] - baseline_mae) / baseline_mae * 100

        all_results.append(result)

        # Log to wandb
        wandb.log({
            "config_idx": i,
            "target": cfg["target"],
            "n_mask": cfg["n_mask"],
            "ttt_steps": cfg["ttt_steps"],
            "lr": cfg["lr"],
            "optimizer": cfg["optimizer"],
            "avg_mse": metrics["avg_mse"],
            "avg_mae": metrics["avg_mae"],
            **({"mse_change_pct": result["mse_change_pct"],
                "mae_change_pct": result["mae_change_pct"]}
               if baseline_mse is not None else {}),
        })

        # Update progress bar with running best
        best_so_far = min(all_results, key=lambda r: r["avg_mse"])
        pbar.set_postfix(
            current_mse=f"{metrics['avg_mse']:.4f}",
            best_mse=f"{best_so_far['avg_mse']:.4f}",
        )

    # ---- Summary table ----
    # Sort by MSE ascending
    all_results.sort(key=lambda r: r["avg_mse"])

    print(f"\n{'='*100}")
    print(f"ABLATION RESULTS ({total} configs, {n} samples)")
    print(f"{'='*100}")

    header = (f"  {'Rank':<5} {'Target':<8} {'n_mask':<7} {'Steps':<6} "
              f"{'LR':<10} {'Opt':<6} {'MSE':>10} {'MAE':>10}")
    if baseline_mse is not None:
        header += f" {'MSE %':>8}"
    print(header)
    print(f"  {'-'*len(header.strip())}")

    if baseline_mse is not None:
        print(f"  {'base':<5} {'--':<8} {'--':<7} {'--':<6} "
              f"{'--':<10} {'--':<6} {baseline_mse:>10.4f} {baseline_mae:>10.4f} {'0.0%':>8}")

    for rank, r in enumerate(all_results, 1):
        line = (f"  {rank:<5} {r['target']:<8} {r['n_mask']:<7} {r['ttt_steps']:<6} "
                f"{r['lr']:<10.0e} {r['optimizer']:<6} {r['avg_mse']:>10.4f} {r['avg_mae']:>10.4f}")
        if baseline_mse is not None:
            line += f" {r['mse_change_pct']:>+7.1f}%"
        print(line)

    # ---- Best config ----
    best = all_results[0]
    print(f"\n{'='*100}")
    print(f"BEST CONFIG:")
    print(f"  target={best['target']}, n_mask={best['n_mask']}, "
          f"ttt_steps={best['ttt_steps']}, lr={best['lr']}, optimizer={best['optimizer']}")
    print(f"  MSE: {best['avg_mse']:.6f}  MAE: {best['avg_mae']:.6f}")
    if baseline_mse is not None:
        print(f"  vs Baseline: MSE {best['mse_change_pct']:+.1f}%, MAE {best['mae_change_pct']:+.1f}%")
    print(f"{'='*100}")

    wandb.log({
        "best_mse": best["avg_mse"],
        "best_mae": best["avg_mae"],
        "best_target": best["target"],
        "best_n_mask": best["n_mask"],
        "best_ttt_steps": best["ttt_steps"],
        "best_lr": best["lr"],
        "best_optimizer": best["optimizer"],
    })

    # ---- Save results ----
    results_dir = Path(f"results/ablations/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Strip per-sample lists for the summary file (keeps it small)
    summary_results = []
    for r in all_results:
        summary = {k: v for k, v in r.items()
                   if k not in ("per_sample_mse", "per_sample_mae")}
        summary_results.append(summary)

    output = {
        "grid": GRID,
        "num_samples": n,
        "baseline_mse": baseline_mse,
        "baseline_mae": baseline_mae,
        "best": summary_results[0],
        "all_results": summary_results,
    }

    results_path = results_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

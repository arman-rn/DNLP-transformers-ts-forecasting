"""Smoke test for TTT implementation on Chronos-2.

Run from project root:
    python scripts/test_ttt.py
    python scripts/test_ttt.py --use_etth1
    python scripts/test_ttt.py --n_mask 5 --ttt_steps 3 --lr 1e-4
"""

import argparse
import copy
import sys
from pathlib import Path

# Ensure project root is on sys.path so both src and data imports work
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
from chronos import Chronos2Pipeline

from src.ttt import TTTChronos


def make_synthetic_context(length=512, seed=42):
    """Sine wave + linear trend + noise. More meaningful than pure random."""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)
    series = (
        10.0 * np.sin(2 * np.pi * t / 50)
        + 0.01 * t
        + rng.normal(0, 0.5, length).astype(np.float32)
    )
    return torch.tensor(series)


def load_etth1_context(context_length):
    """Load first context_length values from ETTh1 test set."""
    from data.download import get_etth1

    dataset = get_etth1()
    entry = list(dataset.test)[0]
    ts = np.array(entry["target"], dtype=np.float32)
    return torch.tensor(ts[:context_length])


def main():
    parser = argparse.ArgumentParser(description="Smoke test for TTT on Chronos-2")
    parser.add_argument("--model", default="amazon/chronos-2", help="HF model name or local path")
    parser.add_argument("--device", default=None, help="Device (default: cuda if available)")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=96)
    parser.add_argument("--n_mask", type=int, default=3)
    parser.add_argument("--ttt_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_etth1", action="store_true", help="Use ETTh1 data instead of synthetic")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    print(f"Loading Chronos-2 from '{args.model}' on {device} ...")
    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=torch.float32,
    )
    model = pipeline.model
    print(f"  Model class : {type(model).__name__}")
    print(f"  Patch sizes : input={model.chronos_config.input_patch_size}, "
          f"output={model.chronos_config.output_patch_size}")
    print(f"  Quantiles   : {model.chronos_config.quantiles}")
    total_params = sum(p.numel() for p in model.parameters())
    out_emb_params = sum(p.numel() for p in model.output_patch_embedding.parameters())
    in_emb_params = sum(p.numel() for p in model.input_patch_embedding.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  Input embedding params:  {in_emb_params:,} ({in_emb_params / total_params * 100:.1f}%)")
    print(f"  Output embedding params: {out_emb_params:,} ({out_emb_params / total_params * 100:.1f}%)")

    # ---- Prepare context ----
    if args.use_etth1:
        print(f"\nLoading ETTh1 context (first {args.context_length} values) ...")
        context = load_etth1_context(args.context_length)
    else:
        print(f"\nUsing synthetic context (sine + trend + noise, length={args.context_length}) ...")
        context = make_synthetic_context(length=args.context_length)

    print(f"  shape={tuple(context.shape)}, dtype={context.dtype}")
    print(f"  range=[{context.min():.4f}, {context.max():.4f}], "
          f"mean={context.mean():.4f}, std={context.std():.4f}")

    # ---- Snapshot output_patch_embedding BEFORE TTT ----
    pre_state = copy.deepcopy(model.output_patch_embedding.state_dict())

    # ---- Run TTT ----
    print(f"\n{'='*60}")
    print(f"Running TTT  n_mask={args.n_mask}  steps={args.ttt_steps}  lr={args.lr}")
    print(f"{'='*60}")

    ttt_model = TTTChronos(
        pipeline,
        n_mask=args.n_mask,
        ttt_steps=args.ttt_steps,
        lr=args.lr,
    )
    forecast = ttt_model.predict(context, prediction_length=args.prediction_length)

    # ---- Inspect forecast ----
    print(f"\n{'='*60}")
    print("Forecast results")
    print(f"{'='*60}")
    print(f"  pipeline.predict() returned {len(forecast)} tensor(s)")
    for i, f in enumerate(forecast):
        print(f"  forecast[{i}]: shape={tuple(f.shape)}, dtype={f.dtype}")
        # Expected shape: (n_variates, n_quantiles, prediction_length)
        if f.ndim == 3:
            quantiles = model.chronos_config.quantiles
            if 0.5 in quantiles:
                median_idx = quantiles.index(0.5)
            else:
                median_idx = len(quantiles) // 2
            median = f[0, median_idx]  # first variate, median quantile
            print(f"    median (q=0.5) range: [{median.min():.4f}, {median.max():.4f}]")

    # ---- Verify embedding reset ----
    print(f"\n{'='*60}")
    print("Embedding reset verification")
    print(f"{'='*60}")
    post_state = model.output_patch_embedding.state_dict()

    all_match = True
    for key in pre_state:
        if not torch.equal(pre_state[key], post_state[key]):
            all_match = False
            diff = (pre_state[key] - post_state[key]).abs().max().item()
            print(f"  MISMATCH  '{key}': max abs diff = {diff:.6e}")

    if all_match:
        print("  PASS  All embedding weights restored to pre-TTT values.")
    else:
        print("  FAIL  Embedding weights differ from pre-TTT snapshot!")

    print("\nDone.")


if __name__ == "__main__":
    main()

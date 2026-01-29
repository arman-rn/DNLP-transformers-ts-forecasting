"""Inspect Chronos-2 architecture to find the embedding layer for TTT."""

import torch
from chronos import Chronos2Pipeline


def main():
    print("Loading Chronos-2 pipeline...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.float32,
    )

    model = pipeline.model
    print(f"\nModel type: {type(model)}")
    print(f"Model class name: {type(model).__name__}")

    # Overall structure (top-level children only)
    print("\n" + "=" * 70)
    print("TOP-LEVEL MODULES")
    print("=" * 70)
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"  {name}: {type(child).__name__} ({params:,} params)")

    # All named modules with param counts
    print("\n" + "=" * 70)
    print("ALL NAMED MODULES (with parameters)")
    print("=" * 70)
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            print(f"  {name}: {type(module).__name__} ({params:,} params)")

    # Candidate embedding layers
    keywords = ["embed", "proj", "input", "patch", "token"]
    print("\n" + "=" * 70)
    print(f"CANDIDATE EMBEDDING LAYERS (keywords: {keywords})")
    print("=" * 70)
    for name, module in model.named_modules():
        if any(kw in name.lower() for kw in keywords):
            params = sum(p.numel() for p in module.parameters())
            print(f"\n  {name}: {type(module).__name__} ({params:,} params)")
            for pname, param in module.named_parameters(recurse=False):
                print(f"    .{pname}: shape={list(param.shape)}, dtype={param.dtype}")

    # All named parameters for full picture
    print("\n" + "=" * 70)
    print("ALL NAMED PARAMETERS")
    print("=" * 70)
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        print(f"  {name}: shape={list(param.shape)} ({param.numel():,} params)")
    print(f"\nTotal parameters: {total:,}")


if __name__ == "__main__":
    main()

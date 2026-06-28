# Adaptive Tokenization in Time Series Foundation Models
### via Volatility-Aware Patching

> Companion code for the paper *"Adaptive Tokenization in Time Series Foundation Models via Volatility-Aware Patching"* (DNLP, 2026).

This repository contains the implementation and evaluation pipeline for **Patch Adaptation (PA)**, a non-uniform tokenization framework for the Chronos-2 time series foundation model. PA preserves constant patch length while adapting the stride to local signal volatility, leaving the pretrained encoder weights untouched.

---

## Repository Structure

```
.
├── chronos2WOA/
│   ├── finetuning.py        # main training & evaluation script
│   ├── modelWOA.py          # PA model implementation
│   ├── configWOA.py         # configuration schema
│   ├── model.py             # vanilla Chronos-2 model
│   └── dataset.py           # data loading & patching
├── notebooks/
│   ├── compare_models_electricity.ipynb
│   └── compare_models_solar.ipynb
├── requirements.txt
└── README.md
```

---

## Setup

We assume a single **NVIDIA RTX 4090 (24 GB)** with CUDA ≥ 12.1.

```bash
# 1. Clone
git clone https://github.com/arman-rn/DNLP-transformers-ts-forecasting.git
cd DNLP-transformers-ts-forecasting

# 2. Environment
conda create -n pa python=3.10 -y
conda activate pa
pip install -r requirements.txt

# 3. Download the pretrained Chronos-2 checkpoint
#    (the paths in finetuning.py expect: /mnt/share/.../chronos/amazonweights_v2)
#    Adjust LOCAL_DIR and FINAL_DEST in finetuning.py if your storage differs.
```

The datasets (Solar Energy Alabama, Electricity Load Diagrams) are downloaded automatically from public mirrors on the first run.

---

## Quick Start — One Command to Reproduce the Paper

```bash
python chronos2WOA/finetuning.py
```

This runs the **full ablation suite**: vanilla baseline + 5 PA variants × 2 datasets = 12 fine-tuning runs, each from the same pretrained checkpoint under identical optimization schedules. Total runtime ≈ 16 h on a single RTX 4090. All metrics are logged to Weights & Biases (`project: chronos2-woa`).

---

## Running a Single Configuration

`finetuning.py` exposes a `train(...)` function that takes the variant identifier and dataset. To run just one model (e.g., for a quick smoke test):

```python
from chronos2WOA.finetuning import train

# Vanilla baseline on Solar
train(
    sensitivity_val=0.0,
    output_name="chronos2og",
    description="Standard Chronos-2 baseline",
    which="standard",
)

# PA Per-Seq on Solar
train(
    sensitivity_val=15,
    output_name="chronos2WOA_perseq",
    description="PA per-sequence volatility",
    which="woa",
    per_sequence_volatility=True,
)
```

---

## Variants Implemented

All variants share the same adaptive patching core; they differ in regularization or gating.

| Variant | Key flags | Loss objective |
|---|---|---|
| **Vanilla**       | `which="standard"`                                                                  | Pinball |
| **Per-Seq**       | `which="woa", per_sequence_volatility=True`                                         | Pinball |
| **ReZero**        | `... use_rezero_stride=True`                                                        | Pinball + ReZero gate |
| **Coverage**      | `... coverage_lambda=0.1`                                                           | Pinball + coverage penalty |
| **Vol-Weight**    | `... volatility_weighting=True`                                                     | Volatility-weighted pinball |
| **Distillation**  | `... distill_lambda=1.0`                                                            | Pinball + distillation from vanilla teacher |

PA hyperparameters are held constant across all variants: `sensitivity λ = 15`, `min_stride = 1`, `patch_size P = 16`.

---

## Comparing Vanilla vs PA — End-to-End

```bash
# 1. Fine-tune both models on the same dataset
python -c "
from chronos2WOA.finetuning import train
train(0.0, 'electricity_vanilla', 'Vanilla baseline', which='standard')
train(15,  'electricity_perseq',  'PA Per-Seq',       which='woa', per_sequence_volatility=True)
"

# 2. Open the comparison notebook
jupyter notebook notebooks/compare_models_electricity.ipynb
```

The notebook loads both checkpoints, runs them on the same test windows, and reports a single table containing MAE, MSE, WQL, and per-step error curves.

---

## Training Protocol (Identical Across All Runs)

| Hyperparameter | Value |
|---|---|
| Optimizer            | AdamW             |
| Learning rate        | `1e-4`            |
| Weight decay         | `0.01`            |
| LR schedule          | Cosine, 1000-step warmup |
| Batch size           | 32                |
| Gradient clipping    | `‖g‖ ≤ 1.0`       |
| Steps per dataset    | 50,000            |
| Precision            | bfloat16 (autocast forward) |
| Seed                 | `1738`            |

This **like-for-like fine-tuning regime** means that every reported delta between vanilla and a PA variant is attributable to the patching mechanism alone, not to differing optimization budgets.

---

## Expected Output

After a successful run, you should obtain results matching Tables I (Solar) and II (Electricity) of the paper:

| Dataset       | Vanilla MAE / MSE / WQL          | Best PA MAE / MSE / WQL                          |
|---------------|----------------------------------|--------------------------------------------------|
| Solar         | **138.66** / 706,080 / 41.55     | 176.47 / **600,194** / **4.09** *(Distillation)* |
| Electricity   | **4.45** / 67.41 / 37.49         | 4.74 / **61.73** / **35.63** *(ReZero)*          |

Numbers within ±1% are expected as a function of seed and CUDA non-determinism.

---

## Citation

```bibtex
@inproceedings{elezi2026pa,
  title  = {Adaptive Tokenization in Time Series Foundation Models via Volatility-Aware Patching},
  author = {Elezi, Kevin and Lotf Ranaei, Amirhossein and Gentile, Alessandro and Sammartino, Stefano and Vazirpanah, Niloofar},
  year   = {2026},
  note   = {DNLP — Politecnico di Torino, WORK\_OF\_ART}
}
```

---

## Contact

For questions about reproducing the results, please contact any of the authors via their `@studenti.polito.it` address.

<p align="center">
  <img src="schema.png" alt="Volatility-Aware Patch Adaptation Pipeline" width="100%">
</p>

# Adaptive Tokenization in Time Series Foundation Models via Volatility-Aware Patching

University project · **DNLP** (Deep Natural Language Processing) · **Politecnico di Torino**, M.Sc. Data Science and Engineering · A.Y. 2025/26.

We modify the tokenization front-end of **Chronos-2** so that patch stride adapts to local volatility, leaving the pretrained encoder weights untouched. The four-stage pipeline is summarized in the figure above.

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
├── schema.png               # PA pipeline diagram (Fig. 1 of the report)
├── requirements.txt
└── README.md
```

---

## Setup

Single **NVIDIA RTX 4090 (24 GB)**, CUDA ≥ 12.1.

```bash
git clone https://github.com/arman-rn/DNLP-transformers-ts-forecasting.git
cd DNLP-transformers-ts-forecasting

conda create -n pa python=3.10 -y && conda activate pa
pip install -r requirements.txt
```

Adjust the pretrained-checkpoint paths inside `finetuning.py` (`LOCAL_DIR`, `FINAL_DEST`) if your storage layout differs from ours. The Electricity Load Diagrams dataset is downloaded automatically on first run.

---

## Running the Experiments

All experiments target the **Electricity Load Diagrams** dataset (321 households, hourly, ~8.4M data points). `finetuning.py` runs the full ablation when executed directly:

```bash
python chronos2WOA/finetuning.py
```

This fine-tunes the vanilla baseline plus all five PA variants from the same pretrained Chronos-2 checkpoint under identical optimization schedules. Total runtime ≈ 8 h on a single RTX 4090.

To run only one configuration, call `train(...)` programmatically:

```python
from chronos2WOA.finetuning import train

# Vanilla baseline
train(0.0, "chronos2og", "Standard Chronos-2", which="standard")

# PA Per-Seq
train(15, "chronos2WOA_perseq", "PA per-sequence volatility",
      which="woa", per_sequence_volatility=True)
```

---

## Variants

All variants share the adaptive patching core (`λ = 15`, `min_stride = 1`, `P = 16`); they differ only in regularization or gating.

| Variant            | Key flags                                    | Loss objective                       |
|--------------------|----------------------------------------------|--------------------------------------|
| **Vanilla**        | `which="standard"`                           | Pinball                              |
| **Per-Seq**        | `per_sequence_volatility=True`               | Pinball                              |
| **ReZero**         | `... use_rezero_stride=True`                 | Pinball + ReZero gate                |
| **Coverage**       | `... coverage_lambda=0.1`                    | Pinball + coverage penalty           |
| **Vol-Weight**     | `... volatility_weighting=True`              | Volatility-weighted pinball          |
| **Distillation**   | `... distill_lambda=1.0`                     | Pinball + vanilla-teacher distill    |

---

## Training Protocol (Identical Across All Runs)

| Hyperparameter         | Value                       |
|------------------------|-----------------------------|
| Optimizer              | AdamW, lr `1e-4`, wd `0.01` |
| LR schedule            | Cosine, 1k-step warmup      |
| Batch size             | 32                          |
| Gradient clipping      | `‖g‖ ≤ 1.0`                 |
| Steps                  | 50,000                      |
| Precision              | bfloat16 (autocast forward) |
| Random seed            | `1738`                      |

Every model — vanilla and PA — uses this exact schedule, so any reported delta is attributable to the patching mechanism alone.

---

## Expected Results — Electricity

| Chronos-2     | MAE ↓    | MSE ↓     | WQL ↓     |
|---------------|----------|-----------|-----------|
| Standard      | **4.45** | 67.41     | 37.49     |
| Per-Seq       | 4.83     | 64.48     | 36.13     |
| ReZero        | 4.74     | **61.73** | **35.63** |
| Coverage      | 4.82     | 64.39     | 36.01     |
| Vol-Weight    | 4.82     | 63.59     | 36.15     |
| Distillation  | 4.84     | 64.51     | 36.05     |

Variations of ±1 % across reruns are expected (CUDA non-determinism). ReZero achieves the best balance: lowest MSE and WQL with the smallest MAE regression.

---

## Authors

Kevin Elezi · Amirhossein Lotf Ranaei · Alessandro Gentile · Stefano Sammartino · Niloofar Vazirpanah
*`@studenti.polito.it`*

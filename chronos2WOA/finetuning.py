import random

import numpy as np
import torch

SEED = 1738
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

import gc
import math
import os
import shutil

import pandas as pd
import wandb

# --- IMPORT YOUR MODULES ---
from configWOA import Chronos2CoreConfig, Chronos2ForecastingConfig
from dataset import Chronos2Dataset
from model import Chronos2Model as Chronos2ModelOG
from modelWOA import Chronos2Model
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("WANDB_API_KEY", "")
if not os.environ["WANDB_API_KEY"]:
    wandb.login()

# --- CONFIG ---
MAX_STEPS = 4000
GRAD_ACCUMULATION = 32
VAL_SAMPLES = 250  # sampled sequentially 'validation' mode
TEST_SAMPLES = 500  # sampled sequentially 'validation' mode
# 1 samples is [x_i,y_i] = [context_length, prediction_length] = [1024, 96]

LEARNING_RATE = 1e-6

PREDICTION_LENGTH = 96
CONTEXT_LENGHT = 1024
PATCH_SIZE = 16
PREDICTED_PATCHES = math.ceil(
    PREDICTION_LENGTH / PATCH_SIZE
)  # how many patches in output per sample
INPUT_PATCHES = math.ceil(
    CONTEXT_LENGHT / PATCH_SIZE
)  # how many pathes in input per sample

BATCH_SIZE = 4  # how many sequences (of length CONTEXT_LENGHT) you want to process in parallel during training
# NOTE : in the paper notes, look how they pass from [batch, feature, time] to [batch * feature, time] when passing the data to Chronos2Dataset

# --- STABILITY CONFIG ---
PATIENCE = 6
EVAL_INTERVAL = 50
MAX_GRAD_NORM = 1.0  # <--- NEW: The Speed Limit for gradients

# --- PATHS ---
LOCAL_DIR = "/tmp/kevin_chronos_run"
FINAL_DEST = "/mnt/share/kelezi/DNLP-transformers-ts-forecasting/finetuned_weights"


def calculate_metrics(model, dataset, max_samples, desc="Eval"):
    model.eval()
    # Increased batch size to speed up eval; 1 is too slow!
    loader = DataLoader(dataset, batch_size=None)

    losses = []
    abs_errors = []
    sq_errors = []
    wql_errors = []

    quantiles = torch.tensor(
        model.chronos_config.quantiles, device="cuda", dtype=torch.float32
    )
    median_idx = model.chronos_config.quantiles.index(0.5)

    print(f"evaluation of ({max_samples} samples)...")
    num_samples_processed = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if num_samples_processed >= max_samples:
                break  # avoid to validate the whole validation context #avoid to validate the whole validation context #avoid to validate the whole validation context

            batch = {
                k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            context = batch.get("context")
            target = batch.get("future_target")
            fut_cov = batch.get("future_covariates")

            outputs = model(
                context=context,
                future_target=target,
                num_output_patches=PREDICTED_PATCHES,
                future_covariates=fut_cov,
            )

            # 1. Use the model's internal loss for consistency with training-------------
            losses.append(outputs.loss.item())

            # 2. WQL Calculation (Chronos-2 Standard)------------------------------------
            # target: [B, T] -> y_true: [B, 1, T]
            # quantile_preds: [B, Q, T]
            y_true = target.unsqueeze(1)
            y_pred = outputs.quantile_preds

            # Pinball loss
            errors = y_true - y_pred
            q = quantiles.view(1, -1, 1)
            q_loss = torch.max(
                q * errors, (q - 1) * errors
            )  # max(underprediction, overprediction)
            # for q = 0.90 it penalizes underpredictions more than overpredictions, for q = 0.10 it penalizes overpredictions more than underpredictions, for q = 0.50 it penalizes both equally (median)

            # Weighted normalization: sum of errors / sum of absolute targets
            # We sum over Q and T, then average over the Batch
            sum_q_loss = torch.sum(q_loss, dim=(1, 2))  # [Batch]
            sum_y_true = torch.sum(torch.abs(y_true), dim=(1, 2))  # [Batch]

            batch_wql = 2 * sum_q_loss / (sum_y_true + 1e-6)
            wql_errors.extend(batch_wql.cpu().tolist())

            # 3. Median Point Metrics (MAE/MSE)------------------------------------------
            median_pred = outputs.quantile_preds[:, median_idx, :]
            min_len = min(
                median_pred.shape[1], target.shape[1]
            )  # in case PREDICTION_LENGHT > or < PREDICTED_PATCHES * PATCH_SIZE (in our case they are aligned but this is a safety check)

            point_error = median_pred[:, :min_len] - target[:, :min_len]
            abs_errors.append(torch.mean(torch.abs(point_error)).item())
            sq_errors.append(torch.mean(point_error**2).item())

            num_samples_processed += context.shape[0]

    avg_loss = np.mean(losses)
    avg_mae = np.mean(abs_errors)
    avg_mse = np.mean(sq_errors)
    avg_wql = np.mean(wql_errors) if wql_errors else 0.0

    print(
        f"   ✅ {desc} -> Loss: {avg_loss:.4f} | WQL: {avg_wql:.4f} | MAE: {avg_mae:.4f}"
    )
    return avg_loss, avg_mae, avg_mse, avg_wql


def train(
    sensitivity_val, output_name, description, which="standard", uniform_stride=None, per_sequence_volatility=False
):
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(SEED)

    wandb.init(
        project="chronos2-woa",
        name=output_name,
        tags=["electricity", which],
    )

    temp_output_path = os.path.join(LOCAL_DIR, output_name)
    best_model_path = os.path.join(LOCAL_DIR, f"{output_name}_BEST.pt")

    if os.path.exists(temp_output_path):
        shutil.rmtree(temp_output_path)
    os.makedirs(temp_output_path, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🚀 STARTING STABLE RUN: {description}")
    print(f"   Sensitivity: {sensitivity_val}")
    print(f"   Uniform Stride: {uniform_stride}")
    print(f"   Per-Sequence Volatility: {per_sequence_volatility}")
    print(f"   Gradient Clipping: {MAX_GRAD_NORM}")
    print(f"{'=' * 60}\n")

    # 2. CONFIG
    chronos_config_settings = Chronos2ForecastingConfig(
        input_patch_size=PATCH_SIZE,
        output_patch_size=PATCH_SIZE,
        context_length=CONTEXT_LENGHT,
        time_encoding_scale=2048,
        prediction_length=PREDICTION_LENGTH,
        num_samples=20,
        quantiles=[
            0.01,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
        ],
        use_reg_token=True,
        sensitivity=sensitivity_val,
        min_stride=1,
        input_patch_stride=PATCH_SIZE,
        uniform_stride=uniform_stride,
        per_sequence_volatility=per_sequence_volatility,
    )

    config = Chronos2CoreConfig(
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        d_kv=64,
        dropout_rate=0.1,
        chronos_config=chronos_config_settings.__dict__,
    )

    wandb.config.update(
        {
            **chronos_config_settings.__dict__,
            "MAX_STEPS": MAX_STEPS,
            "GRAD_ACCUMULATION": GRAD_ACCUMULATION,
            "LEARNING_RATE": LEARNING_RATE,
            "BATCH_SIZE": BATCH_SIZE,
            "PATIENCE": PATIENCE,
            "EVAL_INTERVAL": EVAL_INTERVAL,
            "MAX_GRAD_NORM": MAX_GRAD_NORM,
            "PREDICTION_LENGTH": PREDICTION_LENGTH,
            "CONTEXT_LENGHT": CONTEXT_LENGHT,
            "PATCH_SIZE": PATCH_SIZE,
        }
    )

    if which == "woa":
        model = Chronos2Model(config)
    elif which == "standard":
        model = Chronos2ModelOG(config)

    local_folder = "/mnt/share/kelezi/chronos/amazonweights_v2"
    weights_path = os.path.join(local_folder, "model.safetensors")
    results = model.load_state_dict(load_file(weights_path), strict=False)
    print(f"{which} additional layers initialized: {results.missing_keys}")

    model.to("cuda")

    url = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz"
    df = pd.read_csv(url, compression="gzip", header=None)

    n_series = df.shape[1]
    timeserie_duration = df.shape[0]
    train_idx = int(timeserie_duration * 0.8)
    val_idx = int(timeserie_duration * 0.9)

    portion = (CONTEXT_LENGHT + PREDICTION_LENGTH) / timeserie_duration
    print(
        f"Series: {n_series} | Length: {timeserie_duration} | Context + Prediction portion: {portion:.2%}"
    )

    # Each column is one household; treat each as an independent univariate series.
    # Chronos2Dataset accepts a list of {"target": ...} dicts, one per series.
    train_data, val_data, test_data = [], [], []
    for col_idx in range(n_series):
        series = df.iloc[:, col_idx].values.astype(np.float32)
        train_data.append({"target": torch.from_numpy(series[:train_idx])})
        val_data.append({"target": torch.from_numpy(series[train_idx:val_idx])})
        test_data.append({"target": torch.from_numpy(series[val_idx:])})
    # if we had past and future covariates it would be something like this:
    # train_data = [{"target": [train_vals], "past_covariates": past_cov_train, "future_covariates": fut_cov_train}]

    # train mode it will picks random samples withing the context (augmentation), 'validation' mode it picks samples sequantially trough the entire val/test context (no augmentation, full coverage)
    train_ds = Chronos2Dataset(
        inputs=train_data,
        context_length=CONTEXT_LENGHT,
        prediction_length=PREDICTION_LENGTH,
        batch_size=BATCH_SIZE,
        output_patch_size=PATCH_SIZE,
        min_past=64,
        mode="train",
    )
    val_ds = Chronos2Dataset(
        inputs=val_data,
        context_length=CONTEXT_LENGHT,
        prediction_length=PREDICTION_LENGTH,
        batch_size=BATCH_SIZE,
        output_patch_size=PATCH_SIZE,
        min_past=64,
        mode="validation",
    )
    test_ds = Chronos2Dataset(
        inputs=test_data,
        context_length=CONTEXT_LENGHT,
        prediction_length=PREDICTION_LENGTH,
        batch_size=BATCH_SIZE,
        output_patch_size=PATCH_SIZE,
        min_past=64,
        mode="validation",
    )
    # NOTE : as notes written in paper , now we have [bath * features, time], figure 1.

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS)
    scaler = torch.amp.GradScaler("cuda")

    # Chronos2Dataset already returnes batches, avoid redundancy
    train_loader = DataLoader(train_ds, batch_size=None)

    model.train()
    optimizer.zero_grad()
    progress_bar = tqdm(range(MAX_STEPS), desc="Training")

    current_step = 0
    accum_steps = 0
    running_loss = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    stop_training = False

    # start training
    while current_step < MAX_STEPS and not stop_training:
        for batch in train_loader:
            if current_step >= MAX_STEPS or stop_training:
                break

            batch = {
                k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # batch = { "context": v, "future_target": v, "future_covariates": v, "group_ids": v, "num_output_patches": v }

            context = batch.get("context")
            target = batch.get("future_target")
            fut_cov = batch.get("future_covariates")

            # forward pass
            with torch.amp.autocast("cuda"):
                outputs = model(
                    context=context,
                    future_target=target,
                    num_output_patches=PREDICTED_PATCHES,
                    future_covariates=fut_cov,
                )
                loss = (
                    outputs.loss / GRAD_ACCUMULATION
                )  # scale down the loss such that the loss at GRAD_ACCUMULATION step has a magnitude similar to the non-accumulated case
            scaler.scale(
                loss
            ).backward()  # computing and adding new gradients to the computational graph, not updating weights yet
            running_loss += loss.item() * GRAD_ACCUMULATION
            accum_steps += 1

            # NOTE : until now we just computed and accumulated the gradients (tiny arrows to each parameter) trough the loss, didnt update the weights yet !
            # now is time to update weights, the optimization step
            if accum_steps % GRAD_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM
                )

                scaler.step(
                    optimizer
                )  # take the massive and accumulated gradients and update the weights accordingly
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()  # then clear the accumulated gradients

                current_step += 1
                current_loss = running_loss / accum_steps
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

                wandb.log(
                    {
                        "train/loss": current_loss,
                        "train/grad_norm": grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm,
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=current_step,
                )

                # stabilizing the logging (better to log every x steps instead of every step when using grad accumulation)
                if current_step % EVAL_INTERVAL == 0:
                    running_loss = 0.0
                    accum_steps = 0

                    val_loss, val_mae, val_mse, val_wql = calculate_metrics(
                        model, val_ds, VAL_SAMPLES, desc=f"Step {current_step} Check"
                    )

                    if val_loss < best_val_loss:
                        print(
                            f" model did improve ({val_loss:.4f} < {best_val_loss:.4f}) , saving locally..."
                        )
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1
                        print(
                            f" model did NOT improve. Patience: {patience_counter}/{PATIENCE}"
                        )

                        if patience_counter >= PATIENCE:
                            print(" Early Stopping Triggered!")
                            stop_training = True

                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/mae": val_mae,
                            "val/mse": val_mse,
                            "val/wql": val_wql,
                            "val/best_loss": best_val_loss,
                            "val/patience": patience_counter,
                        },
                        step=current_step,
                    )

                    if stop_training:
                        break

                    model.train()

    # end training, upload best model and test
    if os.path.exists(best_model_path):
        print("♻️  Reloading Best Model for Final Test...")
        model.load_state_dict(torch.load(best_model_path))

    # 8. SAVE
    print(f"....Saving final artifact to local temp: {temp_output_path}...")
    model.save_pretrained(temp_output_path)

    final_network_path = os.path.join(FINAL_DEST, f"{output_name}")
    print(f"....Moving to Network Drive: {final_network_path}...")
    if os.path.exists(final_network_path):
        shutil.rmtree(final_network_path)
    shutil.copytree(temp_output_path, final_network_path)

    # 9. TEST
    loss, mae, mse, wql = calculate_metrics(
        model, test_ds, TEST_SAMPLES, desc="FINAL TEST"
    )

    wandb.log(
        {
            "test/loss": loss,
            "test/mae": mae,
            "test/mse": mse,
            "test/wql": wql,
        }
    )
    wandb.finish()

    return loss, mae, mse, wql


# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # w_loss, w_mae, w_mse, w_wql = train(15, "chronos2WOA", "WOA Model", which="woa")
    # s_loss, s_mae, s_mse, s_wql = train(
    #     0.0, "chronos2og", "Standard Model", which="standard"
    # )
    u_loss, u_mae, u_mse, u_wql = train(
        0.0,
        "chronos2WOA_stride8",
        "WOA Uniform Stride 8 (overlap isolation)",
        which="woa",
        uniform_stride=8,
    )

    p_loss, p_mae, p_mse, p_wql = train(
        15,
        "chronos2WOA_perseq",
        "WOA Per-Sequence Volatility (leak fix)",
        which="woa",
        per_sequence_volatility=True,
    )

    # print(
    #     f"{'Loss':<10} | {s_loss:<12.4f} | {w_loss:<12.4f} | {'WOA' if w_loss < s_loss else 'Standard'}"
    # )
    # print(
    #     f"{'MAE':<10} | {s_mae:<12.4f} | {w_mae:<12.4f} | {'WOA' if w_mae < s_mae else 'Standard'}"
    # )
    # print(
    #     f"{'MSE':<10} | {s_mse:<12.4f} | {w_mse:<12.4f} | {'WOA' if w_mse < s_mse else 'Standard'}"
    # )
    # print(
    #     f"{'WQL':<10} | {s_wql:<12.4f} | {w_wql:<12.4f} | {'WOA' if w_wql < s_wql else 'Standard'}"
    # )

    print("\nUniform-stride-8 ablation final test results:")
    print(f"  Loss: {u_loss:.4f}")
    print(f"  MAE:  {u_mae:.4f}")
    print(f"  MSE:  {u_mse:.4f}")
    print(f"  WQL:  {u_wql:.4f}")

    print(f"\nPer-sequence volatility ablation final test results:")
    print(f"  Loss: {p_loss:.4f}")
    print(f"  MAE:  {p_mae:.4f}")
    print(f"  MSE:  {p_mse:.4f}")
    print(f"  WQL:  {p_wql:.4f}")

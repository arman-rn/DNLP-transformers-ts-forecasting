import torch
import os
import pandas as pd
import numpy as np
import gc
import random
import shutil
import math
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# --- IMPORT YOUR MODULES ---
from configWOA import Chronos2ForecastingConfig, Chronos2CoreConfig
from modelWOA import Chronos2Model
from dataset import Chronos2Dataset

# --- CONFIG ---
SEED = 42
MAX_STEPS = 5000        
GRAD_ACCUMULATION = 16  
VAL_SAMPLES = 250        
TEST_SAMPLES = 500      
LEARNING_RATE = 1e-5
PREDICTION_LENGTH = 96
PATCH_SIZE = 16
REQUIRED_PATCHES = math.ceil(PREDICTION_LENGTH / PATCH_SIZE)

# --- STABILITY CONFIG ---
PATIENCE = 6            # Increased Patience to allow recovery
EVAL_INTERVAL = 250    
MAX_GRAD_NORM = 1.0     # <--- NEW: The Speed Limit for gradients

# --- PATHS ---
LOCAL_DIR = "/tmp/kevin_chronos_run" 
FINAL_DEST = "/mnt/share/kelezi/DNLP-transformers-ts-forecasting/finetuned_weights"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pass_through_collator(batch):
    real_batch = batch[0]
    processed_batch = {}
    for k, v in real_batch.items():
        if isinstance(v, int):
            processed_batch[k] = torch.tensor([v]) 
        elif isinstance(v, torch.Tensor) and v.ndim == 0:
            processed_batch[k] = v.unsqueeze(0)
        else:
            processed_batch[k] = v
    return processed_batch

# --- METRIC CALCULATOR ---
def calculate_metrics(model, dataset, num_samples, desc="Eval"):
    model.eval()
    # Increased batch size to speed up eval; 1 is too slow!
    loader = DataLoader(dataset, batch_size=8, collate_fn=pass_through_collator)
    
    losses = []
    abs_errors = [] 
    sq_errors = []  
    wql_errors = [] 
    
    # Ensure quantiles are on the correct device
    quantiles = torch.tensor(model.chronos_config.quantiles, device="cuda", dtype=torch.float32)
    try:
        median_idx = model.chronos_config.quantiles.index(0.5)
    except ValueError:
        median_idx = len(quantiles) // 2 

    print(f"🔎 {desc} ({num_samples} samples)...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i * loader.batch_size >= num_samples: break
            
            batch = {k: v.to("cuda") for k, v in batch.items()}
            
            # --- FIXED LOGIC ---
            target = batch.get("target")
            if target is None:
                target = batch.get("future_target")
            if target is None:
                target = batch.get("labels")
            # -------------------
            
            if target is None: continue  

            
            outputs = model(
                context=batch["context"],
                future_target=target,
                num_output_patches=REQUIRED_PATCHES,
                future_covariates=batch.get("future_covariates")
            )
            
            # 1. Use the model's internal loss for consistency with training
            losses.append(outputs.loss.item())
            
            # 2. WQL Calculation (Chronos-2 Standard)
            # target: [B, T] -> y_true: [B, 1, T]
            # quantile_preds: [B, Q, T]
            y_true = target.unsqueeze(1) 
            y_pred = outputs.quantile_preds 
            
            # Pinball loss
            errors = y_true - y_pred
            q_loss = torch.max(
                quantiles.view(1, -1, 1) * errors, 
                (quantiles.view(1, -1, 1) - 1) * errors
            )
            
            # Weighted normalization: sum of errors / sum of absolute targets
            # We sum over Q and T, then average over the Batch
            sum_q_loss = torch.sum(q_loss, dim=(1, 2)) # [Batch]
            sum_y_true = torch.sum(torch.abs(y_true), dim=(1, 2)) # [Batch]
            
            batch_wql = 2 * sum_q_loss / (sum_y_true + 1e-6)
            wql_errors.extend(batch_wql.cpu().tolist())

            # 3. Median Point Metrics (MAE/MSE)
            median_pred = outputs.quantile_preds[:, median_idx, :]
            min_len = min(median_pred.shape[1], target.shape[1])
            
            point_error = median_pred[:, :min_len] - target[:, :min_len]
            abs_errors.append(torch.mean(torch.abs(point_error)).item())
            sq_errors.append(torch.mean(point_error**2).item())

    avg_loss = np.mean(losses)
    avg_mae = np.mean(abs_errors)
    avg_mse = np.mean(sq_errors)
    avg_wql = np.mean(wql_errors) if wql_errors else 0.0
    
    print(f"   ✅ {desc} -> Loss: {avg_loss:.4f} | WQL: {avg_wql:.4f} | MAE: {avg_mae:.4f}")
    return avg_loss, avg_mae, avg_mse, avg_wql

def train(sensitivity_val, output_name, description):
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(SEED)
    
    temp_output_path = os.path.join(LOCAL_DIR, output_name)
    best_model_path = os.path.join(LOCAL_DIR, f"{output_name}_BEST.pt")
    
    if os.path.exists(temp_output_path): shutil.rmtree(temp_output_path)
    os.makedirs(temp_output_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING STABLE RUN: {description}")
    print(f"   Sensitivity: {sensitivity_val}")
    print(f"   Gradient Clipping: {MAX_GRAD_NORM}")
    print(f"{'='*60}\n")

    # 2. CONFIG
    chronos_config_settings = Chronos2ForecastingConfig(
        input_patch_size=16, 
        output_patch_size=PATCH_SIZE, 
        context_length=2048,   
        time_encoding_scale=2048, 
        prediction_length=PREDICTION_LENGTH, 
        num_samples=20,
        quantiles=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
        use_reg_token=True, 
        sensitivity=sensitivity_val, 
        min_stride=1, 
        input_patch_stride=16,
    )
    config = Chronos2CoreConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12, d_kv=64, dropout_rate=0.1, chronos_config=chronos_config_settings.__dict__)
    model = Chronos2Model(config)
    
    # 3. LOAD WEIGHTS
    local_folder = "/mnt/share/kelezi/chronos/amazonweights_v2"
    model.load_state_dict(load_file(os.path.join(local_folder, "model.safetensors")), strict=False)
    model.to("cuda")

    # 4. DATA
    csv_path = '/mnt/share/kelezi/chronos/data/Jena/jena_climate_2009_2016.csv'
    df = pd.read_csv(csv_path)
    all_values = df['T (degC)'].values.astype(np.float32)
    split_idx = int(len(all_values) * 0.8)
    val_idx = int(len(all_values) * 0.9)

    train_values = all_values[:split_idx]
    val_values = all_values[split_idx + 2048 : val_idx]
    test_values = all_values[val_idx + 2048 :]
    
    train_data = [{"target": torch.tensor(train_values), "past_covariates": {}, "future_covariates": {}}]
    val_data = [{"target": torch.tensor(val_values), "past_covariates": {}, "future_covariates": {}}]
    test_data = [{"target": torch.tensor(test_values), "past_covariates": {}, "future_covariates": {}}]

    train_ds = Chronos2Dataset(inputs=train_data, context_length=2048, prediction_length=96, batch_size=4, output_patch_size=16, mode="train")
    val_ds = Chronos2Dataset(inputs=val_data, context_length=2048, prediction_length=96, batch_size=4, output_patch_size=16, mode="train")
    test_ds = Chronos2Dataset(inputs=test_data, context_length=2048, prediction_length=96, batch_size=4, output_patch_size=16, mode="train")

    # 5. OPTIMIZER
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS)
    scaler = torch.amp.GradScaler('cuda') 
    
    # 6. TRAINING LOOP
    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=pass_through_collator)
    model.train()
    optimizer.zero_grad()
    progress_bar = tqdm(range(MAX_STEPS), desc="Training")
    
    current_step = 0
    accum_steps = 0
    running_loss = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    stop_training = False
    
    while current_step < MAX_STEPS and not stop_training:
        for batch in train_loader:
            if current_step >= MAX_STEPS or stop_training: break
            
            batch = {k: v.to("cuda") for k, v in batch.items()}
            
            # Safe Target
            target = batch.get("target")
            if target is None: target = batch.get("future_target")
            if target is None: target = batch.get("labels")
            if target is None: continue
            
            fut_cov = batch.get("future_covariates")
            if fut_cov is not None and not isinstance(fut_cov, torch.Tensor): fut_cov = None

            with torch.amp.autocast('cuda'): 
                outputs = model(
                    context=batch["context"],
                    future_target=target,
                    num_output_patches=REQUIRED_PATCHES,
                    future_covariates=fut_cov
                )
                loss = outputs.loss / GRAD_ACCUMULATION
            
            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUMULATION
            accum_steps += 1
            
            if accum_steps % GRAD_ACCUMULATION == 0:
                # --- FIX: GRADIENT CLIPPING ---
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                # ------------------------------
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                current_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{running_loss / accum_steps:.4f}"})
                
                if current_step % 50 == 0:
                    running_loss = 0.0
                    accum_steps = 0
                
                if current_step % EVAL_INTERVAL == 0:
                    val_loss, _, _ , _= calculate_metrics(model, val_ds, VAL_SAMPLES, desc=f"Step {current_step} Check")
                    
                    if val_loss < best_val_loss:
                        print(f"   ⭐ New Best Model! ({val_loss:.4f} < {best_val_loss:.4f}) Saving locally...")
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1
                        print(f"   📉 No Improvement. Patience: {patience_counter}/{PATIENCE}")
                        
                        if patience_counter >= PATIENCE:
                            print("   🛑 Early Stopping Triggered!")
                            stop_training = True
                            break
                    
                    model.train() 

    # 7. RELOAD BEST MODEL
    if os.path.exists(best_model_path):
        print("♻️  Reloading Best Model for Final Test...")
        model.load_state_dict(torch.load(best_model_path))
    
    # 8. SAVE
    print(f"💾 Saving final artifact to local temp: {temp_output_path}...")
    model.save_pretrained(temp_output_path)
    
    final_network_path = os.path.join(FINAL_DEST, f"{output_name}")
    print(f"🚚 Moving to Network Drive: {final_network_path}...")
    if os.path.exists(final_network_path): shutil.rmtree(final_network_path)
    shutil.copytree(temp_output_path, final_network_path)
    
    # 9. TEST
    loss, mae, mse, wql = calculate_metrics(model, test_ds, TEST_SAMPLES, desc="FINAL TEST")
    return loss, mae, mse, wql

if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print("✅ EARLY STOPPING BENCHMARK (Fixed Tensors) ✅")
    
    s_loss, s_mae, s_mse, s_wql = train(0.0, "chronos_jena_STANDARD_scissione", "Standard Model")
    w_loss, w_mae, w_mse, w_wql = train(10.0, "chronos2_PAS10_scissione", "WOA Model")
    
    print("\n" + "#"*60)
    print("🏆 FINAL COMPARISON 🏆")
    # Adjusted widths to accommodate 'WQL'
    print(f"{'Metric':<10} | {'Standard':<12} | {'WOA (Yours)':<12} | {'Winner':<10}")
    print("-" * 60)
    
    # Standard Metrics
    print(f"{'Loss':<10} | {s_loss:<12.4f} | {w_loss:<12.4f} | {'WOA' if w_loss < s_loss else 'Standard'}")
    print(f"{'MAE':<10} | {s_mae:<12.4f} | {w_mae:<12.4f} | {'WOA' if w_mae < s_mae else 'Standard'}")
    print(f"{'MSE':<10} | {s_mse:<12.4f} | {w_mse:<12.4f} | {'WOA' if w_mse < s_mse else 'Standard'}")
    
    # Added Weighted Quantile Loss (WQL)
    # Assuming variables s_wql and w_wql are already calculated in your script
    print(f"{'WQL':<10} | {s_wql:<12.4f} | {w_wql:<12.4f} | {'WOA' if w_wql < s_wql else 'Standard'}")
    
    print("#"*60)
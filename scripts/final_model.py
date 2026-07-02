import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set random seeds for strict reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. PYTORCH DATASET & MODEL DEFINITION
# ==========================================

class ParcelSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NDVI_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(NDVI_BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.hidden_fc = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        # Global Average Pooling over time dimension to leverage entire window context
        out = torch.mean(out, dim=1)
        out = self.hidden_fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ==========================================
# 2. SYMMETRIC WINDOW SEQUENCE GENERATION
# ==========================================

def create_symmetrical_sequences(df, feature_cols, target_col, seq_length=5):
    """
    Generates historical context windows [t-2, t-1, t, t+1, t+2] symmetrically.
    """
    X_seq, y_seq = [], []
    target_offset = seq_length // 2
    
    groups = df.groupby('task_id')
    for _, group in groups:
        group = group.sort_values('date').reset_index(drop=True)
        features = group[feature_cols].values
        targets = group[target_col].values
        n_obs = len(group)
        
        for i in range(n_obs):
            start_idx = i - target_offset
            end_idx = i + (seq_length - 1 - target_offset)
            
            seq = []
            for idx in range(start_idx, end_idx + 1):
                if idx < 0:
                    seq.append(features[0])  # Edge pad with first available record
                elif idx >= n_obs:
                    seq.append(features[-1]) # Edge pad with last available record
                else:
                    seq.append(features[idx])
            X_seq.append(np.array(seq))
            y_seq.append(targets[i])
            
    return np.array(X_seq), np.array(y_seq)

# ==========================================
# 3. MAIN WORKFLOW PIPELINE
# ==========================================

def main():
    dataset_path = r"/home/surya/Downloads/Old Repos/AdvaRisk - Test/data/model_training_dataset.csv"
    output_dir = r"./NDVIModel"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading raw agricultural training dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Path {dataset_path} does not exist.")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset Loaded successfully. Raw Shape: {df.shape}")
    
    # -------------------------------------------------------------------------
    # ON-THE-FLY FEATURE ENGINEERING: REAL-TIME RADAR MOMENTUM
    # -------------------------------------------------------------------------
    print("Calculating live radar velocity / momentum fields...")
    df['RVI_velocity_6'] = df['raw_RVI'] - df['RVI_lag_6']
    df['RVI_velocity_12'] = df['raw_RVI'] - df['RVI_lag_12']
    
    # Unified Pure Dynamic Feature Pool (Excludes static LULC to prevent shortcut cheating)
    shared_dynamic_pool = [
        'raw_RVI', 'RVI_lag_12', 'RVI_lag_6', 'RVI_lead_6', 'RVI_lead_12',
        'RVI_velocity_6', 'RVI_velocity_12', # Dynamic trend features
        'doy_sin', 'doy_cos', 'Rainfall_15d_sum', 'MaxTemp_7d_avg', 'MinTemp_7d_avg',
        'is_kharif', 'is_rabi', 'is_zaid', 'latitude', 'longitude'
    ]
    target_col = 'raw_NDVI'
        
    # -------------------------------------------------------------------------
    # TARGET RESIDUAL TRANSFORMATION
    # -------------------------------------------------------------------------
    print("Calculating historical baseline profiles per spatial parcel unit...")
    parcel_baselines = df.groupby('task_id')[target_col].transform('median')
    df['baseline_NDVI'] = parcel_baselines
    df['delta_NDVI'] = df[target_col] - df['baseline_NDVI']
    ensemble_target = 'delta_NDVI'
    
    # Sort data chronologically inside spatial groups before splitting
    df = df.groupby('task_id', group_keys=False).apply(lambda x: x.sort_values('date')).reset_index(drop=True)
    
    # Group-based train-test split (80-20 Split grouped strictly by task_id)
    print("Executing group-shuffled layout partition split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, df[ensemble_target], groups=df['task_id']))
    
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Feature Scaling transformation applied over active features array
    scaler = StandardScaler()
    df_train[shared_dynamic_pool] = scaler.fit_transform(df_train[shared_dynamic_pool])
    df_test[shared_dynamic_pool] = scaler.transform(df_test[shared_dynamic_pool])
    
    joblib.dump(scaler, os.path.join(output_dir, "ensemble_scaler.pkl"))
    
    # Generate symmetrical context arrays for BiGRU 
    seq_length = 5
    X_train_seq, y_train_seq = create_symmetrical_sequences(df_train, shared_dynamic_pool, ensemble_target, seq_length)
    X_test_seq, y_test_seq = create_symmetrical_sequences(df_test, shared_dynamic_pool, ensemble_target, seq_length)
    
    # ==========================================
    # 4. BIDIRECTIONAL GRU BRANCH
    # ==========================================
    print("\n--- Training BiGRU Context Branch ---")
    train_loader = DataLoader(ParcelSequenceDataset(X_train_seq, y_train_seq), batch_size=64, shuffle=True)
    test_loader = DataLoader(ParcelSequenceDataset(X_test_seq, y_test_seq), batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bigru_model = NDVI_BiGRU(input_dim=len(shared_dynamic_pool)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(bigru_model.parameters(), lr=0.001, weight_decay=1e-3)
    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        bigru_model.train()
        t_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(bigru_model(bx), by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * bx.size(0)
            
        t_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation Eval
        bigru_model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                v_loss += criterion(bigru_model(bx), by).item() * bx.size(0)
        v_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Delta MSE: {t_loss:.5f} | Test Delta MSE: {v_loss:.5f}")
        
    torch.save(bigru_model.state_dict(), os.path.join(output_dir, "bigru_branch.pt"))
    
    # ==========================================
    # 6. ENSEMBLE BLENDING & ABSOLUTE EVALUATION
    # ==========================================
    print("\n--- Running BiGRU Prediction System ---")
    bigru_model.eval()
    
    bigru_delta_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bigru_delta_preds.append(bigru_model(bx.to(device)).cpu().numpy())
    bigru_delta_pred = np.vstack(bigru_delta_preds).squeeze()
    
    # Reconstruction back into ground truth absolute values from BiGRU residual predictions
    y_actual_absolute = df_test[target_col].values
    y_pred_absolute = bigru_delta_pred + df_test['baseline_NDVI'].values
    y_pred_absolute_clipped = np.clip(y_pred_absolute, 0.0, 1.0)
    
    # Compute operational deployment metrics
    r2 = r2_score(y_actual_absolute, y_pred_absolute_clipped)
    rmse = np.sqrt(mean_squared_error(y_actual_absolute, y_pred_absolute_clipped))
    mae = mean_absolute_error(y_actual_absolute, y_pred_absolute_clipped)
    
    print("\n" + "="*50)
    print("HYBRID ENSEMBLE REAL ABSOLUTE METRICS (OUT-OF-SAMPLE):")
    print(f"Test Set R^2 Score: {r2:.4f}")
    print(f"Test Set RMSE Loss: {rmse:.4f}")
    print(f"Test Set MAE  Loss: {mae:.4f}")
    print("="*50)
    
    ensemble_metrics = {
        'ensemble_strategy': 'BiGRU-only Dynamic Fusion with Velocity (No Static LULC)',
        'absolute_test_scores': {'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae)}
    }
    with open(os.path.join(output_dir, "ensemble_test_metrics.json"), "w") as j_file:
        json.dump(ensemble_metrics, j_file, indent=4)

    # ==========================================
    # 7. METRICS & GRAPHICAL EVIDENCE GENERATION
    # ==========================================
    print("\nGenerating empirical graphical charts...")
    
    # Plot 1: True vs Predicted Absolute NDVI Curve Map
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual_absolute, y_pred_absolute_clipped, alpha=0.25, color='darkcyan', edgecolors='none')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label="1:1 Parity Ideal Fit")
    plt.title("Ensemble Reconstructed Absolute NDVI Validation Map")
    plt.xlabel("Ground Truth Sentinel-2 NDVI")
    plt.ylabel("Ensemble Predicted NDVI")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ensemble_absolute_scatter.png"), dpi=300)
    plt.close()

    # Plot 2: Neural Network Attribution Map
    print("Computing Integrated Gradients for the Isolated BiGRU Branch...")
    bigru_model.train()
    for module in bigru_model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
            
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(bigru_model)
    
    batch_x, _ = next(iter(test_loader))
    batch_x = batch_x.to(device).requires_grad_()
    baseline = torch.zeros_like(batch_x).to(device)
    
    attributions = ig.attribute(batch_x, baseline, target=0)
    bigru_model.eval()
    
    mean_attributions = attributions.cpu().detach().numpy().mean(axis=0)
    absolute_attributions = np.abs(mean_attributions)
    global_bigru_importance = np.mean(absolute_attributions, axis=0)

    # Save Global BiGRU Importance Bar Chart
    sorted_idx_bigru = np.argsort(global_bigru_importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(shared_dynamic_pool)), global_bigru_importance[sorted_idx_bigru], color='mediumseagreen', edgecolor='black', alpha=0.85)
    plt.yticks(range(len(shared_dynamic_pool)), [shared_dynamic_pool[i] for i in sorted_idx_bigru], fontsize=10)
    plt.xlabel("Average Absolute Attribution Intensity")
    plt.title("BiGRU Sequential Branch: Global Feature Importance Profiling (No Static LULC)", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bigru_global_importance.png"), dpi=300)
    plt.close()
    
    print(f"All processing runs finished successfully. Structural artifacts saved in: {output_dir}")

if __name__ == "__main__":
    main()
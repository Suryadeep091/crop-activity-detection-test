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

# ==========================================
# 0. REPRODUCIBILITY
# ==========================================
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. FEATURE GROUP DEFINITIONS
#    Informed by Integrated Gradients chart from single-BiGRU baseline.
#    Each group is isolated into its own BiGRU tower.
# ==========================================

# Tower 1: RVI (Radar Vegetation Index) — highest single-domain attribution
TOWER_RVI = [
    'raw_RVI', 'RVI_lag_6', 'RVI_lag_12',
    'RVI_lead_6', 'RVI_lead_12',
    'RVI_velocity_6', 'RVI_velocity_12',
]

# Tower 2: VV Backscatter — velocity_6 showed notable attribution
TOWER_VV = [
    'raw_VV', 'VV_lag_6', 'VV_lag_12',
    'VV_lead_6', 'VV_lead_12',
    'VV_velocity_6', 'VV_velocity_12',
]

# Tower 3: VH Backscatter — complementary polarization channel
TOWER_VH = [
    'raw_VH', 'VH_lag_6', 'VH_lag_12',
    'VH_lead_6', 'VH_lead_12',
    'VH_velocity_6', 'VH_velocity_12',
]

# Tower 4: Weather — MinTemp and MaxTemp were 2nd and 4th highest globally
TOWER_WX = [
    'Rainfall_15d_sum', 'MaxTemp_7d_avg', 'MinTemp_7d_avg',
]

# Tower 5: Phenological context
TOWER_CTX = [
    'doy_sin', 'doy_cos',
    'is_kharif', 'is_rabi', 'is_zaid',
    'expected_baseline_ndvi',
]

ALL_FEATURE_COLS = TOWER_RVI + TOWER_VV + TOWER_VH + TOWER_WX + TOWER_CTX
TARGET_COL = 'raw_NDVI'

def get_tower_indices():
    rvi_idx = list(range(0, len(TOWER_RVI)))
    vv_idx  = list(range(len(TOWER_RVI), len(TOWER_RVI) + len(TOWER_VV)))
    vh_idx  = list(range(len(TOWER_RVI) + len(TOWER_VV),
                         len(TOWER_RVI) + len(TOWER_VV) + len(TOWER_VH)))
    wx_idx  = list(range(len(TOWER_RVI) + len(TOWER_VV) + len(TOWER_VH),
                         len(TOWER_RVI) + len(TOWER_VV) + len(TOWER_VH) + len(TOWER_WX)))
    ctx_idx = list(range(len(TOWER_RVI) + len(TOWER_VV) + len(TOWER_VH) + len(TOWER_WX),
                         len(ALL_FEATURE_COLS)))
    return rvi_idx, vv_idx, vh_idx, wx_idx, ctx_idx

# ==========================================
# 2. DATASET
# ==========================================

class MultiTowerDataset(Dataset):
    def __init__(self, X, y):
        rvi_idx, vv_idx, vh_idx, wx_idx, ctx_idx = get_tower_indices()
        self.x_rvi = torch.tensor(X[:, :, rvi_idx], dtype=torch.float32)
        self.x_vv  = torch.tensor(X[:, :, vv_idx],  dtype=torch.float32)
        self.x_vh  = torch.tensor(X[:, :, vh_idx],  dtype=torch.float32)
        self.x_wx  = torch.tensor(X[:, :, wx_idx],  dtype=torch.float32)
        self.x_ctx = torch.tensor(X[:, :, ctx_idx], dtype=torch.float32)
        self.y     = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.x_rvi[idx], self.x_vv[idx], self.x_vh[idx],
            self.x_wx[idx], self.x_ctx[idx], self.y[idx]
        )

# ==========================================
# 3. MULTI-TOWER BiGRU MODEL
# ==========================================

class MultiTowerBiGRU(nn.Module):
    """
    5 parallel BiGRU towers, each processing one feature domain.
    Center-step hidden states are concatenated and fused via Dense layers.

      Tower dims (bidir output):
        RVI  : hidden=48  -> 96
        VV   : hidden=32  -> 64
        VH   : hidden=32  -> 64
        Wx   : hidden=24  -> 48
        Ctx  : hidden=24  -> 48
        Fusion: 320 -> 128 -> 32 -> 1
    """
    def __init__(
        self,
        rvi_dim=7, vv_dim=7, vh_dim=7, wx_dim=3, ctx_dim=6,
        rvi_hidden=48, vv_hidden=32, vh_hidden=32, wx_hidden=24, ctx_hidden=24,
        fusion_dropout1=0.3, fusion_dropout2=0.2,
    ):
        super().__init__()
        self.tower_rvi = nn.GRU(rvi_dim, rvi_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.tower_vv  = nn.GRU(vv_dim,  vv_hidden,  num_layers=1, batch_first=True, bidirectional=True)
        self.tower_vh  = nn.GRU(vh_dim,  vh_hidden,  num_layers=1, batch_first=True, bidirectional=True)
        self.tower_wx  = nn.GRU(wx_dim,  wx_hidden,  num_layers=1, batch_first=True, bidirectional=True)
        self.tower_ctx = nn.GRU(ctx_dim, ctx_hidden, num_layers=1, batch_first=True, bidirectional=True)

        fusion_in = (rvi_hidden + vv_hidden + vh_hidden + wx_hidden + ctx_hidden) * 2  # 320

        self.norm   = nn.LayerNorm(fusion_in)
        self.fc1    = nn.Linear(fusion_in, 128)
        self.fc2    = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.relu   = nn.ReLU()
        self.drop1  = nn.Dropout(fusion_dropout1)
        self.drop2  = nn.Dropout(fusion_dropout2)

    def _center(self, gru_out, seq_len):
        return gru_out[:, seq_len // 2, :]

    def forward(self, x_rvi, x_vv, x_vh, x_wx, x_ctx):
        seq = x_rvi.size(1)
        h1 = self._center(self.tower_rvi(x_rvi)[0], seq)
        h2 = self._center(self.tower_vv(x_vv)[0],   seq)
        h3 = self._center(self.tower_vh(x_vh)[0],   seq)
        h4 = self._center(self.tower_wx(x_wx)[0],   seq)
        h5 = self._center(self.tower_ctx(x_ctx)[0], seq)
        h = torch.cat([h1, h2, h3, h4, h5], dim=-1)
        h = self.norm(h)
        h = self.drop1(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        return self.fc_out(h)

# ==========================================
# 4. SEQUENCE GENERATION
# ==========================================

def create_symmetrical_sequences(df, feature_cols, target_col, seq_length=5):
    X_seq, y_seq = [], []
    target_offset = seq_length // 2
    for _, group in df.groupby('task_id'):
        group = group.sort_values('date').reset_index(drop=True)
        features = group[feature_cols].values
        targets  = group[target_col].values
        n_obs    = len(group)
        for i in range(n_obs):
            seq = []
            for idx in range(i - target_offset, i + (seq_length - 1 - target_offset) + 1):
                if idx < 0:
                    seq.append(features[0])
                elif idx >= n_obs:
                    seq.append(features[-1])
                else:
                    seq.append(features[idx])
            X_seq.append(np.array(seq))
            y_seq.append(targets[i])
    return np.array(X_seq), np.array(y_seq)

# ==========================================
# 5. MAIN TRAINING PIPELINE
# ==========================================

def main():
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "../data/model_training_dataset.csv")
    output_dir   = os.path.join(script_dir, "../NDVIModel")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Raw shape: {df.shape}")

    # Feature engineering
    df['RVI_velocity_6']  = df['raw_RVI'] - df['RVI_lag_6']
    df['RVI_velocity_12'] = df['raw_RVI'] - df['RVI_lag_12']
    df['VV_velocity_6']   = df['raw_VV']  - df['VV_lag_6']
    df['VV_velocity_12']  = df['raw_VV']  - df['VV_lag_12']
    df['VH_velocity_6']   = df['raw_VH']  - df['VH_lag_6']
    df['VH_velocity_12']  = df['raw_VH']  - df['VH_lag_12']
    tier_0 = df[['water', 'bare', 'snow_and_ice']].sum(axis=1)
    tier_1 = df[['shrub_and_scrub', 'built', 'flooded_vegetation', 'crops']].sum(axis=1)
    tier_2 = df[['grass', 'trees']].sum(axis=1)
    df['expected_baseline_ndvi'] = 0.0531 * tier_0 + 0.3817 * tier_1 + 0.6207 * tier_2

    df = df.dropna(subset=[TARGET_COL])
    df = (df.groupby('task_id', group_keys=False)
            .apply(lambda x: x.sort_values('date'))
            .reset_index(drop=True))

    # Train/Test split
    print("Group-shuffled 80/20 split by parcel...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, df[TARGET_COL], groups=df['task_id']))
    df_train = df.iloc[train_idx].copy()
    df_test  = df.iloc[test_idx].copy()
    print(f"Train parcels: {df_train['task_id'].nunique()} | Test parcels: {df_test['task_id'].nunique()}")

    # Scaling
    scaler = StandardScaler()
    df_train[ALL_FEATURE_COLS] = scaler.fit_transform(df_train[ALL_FEATURE_COLS])
    df_test[ALL_FEATURE_COLS]  = scaler.transform(df_test[ALL_FEATURE_COLS])
    joblib.dump(scaler, os.path.join(output_dir, "mt_bigru_scaler.pkl"))
    print("Scaler saved.")

    # Sequences
    SEQ_LEN = 9  # ±4 observations ≈ 24-48 calendar days of SAR context
    print(f"Building symmetric windows (seq_len={SEQ_LEN})...")
    X_train, y_train = create_symmetrical_sequences(df_train, ALL_FEATURE_COLS, TARGET_COL, SEQ_LEN)
    X_test,  y_test  = create_symmetrical_sequences(df_test,  ALL_FEATURE_COLS, TARGET_COL, SEQ_LEN)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # DataLoaders
    train_loader = DataLoader(MultiTowerDataset(X_train, y_train), batch_size=64,  shuffle=True,  pin_memory=True)
    test_loader  = DataLoader(MultiTowerDataset(X_test,  y_test),  batch_size=256, shuffle=False, pin_memory=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    model = MultiTowerBiGRU(
        rvi_dim=len(TOWER_RVI), vv_dim=len(TOWER_VV), vh_dim=len(TOWER_VH),
        wx_dim=len(TOWER_WX),   ctx_dim=len(TOWER_CTX),
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    epochs    = 50
    patience  = 8   # stop if val MSE doesn't improve for 8 consecutive epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss   = float('inf')
    best_state      = None
    patience_count  = 0
    train_history, val_history = [], []

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for x_rvi, x_vv, x_vh, x_wx, x_ctx, by in train_loader:
            x_rvi, x_vv, x_vh = x_rvi.to(device), x_vv.to(device), x_vh.to(device)
            x_wx,  x_ctx, by  = x_wx.to(device),  x_ctx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_rvi, x_vv, x_vh, x_wx, x_ctx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item() * by.size(0)
        t_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x_rvi, x_vv, x_vh, x_wx, x_ctx, by in test_loader:
                x_rvi, x_vv, x_vh = x_rvi.to(device), x_vv.to(device), x_vh.to(device)
                x_wx,  x_ctx, by  = x_wx.to(device),  x_ctx.to(device), by.to(device)
                v_loss += criterion(model(x_rvi, x_vv, x_vh, x_wx, x_ctx), by).item() * by.size(0)
        v_loss /= len(test_loader.dataset)

        train_history.append(t_loss)
        val_history.append(v_loss)
        improved = v_loss < best_val_loss
        if improved:
            best_val_loss  = v_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {t_loss:.5f} | Val MSE: {v_loss:.5f}"
              + (" ← best" if improved else f" (patience {patience_count}/{patience})"))

        if patience_count >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x_rvi, x_vv, x_vh, x_wx, x_ctx, by in test_loader:
            x_rvi, x_vv, x_vh = x_rvi.to(device), x_vv.to(device), x_vh.to(device)
            x_wx,  x_ctx       = x_wx.to(device),  x_ctx.to(device)
            all_preds.append(model(x_rvi, x_vv, x_vh, x_wx, x_ctx).cpu().numpy())
            all_true.append(by.numpy())

    y_pred = np.clip(np.vstack(all_preds).squeeze(), 0.0, 1.0)
    y_true = np.vstack(all_true).squeeze()

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    print("\n" + "="*55)
    print("MULTI-TOWER BiGRU — OUT-OF-SAMPLE METRICS")
    print(f"  R²   : {r2:.4f}   (baseline single-BiGRU: 0.8381)")
    print(f"  RMSE : {rmse:.4f}  (baseline: 0.1105)")
    print(f"  MAE  : {mae:.4f}  (baseline: 0.0703)")
    print("="*55)

    metrics = {
        "architecture": "MultiTowerBiGRU-5tower",
        "towers": {"rvi": TOWER_RVI, "vv": TOWER_VV, "vh": TOWER_VH, "weather": TOWER_WX, "context": TOWER_CTX},
        "seq_length": SEQ_LEN,
        "epochs_trained": epochs,
        "best_val_mse": float(best_val_loss),
        "test_metrics": {"r2": float(r2), "rmse": float(rmse), "mae": float(mae)},
        "baseline_single_bigru": {"r2": 0.8381, "rmse": 0.1105, "mae": 0.0703}
    }
    with open(os.path.join(output_dir, "mt_bigru_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    torch.save(model.state_dict(), os.path.join(output_dir, "multi_tower_bigru.pt"))
    print(f"Model  : {output_dir}/multi_tower_bigru.pt")
    print(f"Scaler : {output_dir}/mt_bigru_scaler.pkl")
    print(f"Metrics: {output_dir}/mt_bigru_metrics.json")

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs+1), train_history, label='Train MSE', color='steelblue')
    plt.plot(range(1, epochs+1), val_history,   label='Val MSE',   color='tomato')
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Multi-Tower BiGRU — Training Curve")
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mt_bigru_training_curve.png"), dpi=200); plt.close()

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.2, color='mediumseagreen', edgecolors='none', s=8)
    plt.plot([0, 1], [0, 1], 'r--', lw=1.5, label="1:1 Parity")
    plt.title(f"Multi-Tower BiGRU — NDVI Scatter (R²={r2:.3f})")
    plt.xlabel("True NDVI"); plt.ylabel("Predicted NDVI")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(); plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mt_bigru_scatter.png"), dpi=200); plt.close()

    print("\nAll artifacts saved. Training complete.")
    return metrics

if __name__ == "__main__":
    main()

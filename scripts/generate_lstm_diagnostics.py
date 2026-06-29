import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

class NDVI_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(NDVI_BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        out = out[:, x.size(1) // 2, :]
        out = self.hidden_fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def create_sequences(df, feature_cols, target_col, seq_length=5, df_raw=None):
    X_seq = []
    y_seq = []
    raw_features_sorted = []
    
    target_offset = seq_length // 2
    
    if 'task_id' in df.columns:
        groups = df.groupby('task_id')
    else:
        groups = [('single', df)]
        
    for task_id, group in groups:
        if df_raw is not None:
            group_raw = df_raw[df_raw['task_id'] == task_id].sort_values('date')
            features_raw = group_raw[feature_cols].values
        group = group.sort_values('date')
        features = group[feature_cols].values
        targets = group[target_col].values
        
        n_obs = len(group)
        for i in range(n_obs):
            start_idx = i - target_offset
            end_idx = i + (seq_length - 1 - target_offset)
            
            seq = []
            for idx in range(start_idx, end_idx + 1):
                if idx < 0 or idx >= n_obs:
                    seq.append(np.zeros(features.shape[1]))
                else:
                    seq.append(features[idx])
                    
            X_seq.append(np.array(seq))
            y_seq.append(targets[i])
            if df_raw is not None:
                raw_features_sorted.append(features_raw[i])
                
    if df_raw is not None:
        return np.array(X_seq), np.array(y_seq), np.array(raw_features_sorted)
    return np.array(X_seq), np.array(y_seq)

def main():
    dataset_path = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\data\model_training_dataset.csv"
    models_dir = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\models"
    analysis_dir = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\analysis"
    brain_assets_dir = r"C:\Users\Suryadeep Singh\.gemini\antigravity-ide\brain\ea250939-4c8f-45b7-a97e-d60cb3693660\assets"
    
    df = pd.read_csv(dataset_path)
    
    feature_cols = [
        'latitude', 'longitude', 'raw_RVI',
        'is_kharif', 'is_rabi', 'is_zaid', 'doy_sin', 'doy_cos',
        'Rainfall_15d_sum', 'MaxTemp_7d_avg', 'MinTemp_7d_avg',
        'water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice',
        'RVI_lag_12', 'RVI_lag_6', 'RVI_lead_6', 'RVI_lead_12'
    ]
    target_col = 'raw_NDVI'
    
    # 1. Split and scale features
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, df[target_col], groups=df['task_id']))
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    lstm_scaler = joblib.load(os.path.join(models_dir, "lstm_scaler.pkl"))
    df_test_scaled = df_test.copy()
    df_test_scaled[feature_cols] = lstm_scaler.transform(df_test[feature_cols])
    
    # 2. Load BiGRU Model
    lstm_state = torch.load(os.path.join(models_dir, "rvi_to_ndvi_lstm.pt"), map_location='cpu')
    model = NDVI_BiGRU(input_dim=len(feature_cols), hidden_dim=64, num_layers=2)
    model.load_state_dict(lstm_state)
    model.eval()
    
    # 3. Create baseline sequences and get baseline metrics
    seq_length = 5
    X_test, y_test, raw_features_sorted = create_sequences(df_test_scaled, feature_cols, target_col, seq_length, df_raw=df_test)
    
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().squeeze()
    
    baseline_r2 = r2_score(y_test, preds)
    baseline_mse = mean_squared_error(y_test, preds)
    print(f"BiGRU Test R2: {baseline_r2:.4f} | MSE: {baseline_mse:.5f}")
    
    # 3b. Load RF model and calculate Ensemble predictions
    rf_model = joblib.load(os.path.join(models_dir, "rvi_to_ndvi_model.pkl"))
    preds_rf = rf_model.predict(raw_features_sorted)
    preds_ensemble = 0.4 * preds_rf + 0.6 * preds
    ensemble_r2 = r2_score(y_test, preds_ensemble)
    print(f"Ensemble Test R2: {ensemble_r2:.4f}")
    
    # Plot 1: BiGRU Scatter Plot Actual vs. Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.3, color='royalblue', edgecolors='none')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label="1:1 Perfect Prediction Line")
    plt.title(f"BiGRU Model: Actual vs. Predicted NDVI (Test R² = {baseline_r2:.4f})")
    plt.xlabel("Actual NDVI")
    plt.ylabel("Predicted NDVI")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(analysis_dir, "bigru_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"BiGRU scatter plot saved to: {scatter_path}")
    
    # 4. Calculate Permutation Feature Importance
    # For each feature, shuffle its column before sequence creation, and evaluate loss increase
    print("Calculating permutation feature importances (this may take a few seconds)...")
    feature_importances = {}
    
    for idx, col in enumerate(feature_cols):
        df_shuffled = df_test_scaled.copy()
        # Shuffle values within the group/column to break its correlation with target
        df_shuffled[col] = np.random.permutation(df_shuffled[col].values)
        
        X_shuff, y_shuff = create_sequences(df_shuffled, feature_cols, target_col, seq_length)
        
        with torch.no_grad():
            shuff_preds = model(torch.tensor(X_shuff, dtype=torch.float32)).numpy().squeeze()
            
        shuff_mse = mean_squared_error(y_shuff, shuff_preds)
        # Importance = increase in MSE error
        importance = shuff_mse - baseline_mse
        feature_importances[col] = max(0.0, importance)
        
    # Sort and plot feature importances
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features)
    
    # Normalize importances so relative sum is 1 or max is 1. Let's make relative to max.
    max_imp = max(importances) if max(importances) > 0 else 1.0
    relative_importances = [imp / max_imp for imp in importances]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), relative_importances[::-1], align='center', color='royalblue')
    plt.yticks(range(len(features)), features[::-1])
    plt.xlabel("Relative Importance (Increase in Test MSE)")
    plt.title("BiGRU Model: Permutation Feature Importances")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    importance_path = os.path.join(analysis_dir, "bigru_feature_importance.png")
    plt.savefig(importance_path, dpi=300)
    plt.close()
    print(f"BiGRU feature importance plot saved to: {importance_path}")
    
    # 5. Plot Ensemble Scatter Plot Actual vs. Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds_ensemble, alpha=0.3, color='crimson', edgecolors='none')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label="1:1 Perfect Prediction Line")
    plt.title(f"RF + BiGRU Ensemble: Actual vs. Predicted NDVI (Test R² = {ensemble_r2:.4f})")
    plt.xlabel("Actual NDVI")
    plt.ylabel("Predicted NDVI")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ensemble_scatter_path = os.path.join(analysis_dir, "ensemble_scatter.png")
    plt.savefig(ensemble_scatter_path, dpi=300)
    plt.close()
    print(f"Ensemble scatter plot saved to: {ensemble_scatter_path}")
    
    # 6. Copy plots to brain assets and brain root
    os.makedirs(brain_assets_dir, exist_ok=True)
    import shutil
    shutil.copyfile(scatter_path, os.path.join(brain_assets_dir, "bigru_scatter.png"))
    shutil.copyfile(importance_path, os.path.join(brain_assets_dir, "bigru_feature_importance.png"))
    shutil.copyfile(ensemble_scatter_path, os.path.join(brain_assets_dir, "ensemble_scatter.png"))
    
    # Also copy ensemble scatter to brain root
    brain_dir = os.path.dirname(brain_assets_dir)
    shutil.copyfile(ensemble_scatter_path, os.path.join(brain_dir, "ensemble_scatter.png"))
    print("Plots copied to brain assets and root successfully.")

if __name__ == "__main__":
    main()

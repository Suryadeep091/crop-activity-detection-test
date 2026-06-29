import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

class NDVI_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(NDVI_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.hidden_fc = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.hidden_fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def create_sequences(df, feature_cols, target_col, seq_length=5):
    X_seq = []
    y_seq = []
    
    for task_id, group in df.groupby('task_id'):
        group = group.sort_values('date')
        features = group[feature_cols].values
        targets = group[target_col].values
        
        n_obs = len(group)
        for i in range(n_obs):
            start_idx = max(0, i - seq_length + 1)
            seq = features[start_idx:i + 1]
            
            if len(seq) < seq_length:
                pad_len = seq_length - len(seq)
                pad = np.zeros((pad_len, features.shape[1]))
                seq = np.vstack((pad, seq))
                
            X_seq.append(seq)
            y_seq.append(targets[i])
            
    return np.array(X_seq), np.array(y_seq)

def main():
    dataset_path = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\data\model_training_dataset.csv"
    models_dir = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\models"
    analysis_dir = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\analysis"
    brain_dir = r"C:\Users\Suryadeep Singh\.gemini\antigravity-ide\brain\ea250939-4c8f-45b7-a97e-d60cb3693660"
    brain_assets_dir = os.path.join(brain_dir, "assets")
    
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
    df_test = df.iloc[test_idx].copy()
    
    lstm_scaler = joblib.load(os.path.join(models_dir, "lstm_scaler.pkl"))
    df_test_scaled = df_test.copy()
    df_test_scaled[feature_cols] = lstm_scaler.transform(df_test[feature_cols])
    
    # 2. Load Standard LSTM Model
    model_state = torch.load(os.path.join(models_dir, "rvi_to_ndvi_lstm_standard.pt"), map_location='cpu')
    model = NDVI_LSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2)
    model.load_state_dict(model_state)
    model.eval()
    
    # 3. Create sequences
    seq_length = 5
    X_test, y_test = create_sequences(df_test_scaled, feature_cols, target_col, seq_length)
    
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().squeeze()
    
    baseline_r2 = r2_score(y_test, preds)
    baseline_mse = mean_squared_error(y_test, preds)
    print(f"Standard LSTM Baseline Test R2: {baseline_r2:.4f} | MSE: {baseline_mse:.5f}")
    
    # Plot 1: Standard LSTM Scatter Plot Actual vs. Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.3, color='goldenrod', edgecolors='none')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label="1:1 Perfect Prediction Line")
    plt.title(f"Standard LSTM Model: Actual vs. Predicted NDVI (Test R² = {baseline_r2:.4f})")
    plt.xlabel("Actual NDVI")
    plt.ylabel("Predicted NDVI")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(analysis_dir, "lstm_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"LSTM scatter plot saved to: {scatter_path}")
    
    # 4. Calculate Permutation Feature Importance
    print("Calculating permutation feature importances...")
    feature_importances = {}
    
    for idx, col in enumerate(feature_cols):
        df_shuffled = df_test_scaled.copy()
        df_shuffled[col] = np.random.permutation(df_shuffled[col].values)
        X_shuff, y_shuff = create_sequences(df_shuffled, feature_cols, target_col, seq_length)
        
        with torch.no_grad():
            shuff_preds = model(torch.tensor(X_shuff, dtype=torch.float32)).numpy().squeeze()
            
        shuff_mse = mean_squared_error(y_shuff, shuff_preds)
        importance = shuff_mse - baseline_mse
        feature_importances[col] = max(0.0, importance)
        
    # Sort and plot feature importances
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features)
    
    max_imp = max(importances) if max(importances) > 0 else 1.0
    relative_importances = [imp / max_imp for imp in importances]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), relative_importances[::-1], align='center', color='goldenrod')
    plt.yticks(range(len(features)), features[::-1])
    plt.xlabel("Relative Importance (Increase in Test MSE)")
    plt.title("Standard LSTM Model: Permutation Feature Importances")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    importance_path = os.path.join(analysis_dir, "lstm_feature_importance.png")
    plt.savefig(importance_path, dpi=300)
    plt.close()
    print(f"LSTM feature importance plot saved to: {importance_path}")
    
    # 5. Copy plots to brain assets & root
    os.makedirs(brain_assets_dir, exist_ok=True)
    import shutil
    shutil.copyfile(scatter_path, os.path.join(brain_assets_dir, "lstm_scatter.png"))
    shutil.copyfile(importance_path, os.path.join(brain_assets_dir, "lstm_feature_importance.png"))
    shutil.copyfile(scatter_path, os.path.join(brain_dir, "lstm_scatter.png"))
    shutil.copyfile(importance_path, os.path.join(brain_dir, "lstm_feature_importance.png"))
    print("Standard LSTM plots copied to brain successfully.")

if __name__ == "__main__":
    main()

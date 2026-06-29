import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Define BiGRU architecture
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

# Define Standard LSTM architecture
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
    
    group = df.sort_values('date')
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

def create_centered_sequences(df, feature_cols, target_col, seq_length=5):
    X_seq = []
    y_seq = []
    
    target_offset = seq_length // 2
    
    if 'task_id' in df.columns:
        groups = df.groupby('task_id')
    else:
        groups = [('single', df)]
        
    for _, group in groups:
        group = group.sort_values('date')
        features = group[feature_cols].values
        targets = group[target_col].values
        
        n_obs = len(group)
        for i in range(n_obs):
            start_idx = i - target_offset
            end_idx = i + (seq_length - 1 - target_offset)
            
            seq = []
            for idx in range(start_idx, end_idx + 1):
                if idx < 0:
                    seq.append(np.zeros(features.shape[1]))
                elif idx >= n_obs:
                    seq.append(np.zeros(features.shape[1]))
                else:
                    seq.append(features[idx])
                    
            X_seq.append(np.array(seq))
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
    
    # 1. Load Models and Scalers
    print("Loading models and scalers...")
    rf_model = joblib.load(os.path.join(models_dir, "rvi_to_ndvi_model.pkl"))
    lstm_scaler = joblib.load(os.path.join(models_dir, "lstm_scaler.pkl"))
    
    # Load BiGRU
    bigru_state = torch.load(os.path.join(models_dir, "rvi_to_ndvi_lstm.pt"), map_location='cpu')
    bigru_model = NDVI_BiGRU(input_dim=len(feature_cols), hidden_dim=64, num_layers=2)
    bigru_model.load_state_dict(bigru_state)
    bigru_model.eval()
    
    # Load Standard LSTM
    lstm_state = torch.load(os.path.join(models_dir, "rvi_to_ndvi_lstm_standard.pt"), map_location='cpu')
    lstm_model = NDVI_LSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2)
    lstm_model.load_state_dict(lstm_state)
    lstm_model.eval()
    
    # 2. Select representative test parcel
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, df[target_col], groups=df['task_id']))
    df_test = df.iloc[test_idx].copy()
    
    test_parcels = df_test['task_id'].unique()
    sample_parcel_id = next((p for p in test_parcels if len(df_test[df_test['task_id'] == p]) >= 15), test_parcels[0])
    print(f"Generating comparison plot for parcel: {sample_parcel_id}")
    
    df_sample = df[df['task_id'] == sample_parcel_id].copy()
    df_sample['date'] = pd.to_datetime(df_sample['date'])
    df_sample = df_sample.sort_values('date')
    
    # RF Predictions
    y_rf_pred = np.clip(rf_model.predict(df_sample[feature_cols]), 0.0, 1.0)
    
    # Scale features for LSTM / BiGRU
    df_sample_scaled = df_sample.copy()
    df_sample_scaled[feature_cols] = lstm_scaler.transform(df_sample[feature_cols])
    X_seq_centered, _ = create_centered_sequences(df_sample_scaled, feature_cols, target_col, seq_length=5)
    X_seq_standard, _ = create_sequences(df_sample_scaled, feature_cols, target_col, seq_length=5)
    
    # BiGRU Predictions
    with torch.no_grad():
        bigru_preds = bigru_model(torch.tensor(X_seq_centered, dtype=torch.float32))
        y_bigru_pred = np.clip(bigru_preds.numpy().squeeze(), 0.0, 1.0)
        
    # Standard LSTM Predictions
    with torch.no_grad():
        lstm_preds = lstm_model(torch.tensor(X_seq_standard, dtype=torch.float32))
        y_lstm_pred = np.clip(lstm_preds.numpy().squeeze(), 0.0, 1.0)
        
    # RF + BiGRU Ensemble
    y_ensemble_pred = 0.4 * y_rf_pred + 0.6 * y_bigru_pred
    
    # 3. Plot Temporal Curves
    plt.figure(figsize=(12, 5.5))
    plt.plot(df_sample['date'], df_sample['raw_NDVI'], 'o-', label='Ground Truth S2 NDVI', color='forestgreen', markersize=6)
    plt.plot(df_sample['date'], y_rf_pred, 's--', label='Predicted NDVI (Random Forest)', color='darkorange', markersize=6)
    plt.plot(df_sample['date'], y_lstm_pred, 'x-.', label='Predicted NDVI (Standard LSTM)', color='goldenrod', markersize=6)
    plt.plot(df_sample['date'], y_bigru_pred, 'd-.', label='Predicted NDVI (Bidirectional GRU)', color='royalblue', markersize=6)
    plt.plot(df_sample['date'], y_ensemble_pred, '*:', label='Predicted NDVI (RF + BiGRU Ensemble)', color='crimson', markersize=8, linewidth=2)
    
    plt.title(f"Temporal Model Comparison (RF vs LSTM vs BiGRU vs Ensemble) - Parcel: {sample_parcel_id}", fontsize=12, fontweight='bold')
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("NDVI", fontsize=10)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    plot_path = os.path.join(analysis_dir, "all_models_temporal.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")
    
    # 4. Copy to brain
    os.makedirs(brain_assets_dir, exist_ok=True)
    import shutil
    shutil.copyfile(plot_path, os.path.join(brain_assets_dir, "all_models_temporal.png"))
    shutil.copyfile(plot_path, os.path.join(brain_dir, "all_models_temporal.png"))
    print("Copied comparison plot to brain successfully.")

if __name__ == "__main__":
    main()

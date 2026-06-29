import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class ParcelDataset(Dataset):
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
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        # Extract features at target index (middle of sequence)
        out = out[:, x.size(1) // 2, :]
        out = self.hidden_fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def create_sequences(df, feature_cols, target_col, seq_length=5):
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
    dataset_path = r"/home/surya/Downloads/Old Repos/AdvaRisk - Test/data/model_training_dataset.csv"
    models_dir = r"/home/surya/Downloads/Old Repos/AdvaRisk - Test/models"
    analysis_dir = r"/home/surya/Downloads/Old Repos/AdvaRisk - Test/analysis"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} does not exist.")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    feature_cols = [
        'latitude', 'longitude', 'raw_RVI',
        'is_kharif', 'is_rabi', 'is_zaid', 'doy_sin', 'doy_cos',
        'Rainfall_15d_sum', 'MaxTemp_7d_avg', 'MinTemp_7d_avg',
        'water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice',
        'RVI_lag_12', 'RVI_lag_6', 'RVI_lead_6', 'RVI_lead_12'
    ]
    target_col = 'raw_NDVI'
    
    # Group-based train-test split (80-20, grouped by task_id)
    print("Performing group-based train-test split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, df[target_col], groups=df['task_id']))
    
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    
    # Save the scaler
    scaler_path = os.path.join(models_dir, "lstm_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Create time-series sequences
    seq_length = 5
    print(f"Generating sequences of length T={seq_length}...")
    X_train, y_train = create_sequences(df_train, feature_cols, target_col, seq_length)
    X_test, y_test = create_sequences(df_test, feature_cols, target_col, seq_length)
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    # Create datasets and loaders
    train_dataset = ParcelDataset(X_train, y_train)
    test_dataset = ParcelDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    model = NDVI_BiGRU(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # Training Loop
    epochs = 25
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print("Starting LSTM training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Test validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                test_loss += loss.item() * batch_x.size(0)
        test_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train MSE: {train_loss:.5f} | Test MSE: {test_loss:.5f}")
        
    # Save model
    model_path = os.path.join(models_dir, "rvi_to_ndvi_lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Evaluation
    model.eval()
    y_train_preds = []
    y_test_preds = []
    
    with torch.no_grad():
        # Evaluate train set in batches
        train_loader_eval = DataLoader(train_dataset, batch_size=128, shuffle=False)
        for batch_x, _ in train_loader_eval:
            pred = model(batch_x.to(device))
            y_train_preds.append(pred.cpu().numpy())
            
        # Evaluate test set
        for batch_x, _ in test_loader:
            pred = model(batch_x.to(device))
            y_test_preds.append(pred.cpu().numpy())
            
    y_train_preds = np.vstack(y_train_preds).squeeze()
    y_test_preds = np.vstack(y_test_preds).squeeze()
    
    # Clip predictions to biological NDVI limits
    y_train_preds_clipped = np.clip(y_train_preds, 0.0, 1.0)
    y_test_preds_clipped = np.clip(y_test_preds, 0.0, 1.0)
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_train_preds)
    r2_test = r2_score(y_test, y_test_preds)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_preds))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_preds))
    
    mae_train = mean_absolute_error(y_train, y_train_preds_clipped)
    mae_test = mean_absolute_error(y_test, y_test_preds_clipped)
    
    print("\n" + "="*40)
    print("LSTM MODEL EVALUATION RESULTS:")
    print(f"Train R^2: {r2_train:.4f} | Test R^2: {r2_test:.4f}")
    print(f"Train RMSE: {rmse_train:.4f} | Test RMSE: {rmse_test:.4f}")
    print(f"Train MAE: {mae_train:.4f} | Test MAE: {mae_test:.4f}")
    print("="*40)
    
    # Save metrics JSON
    lstm_metrics = {
        'model_type': 'BiGRU Deep Learning Regressor',
        'overall': {
            'train_r2': float(r2_train), 'test_r2': float(r2_test),
            'train_rmse': float(rmse_train), 'test_rmse': float(rmse_test),
            'train_mae': float(mae_train), 'test_mae': float(mae_test)
        }
    }
    
    metrics_path = os.path.join(analysis_dir, "rvi_to_ndvi_lstm_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(lstm_metrics, f, indent=4)
        
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()

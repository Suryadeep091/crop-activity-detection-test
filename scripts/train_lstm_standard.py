import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
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
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Take the output of the final time step
        out = out[:, -1, :]
        out = self.hidden_fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def create_sequences(df, feature_cols, target_col, seq_length=5):
    X_seq = []
    y_seq = []
    
    # Group by task_id and sort chronologically by date
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
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    feature_cols = [
        'latitude', 'longitude', 'raw_RVI',
        'is_kharif', 'is_rabi', 'is_zaid', 'doy_sin', 'doy_cos',
        'Rainfall_15d_sum', 'MaxTemp_7d_avg', 'MinTemp_7d_avg',
        'water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice',
        'RVI_lag_12', 'RVI_lag_6', 'RVI_lead_6', 'RVI_lead_12'
    ]
    target_col = 'raw_NDVI'
    
    # Group-based split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, df[target_col], groups=df['task_id']))
    
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Load the existing scaler (reusing it for consistency)
    scaler_path = os.path.join(models_dir, "lstm_scaler.pkl")
    scaler = joblib.load(scaler_path)
    df_train[feature_cols] = scaler.transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    
    # Create time-series sequences
    seq_length = 5
    X_train, y_train = create_sequences(df_train, feature_cols, target_col, seq_length)
    X_test, y_test = create_sequences(df_test, feature_cols, target_col, seq_length)
    
    train_dataset = ParcelDataset(X_train, y_train)
    test_dataset = ParcelDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = NDVI_LSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # Train model
    epochs = 20
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
    model_path = os.path.join(models_dir, "rvi_to_ndvi_lstm_standard.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Standard LSTM Model saved to: {model_path}")
    
    # Evaluation
    model.eval()
    y_train_preds = []
    y_test_preds = []
    with torch.no_grad():
        train_loader_eval = DataLoader(train_dataset, batch_size=128, shuffle=False)
        for batch_x, _ in train_loader_eval:
            pred = model(batch_x.to(device))
            y_train_preds.append(pred.cpu().numpy())
        for batch_x, _ in test_loader:
            pred = model(batch_x.to(device))
            y_test_preds.append(pred.cpu().numpy())
            
    y_train_preds = np.vstack(y_train_preds).squeeze()
    y_test_preds = np.vstack(y_test_preds).squeeze()
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2_train = r2_score(y_train, y_train_preds)
    r2_test = r2_score(y_test, y_test_preds)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_preds))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_preds))
    mae_train = mean_absolute_error(y_train, np.clip(y_train_preds, 0.0, 1.0))
    mae_test = mean_absolute_error(y_test, np.clip(y_test_preds, 0.0, 1.0))
    
    print("\nLSTM STANDARD MODEL EVALUATION RESULTS:")
    print(f"Train R^2: {r2_train:.4f} | Test R^2: {r2_test:.4f}")
    print(f"Train RMSE: {rmse_train:.4f} | Test RMSE: {rmse_test:.4f}")
    
    metrics = {
        'model_type': 'Standard LSTM Regressor',
        'overall': {
            'train_r2': float(r2_train), 'test_r2': float(r2_test),
            'train_rmse': float(rmse_train), 'test_rmse': float(rmse_test),
            'train_mae': float(mae_train), 'test_mae': float(mae_test)
        }
    }
    
    metrics_path = os.path.join(analysis_dir, "rvi_to_ndvi_lstm_standard_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()

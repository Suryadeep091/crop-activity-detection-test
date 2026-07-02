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

    from captum.attr import IntegratedGradients
    import seaborn as sns
    import matplotlib.pyplot as plt

    # --- WORKAROUND FOR CUDNN RNN BACKWARD EVAL ERROR ---
    # 1. Temporarily switch the model to train mode so cuDNN allows backward passes
    model.train()
    
    # 2. Manually freeze the dropout layers so they act as if they are in eval mode
    # This ensures your feature attributions aren't corrupted by random dropout masks
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    # ---------------------------------------------------

    # Initialize Integrated Gradients with your BiGRU model
    ig = IntegratedGradients(model)

    # Grab a batch of sequences from your DataLoader
    batch_x, _ = next(iter(test_loader))
    batch_x = batch_x.to(device).requires_grad_()

    print("Computing Integrated Gradients attributions...")
    # Calculate attributions for a target prediction (target=0 because output_dim is 1)
    attributions, delta = ig.attribute(batch_x, target=0, return_convergence_delta=True)

    # Revert model safely back to standard eval mode
    model.eval()

    # Average attributions across the test batch for visualization
    mean_attributions = attributions.cpu().detach().numpy().mean(axis=0)

    # Plot as a 2D Heatmap (Time vs. Feature)
    plt.figure(figsize=(14, 7))
    sns.heatmap(
        mean_attributions.T, 
        xticklabels=[f"t={i}" for i in range(seq_length)], 
        yticklabels=feature_cols, 
        cmap="PiYG", 
        center=0,
        cbar_kws={'label': 'Attribution Intensity'}
    )
    plt.title("BiGRU Feature Attribution Map Across Temporal Window")
    plt.xlabel("Sequence Time Step")
    plt.ylabel("Features")
    plt.tight_layout()
    
    # Save the attribution heatmap to your analysis directory
    heatmap_path = os.path.join(analysis_dir, "bigru_feature_attribution_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Attribution heatmap successfully saved to: {heatmap_path}")

    # 1. Take the absolute values of the attributions so negative/positive don't cancel out
    # 2. Average them across the 5 time steps (axis=1)
    global_bigru_importance = np.mean(np.abs(mean_attributions), axis=1)

    # Sort them for a clean bar plot
    indices = np.argsort(global_bigru_importance)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importance = global_bigru_importance[indices]

    # Plot a simple horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_importance[::-1], color='mediumseagreen')
    plt.yticks(range(len(sorted_features)), sorted_features[::-1])
    plt.xlabel("Average Absolute Attribution (Importance)")
    plt.title("Overall Feature Importance in the BiGRU Model")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    
    import matplotlib.pyplot as plt

    # 1. Take absolute attributions to prevent negative/positive scores from canceling out
    # mean_attributions shape from your script: (24, 5) or (5, 24) depending on how you arrayed it.
    # We ensure it computes across the time dimension (axis=1) for all 24 features.
    if mean_attributions.shape[0] == len(feature_cols):
        global_bigru_importance = np.mean(np.abs(mean_attributions), axis=1)
    else:
        global_bigru_importance = np.mean(np.abs(mean_attributions), axis=0)

    # 2. Sort all 24 features in descending order
    indices = np.argsort(global_bigru_importance)  # Ascending order for bottom-to-top plot orientation
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importance = global_bigru_importance[indices]

    # 3. Plot the complete chart
    plt.figure(figsize=(12, 10))  # Expanded height to easily accommodate all 24 feature labels
    plt.barh(range(len(sorted_features)), sorted_importance, color='mediumseagreen', edgecolor='black', alpha=0.8)
    plt.yticks(range(len(sorted_features)), sorted_features, fontsize=10)
    plt.xlabel("Global Average Absolute Attribution Intensity", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title("BiGRU Deep Learning Branch: Global Feature Importance (All 24 Features)", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3, axis='x')
    plt.tight_layout()

    # Save full chart to analysis directory
    plt.savefig(os.path.join(analysis_dir, "bigru_global_importance_all_features.png"), dpi=300)
    plt.show()
    plt.close()

    import matplotlib.pyplot as plt

    # Ensure attributions match feature array layout (24 rows, 5 columns)
    if mean_attributions.shape[0] != len(feature_cols):
        mean_attributions = mean_attributions.T

    # Define chronological time steps matching your create_sequences sequence framework
    time_labels = ['t-2 (Past)', 't-1', 't (Target Day)', 't+1', 't+2 (Future)']

    # Group your features into thematic subsets for clean, side-by-side plotting
    groups = {
        "Radar Indices & Key Lags/Leads": ['raw_RVI', 'RVI_lag_6', 'RVI_lag_12', 'RVI_lead_6', 'RVI_lead_12'],
        "Geographic & Dynamic Weather Trends": ['latitude', 'longitude', 'doy_sin', 'doy_cos', 'Rainfall_15d_sum', 'MaxTemp_7d_avg', 'MinTemp_7d_avg'],
        "Static LULC Context/Guardbands": ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice', 'is_kharif', 'is_rabi', 'is_zaid']
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    fig.suptitle("BiGRU Temporal Attention Profile Across All 24 Features", fontsize=16, fontweight='bold', y=0.96)

    # Generate unique color mappings from a colormap
    cmap = plt.get_cmap('tab20')

    color_idx = 0
    for ax_idx, (group_title, f_list) in enumerate(groups.items()):
        ax = axes[ax_idx]
        ax.axhline(0, color='black', linestyle='-', alpha=0.4, linewidth=1.2) # Baseline zero importance
        
        for f_name in f_list:
            if f_name in feature_cols:
                f_idx = feature_cols.index(f_name)
                temporal_profile = mean_attributions[f_idx, :]
                
                # Plot temporal line
                ax.plot(range(5), temporal_profile, marker='o', linewidth=2, label=f_name, color=cmap(color_idx % 20))
                color_idx += 1
                
        ax.set_title(group_title, fontsize=13, fontweight='semibold', pad=10)
        ax.set_ylabel("Attribution Magnitude", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.02), fontsize=9, framealpha=0.9)

    plt.xticks(range(5), time_labels, fontsize=11)
    plt.xlabel("Timeline Position inside Sequence Window (T=5)", fontsize=12, labelpad=10)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, right=0.85, hspace=0.25)

    # Save full multi-line chart to analysis directory
    plt.savefig(os.path.join(analysis_dir, "bigru_temporal_profile_all_features.png"), dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

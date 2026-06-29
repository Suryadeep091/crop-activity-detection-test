import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    dataset_path = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\data\model_training_dataset.csv"
    models_dir = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\models"
    analysis_dir = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\analysis"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} does not exist. Please run scripts/create_training_dataset.py first.")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # 1. Defining Full Feature Set (24 Features)
    feature_cols = [
        'latitude', 'longitude', 'raw_RVI',
        'is_kharif', 'is_rabi', 'is_zaid', 'doy_sin', 'doy_cos',
        'Rainfall_15d_sum', 'MaxTemp_7d_avg', 'MinTemp_7d_avg',
        'water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice',
        'RVI_lag_12', 'RVI_lag_6', 'RVI_lead_6', 'RVI_lead_12'
    ]
    target_col = 'raw_NDVI'
    
    X = df[feature_cols]
    y = df[target_col]
    groups = df['task_id']
    
    print(f"Features count: {len(feature_cols)}")
    print(f"Features list: {feature_cols}")
    print(f"Target variable: {target_col}")
    
    # 2. Train-Test Split (Group-based to prevent spatial leakage)
    print("Performing group-based train-test split (80-20, grouped by task_id)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    
    # 3. Model Training
    model = None
    model_name = ""
    try:
        import xgboost as xgb
        print("XGBoost detected. Training XGBRegressor...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model_name = "XGBoost Regressor"
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        print("XGBoost not installed. Falling back to RandomForestRegressor...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        model_name = "Random Forest Regressor"
        
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    # Save the trained model
    model_path = os.path.join(models_dir, "rvi_to_ndvi_model.pkl")
    joblib.dump(model, model_path)
    
    # 4. Testing & Evaluation
    print("Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_pred_clipped = np.clip(y_train_pred, 0.0, 1.0)
    y_test_pred_clipped = np.clip(y_test_pred, 0.0, 1.0)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    mae_train = mean_absolute_error(y_train, y_train_pred_clipped)
    mae_test = mean_absolute_error(y_test, y_test_pred_clipped)
    
    # Calculate Season-wise Metrics
    seasons = {'Kharif': 'is_kharif', 'Rabi': 'is_rabi', 'Zaid': 'is_zaid'}
    season_metrics = {}
    
    for s_name, col in seasons.items():
        mask = df_test[col] == 1
        if mask.sum() > 0:
            y_s_actual = y_test[mask]
            y_s_pred = y_test_pred_clipped[mask]
            
            s_r2 = r2_score(y_s_actual, y_s_pred)
            s_rmse = np.sqrt(mean_squared_error(y_s_actual, y_s_pred))
            s_mae = mean_absolute_error(y_s_actual, y_s_pred)
            
            season_metrics[s_name] = {
                'R2': s_r2,
                'RMSE': s_rmse,
                'MAE': s_mae,
                'Samples': int(mask.sum())
            }
            print(f"Season {s_name} Metrics: R2={s_r2:.4f}, RMSE={s_rmse:.4f}, MAE={s_mae:.4f}")
            
    # Save metrics dictionary
    expanded_metrics = {
        'model_type': model_name,
        'overall': {
            'train_r2': r2_train, 'test_r2': r2_test,
            'train_rmse': rmse_train, 'test_rmse': rmse_test,
            'train_mae': mae_train, 'test_mae': mae_test
        },
        'seasonal': season_metrics
    }
    
    metrics_json_path = os.path.join(analysis_dir, "rvi_to_ndvi_expanded_metrics.json")
    with open(metrics_json_path, 'w') as f_json:
        json.dump(expanded_metrics, f_json, indent=4)
        
    # Save standard text file
    with open(os.path.join(analysis_dir, "rvi_to_ndvi_metrics.txt"), "w") as f_metrics:
        f_metrics.write(f"Model Type: {model_name}\n")
        f_metrics.write(f"Train R^2: {r2_train:.4f} | Test R^2: {r2_test:.4f}\n")
        f_metrics.write(f"Train RMSE: {rmse_train:.4f} | Test RMSE: {rmse_test:.4f}\n")
        f_metrics.write(f"Train MAE: {mae_train:.4f} | Test MAE: {mae_test:.4f}\n")
        f_metrics.write("\nSeasonal Metrics (Test Split):\n")
        for s_name, vals in season_metrics.items():
            f_metrics.write(f"  {s_name}: R^2={vals['R2']:.4f} | RMSE={vals['RMSE']:.4f} | MAE={vals['MAE']:.4f} (Samples={vals['Samples']})\n")

    # 5. Visualizations
    print("Generating visualization plots...")
    
    # Plot 1: Scatter Plot Actual vs. Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred_clipped, alpha=0.3, color='forestgreen', edgecolors='none')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label="1:1 Perfect Prediction Line")
    plt.title(f"RVI to NDVI: Actual vs. Predicted ({model_name})")
    plt.xlabel("Actual NDVI")
    plt.ylabel("Predicted NDVI")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "rvi_to_ndvi_scatter.png"), dpi=300)
    plt.close()
    
    # Plot 2: Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_cols[i] for i in indices]
        sorted_importances = importances[indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_features)), sorted_importances[::-1], align='center', color='royalblue')
        plt.yticks(range(len(sorted_features)), sorted_features[::-1])
        plt.xlabel("Relative Feature Importance")
        plt.title("RVI to NDVI Model: Feature Importances")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "feature_importance_ndvi.png"), dpi=300)
        plt.close()
        
    # Plot 3: Temporal prediction curve for a representative test parcel
    test_parcels = df_test['task_id'].unique()
    sample_parcel_id = next((p for p in test_parcels if len(df_test[df_test['task_id'] == p]) >= 15), test_parcels[0] if len(test_parcels) > 0 else None)
        
    if sample_parcel_id is not None:
        df_sample = df[df['task_id'] == sample_parcel_id].copy()
        df_sample['date'] = pd.to_datetime(df_sample['date'])
        df_sample = df_sample.sort_values('date')
        
        X_sample = df_sample[feature_cols]
        y_sample_actual = df_sample[target_col]
        y_sample_pred = np.clip(model.predict(X_sample), 0.0, 1.0)
        
        plt.figure(figsize=(12, 5))
        plt.plot(df_sample['date'], y_sample_actual, 'o-', label='Actual S2 NDVI', color='green', markersize=6)
        plt.plot(df_sample['date'], y_sample_pred, 's--', label='Predicted NDVI (from S1 RVI)', color='darkorange', markersize=6)
        plt.title(f"Temporal Prediction Curve: {sample_parcel_id}")
        plt.xlabel("Date")
        plt.ylabel("NDVI")
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "temporal_prediction_sample.png"), dpi=300)
        plt.close()
        
    # Plot 4: Residuals Distribution Histogram
    residuals = y_test - y_test_pred_clipped
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, color='crimson', alpha=0.7, edgecolor='black', density=True)
    from scipy.stats import norm
    mu, std = norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x_range = np.linspace(xmin, xmax, 100)
    p_fit = norm.pdf(x_range, mu, std)
    plt.plot(x_range, p_fit, 'k-', linewidth=2, label=f'Normal Fit ($\mu={mu:.3f}, \sigma={std:.3f}$)')
    plt.axvline(0, color='blue', linestyle='--', linewidth=1.5, label='Zero Error')
    
    plt.title("RVI to NDVI Model: Residuals Distribution")
    plt.xlabel("Residual Value (Actual - Predicted)")
    plt.ylabel("Density")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "residuals_histogram.png"), dpi=300)
    plt.close()
    
    # Plot 5: Residuals vs. Actual NDVI
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.25, color='darkviolet', edgecolors='none')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title("RVI to NDVI Model: Residuals vs. Ground Truth NDVI")
    plt.xlabel("Ground Truth NDVI")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "residuals_vs_actual.png"), dpi=300)
    plt.close()
    
    # Plot 6: Season-wise Error Metrics (Bar Chart)
    s_names = list(season_metrics.keys())
    rmse_vals = [season_metrics[s]['RMSE'] for s in s_names]
    mae_vals = [season_metrics[s]['MAE'] for s in s_names]
    
    x = np.arange(len(s_names))
    width = 0.35
    
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, rmse_vals, width, label='RMSE', color='indianred')
    plt.bar(x + width/2, mae_vals, width, label='MAE', color='goldenrod')
    
    plt.title("RVI to NDVI Model: Seasonal Error Comparison")
    plt.xlabel("Cropping Season")
    plt.ylabel("Error Magnitude (NDVI units)")
    plt.xticks(x, s_names)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "season_wise_metrics.png"), dpi=300)
    plt.close()
    
    print("All training and metrics tasks successfully completed!")

if __name__ == "__main__":
    main()

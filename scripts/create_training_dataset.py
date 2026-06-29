import os
import pandas as pd
import numpy as np
import time

def main():
    raw_csv_path = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\data\raw_observations_dataset.csv"
    output_dir = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\data"
    output_file = os.path.join(output_dir, "model_training_dataset.csv")
    
    print(f"Loading raw observations from: {raw_csv_path}")
    if not os.path.exists(raw_csv_path):
        print(f"Error: {raw_csv_path} does not exist. Please run create_dataset_from_pickles.py first.")
        return
        
    df_raw_all = pd.read_csv(raw_csv_path)
    df_raw_all['date'] = pd.to_datetime(df_raw_all['date'])
    
    # Sort by task_id and date
    df_raw_all = df_raw_all.sort_values(['task_id', 'date'])
    
    parcels = df_raw_all['task_id'].unique()
    print(f"Processing features and temporal alignment for {len(parcels)} parcels...")
    
    dw_cols = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    
    all_aligned_dfs = []
    processed_count = 0
    start_time = time.time()
    
    for task_id in parcels:
        df_parcel = df_raw_all[df_raw_all['task_id'] == task_id].copy()
        
        # 1. Create a continuous daily series for calculating lags/leads and weather rolling windows
        min_date = df_parcel['date'].min()
        max_date = df_parcel['date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        
        df_daily = pd.DataFrame({'date': all_dates})
        df_daily['task_id'] = task_id
        df_daily['latitude'] = df_parcel['latitude'].iloc[0]
        df_daily['longitude'] = df_parcel['longitude'].iloc[0]
        
        # Merge raw observations
        df_parcel_clean = df_parcel[['date', 'raw_NDVI', 'raw_EVI', 'raw_RVI'] + dw_cols + ['Rainfall_mm', 'Max_temp_celsius', 'Min_temp_celsius']]
        df_daily = df_daily.merge(df_parcel_clean, on='date', how='left')
        
        # 2. Interpolate Dynamic World columns daily (to get daily continuous probs)
        df_daily[dw_cols] = df_daily[dw_cols].interpolate(method='linear', limit_direction='both').fillna(0)
        
        # 3. Interpolate Weather data daily to fill missing daily steps (if any)
        weather_cols = ['Rainfall_mm', 'Max_temp_celsius', 'Min_temp_celsius']
        df_daily[weather_cols] = df_daily[weather_cols].interpolate(method='linear', limit_direction='both').fillna(0)
        
        # Calculate daily weather moving windows
        df_daily['Rainfall_15d_sum'] = df_daily['Rainfall_mm'].rolling(window=15, min_periods=1).sum()
        df_daily['MaxTemp_7d_avg'] = df_daily['Max_temp_celsius'].rolling(window=7, min_periods=1).mean()
        df_daily['MinTemp_7d_avg'] = df_daily['Min_temp_celsius'].rolling(window=7, min_periods=1).mean()
        
        # 4. Create daily interpolated columns for NDVI and RVI to calculate lags/leads
        df_daily['interp_NDVI'] = df_daily['raw_NDVI'].interpolate(method='linear', limit_direction='both').fillna(0)
        df_daily['interp_RVI'] = df_daily['raw_RVI'].interpolate(method='linear', limit_direction='both').fillna(0)
        
        # Calculate lags and leads (shift shifts dates)
        # Shift 6 and 12 days for NDVI
        df_daily['NDVI_lag_12'] = df_daily['interp_NDVI'].shift(12).fillna(method='bfill').fillna(0)
        df_daily['NDVI_lag_6'] = df_daily['interp_NDVI'].shift(6).fillna(method='bfill').fillna(0)
        df_daily['NDVI_lead_6'] = df_daily['interp_NDVI'].shift(-6).fillna(method='ffill').fillna(0)
        df_daily['NDVI_lead_12'] = df_daily['interp_NDVI'].shift(-12).fillna(method='ffill').fillna(0)
        
        # Shift 6 and 12 days for RVI
        df_daily['RVI_lag_12'] = df_daily['interp_RVI'].shift(12).fillna(method='bfill').fillna(0)
        df_daily['RVI_lag_6'] = df_daily['interp_RVI'].shift(6).fillna(method='bfill').fillna(0)
        df_daily['RVI_lead_6'] = df_daily['interp_RVI'].shift(-6).fillna(method='ffill').fillna(0)
        df_daily['RVI_lead_12'] = df_daily['interp_RVI'].shift(-12).fillna(method='ffill').fillna(0)
        
        # 5. Extract circular DOY and India-specific seasons
        doys = df_daily['date'].dt.dayofyear
        df_daily['doy_sin'] = np.sin(2 * np.pi * doys / 365.25)
        df_daily['doy_cos'] = np.cos(2 * np.pi * doys / 365.25)
        
        months = df_daily['date'].dt.month
        # India Cropping Seasons: Kharif (June-Oct), Rabi (Nov-Mar), Zaid (Apr-May)
        df_daily['is_kharif'] = ((months >= 6) & (months <= 10)).astype(int)
        df_daily['is_rabi'] = ((months >= 11) | (months <= 3)).astype(int)
        df_daily['is_zaid'] = ((months == 4) | (months == 5)).astype(int)
        
        # 6. Temporal Alignment: Pair raw NDVI and raw RVI observations within 3 days
        # Get raw NDVI observation dates
        df_raw_ndvi = df_daily[df_daily['raw_NDVI'].notna()][['date', 'raw_NDVI', 'raw_EVI']]
        # Get raw RVI observation dates
        df_raw_rvi = df_daily[df_daily['raw_RVI'].notna()][['date', 'raw_RVI']]
        
        if df_raw_ndvi.empty or df_raw_rvi.empty:
            continue
            
        pairs = []
        for _, row_ndvi in df_raw_ndvi.iterrows():
            d1 = row_ndvi['date']
            # Find closest RVI observation
            time_diffs = (df_raw_rvi['date'] - d1).abs()
            min_diff = time_diffs.min()
            if min_diff <= pd.Timedelta(days=3):
                best_idx = time_diffs.idxmin()
                d2 = df_raw_rvi.loc[best_idx, 'date']
                pairs.append({
                    'ndvi_date': d1,
                    'rvi_date': d2,
                    'raw_NDVI': row_ndvi['raw_NDVI'],
                    'raw_EVI': row_ndvi['raw_EVI'],
                    'raw_RVI': df_raw_rvi.loc[best_idx, 'raw_RVI'],
                    'time_diff_days': (d1 - d2).days
                })
                
        if not pairs:
            continue
            
        df_pairs = pd.DataFrame(pairs)
        # Deduplicate to prevent double-counting
        df_pairs = df_pairs.drop_duplicates(subset=['ndvi_date', 'rvi_date'])
        
        # 7. Merge daily engineered features using the ndvi_date (which will be the main record 'date')
        df_daily_features = df_daily.drop(columns=['raw_NDVI', 'raw_EVI', 'raw_RVI', 'interp_NDVI', 'interp_RVI'])
        
        # Rename ndvi_date to date for merging
        df_pairs = df_pairs.rename(columns={'ndvi_date': 'date'})
        df_aligned = df_pairs.merge(df_daily_features, on='date', how='left')
        
        all_aligned_dfs.append(df_aligned)
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"Engineered features for {processed_count}/{len(parcels)} parcels...")
            
    if all_aligned_dfs:
        print("Combining all aligned datasets...")
        final_df = pd.concat(all_aligned_dfs, ignore_index=True)
        
        # Organize and reorder columns
        cols_order = [
            'task_id', 'date', 'rvi_date', 'time_diff_days', 'latitude', 'longitude',
            'raw_NDVI', 'raw_EVI', 'raw_RVI',
            'is_kharif', 'is_rabi', 'is_zaid',
            'doy_sin', 'doy_cos',
            'Rainfall_15d_sum', 'MaxTemp_7d_avg', 'MinTemp_7d_avg'
        ] + dw_cols + [
            'NDVI_lag_12', 'NDVI_lag_6', 'NDVI_lead_6', 'NDVI_lead_12',
            'RVI_lag_12', 'RVI_lag_6', 'RVI_lead_6', 'RVI_lead_12'
        ]
        
        # Make sure all columns exist
        for col in cols_order:
            if col not in final_df.columns:
                final_df[col] = 0.0
                
        final_df = final_df[cols_order]
        
        # Convert datetime columns to string
        final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
        final_df['rvi_date'] = final_df['rvi_date'].dt.strftime('%Y-%m-%d')
        
        print(f"Saving training dataset of {len(final_df)} aligned rows to: {output_file}")
        final_df.to_csv(output_file, index=False)
        
        elapsed = time.time() - start_time
        print(f"Feature engineering completed in {elapsed:.2f} seconds.")
        print(f"Total processed parcels: {processed_count}")
        print(f"Total training pairs: {len(final_df)}")
    else:
        print("Error: No training pairs could be aligned.")

if __name__ == "__main__":
    main()

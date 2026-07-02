import os
import pandas as pd
import numpy as np
import time

def main():
    raw_csv_path = r"/home/surya/Downloads/Old Repos/AdvaRisk - Test/data/raw_observations_dataset.csv"
    output_dir = r"/home/surya/Downloads/Old Repos/AdvaRisk - Test/data"
    output_file = os.path.join(output_dir, "new_training_dataset.csv")
    
    print(f"Loading raw observations from: {raw_csv_path}")
    if not os.path.exists(raw_csv_path):
        print(f"Error: {raw_csv_path} does not exist.")
        return
        
    df_raw_all = pd.read_csv(raw_csv_path)
    df_raw_all['date'] = pd.to_datetime(df_raw_all['date'])
    df_raw_all = df_raw_all.sort_values(['task_id', 'date'])
    
    parcels = df_raw_all['task_id'].unique()
    print(f"Processing velocity features and structural categories for {len(parcels)} parcels...")
    
    dw_cols = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    all_aligned_dfs = []
    processed_count = 0
    start_time = time.time()
    
    for task_id in parcels:
        df_parcel = df_raw_all[df_raw_all['task_id'] == task_id].copy()
        
        # Continuous daily series setup
        min_date = df_parcel['date'].min()
        max_date = df_parcel['date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        
        df_daily = pd.DataFrame({'date': all_dates})
        df_daily['task_id'] = task_id
        df_daily['latitude'] = df_parcel['latitude'].iloc[0]
        df_daily['longitude'] = df_parcel['longitude'].iloc[0]
        
        # Merge raw metrics
        df_parcel_clean = df_parcel[['date', 'raw_NDVI', 'raw_EVI', 'raw_RVI'] + dw_cols + ['Rainfall_mm', 'Max_temp_celsius', 'Min_temp_celsius']]
        df_daily = df_daily.merge(df_parcel_clean, on='date', how='left')
        
        # Compress 9 LULC channels down into 1 single categorical profile
        df_daily[dw_cols] = df_daily[dw_cols].interpolate(method='linear', limit_direction='both').fillna(0)
        parcel_lulc_profile = df_daily[dw_cols].median()
        dominant_raw_class = parcel_lulc_profile.idxmax()
        
        if dominant_raw_class in ['water', 'built', 'bare', 'snow_and_ice']:
            simplified_guardband = 0
        elif dominant_raw_class == 'crops':
            simplified_guardband = 1
        else:
            simplified_guardband = 2
        df_daily['dominant_land_cover_class'] = simplified_guardband
        
        # Weather moving loops
        weather_cols = ['Rainfall_mm', 'Max_temp_celsius', 'Min_temp_celsius']
        df_daily[weather_cols] = df_daily[weather_cols].interpolate(method='linear', limit_direction='both').fillna(0)
        df_daily['Rainfall_15d_sum'] = df_daily['Rainfall_mm'].rolling(window=15, min_periods=1).sum()
        df_daily['MaxTemp_7d_avg'] = df_daily['Max_temp_celsius'].rolling(window=7, min_periods=1).mean()
        df_daily['MinTemp_7d_avg'] = df_daily['Min_temp_celsius'].rolling(window=7, min_periods=1).mean()
        
        # Radar temporal step extraction
        df_daily['interp_RVI'] = df_daily['raw_RVI'].interpolate(method='linear', limit_direction='both').fillna(0)
        
        df_daily['RVI_lag_12'] = df_daily['interp_RVI'].shift(12).fillna(method='bfill').fillna(0)
        df_daily['RVI_lag_6'] = df_daily['interp_RVI'].shift(6).fillna(method='bfill').fillna(0)
        df_daily['RVI_lead_6'] = df_daily['interp_RVI'].shift(-6).fillna(method='ffill').fillna(0)
        df_daily['RVI_lead_12'] = df_daily['interp_RVI'].shift(-12).fillna(method='ffill').fillna(0)
        
        # Velocity and slope tracking engine variables
        df_daily['RVI_velocity_6'] = df_daily['interp_RVI'] - df_daily['RVI_lag_6']
        df_daily['RVI_velocity_12'] = df_daily['interp_RVI'] - df_daily['RVI_lag_12']
        df_daily['RVI_velocity_lead_6'] = df_daily['RVI_lead_6'] - df_daily['interp_RVI']
        df_daily['RVI_velocity_lead_12'] = df_daily['RVI_lead_12'] - df_daily['interp_RVI']
        
        # Extract astronomical sin/cos features
        doys = df_daily['date'].dt.dayofyear
        df_daily['doy_sin'] = np.sin(2 * np.pi * doys / 365.25)
        df_daily['doy_cos'] = np.cos(2 * np.pi * doys / 365.25)
        
        months = df_daily['date'].dt.month
        df_daily['is_kharif'] = ((months >= 6) & (months <= 10)).astype(int)
        df_daily['is_rabi'] = ((months >= 11) | (months <= 3)).astype(int)
        df_daily['is_zaid'] = ((months == 4) | (months == 5)).astype(int)
        
        # Coordinate matching loops
        df_raw_ndvi = df_daily[df_daily['raw_NDVI'].notna()][['date', 'raw_NDVI', 'raw_EVI']]
        df_raw_rvi = df_daily[df_daily['raw_RVI'].notna()][['date', 'raw_RVI']]
        
        if df_raw_ndvi.empty or df_raw_rvi.empty:
            continue
            
        pairs = []
        for _, row_ndvi in df_raw_ndvi.iterrows():
            d1 = row_ndvi['date']
            time_diffs = (df_raw_rvi['date'] - d1).abs()
            if time_diffs.min() <= pd.Timedelta(days=3):
                best_idx = time_diffs.idxmin()
                pairs.append({
                    'date': d1,
                    'rvi_date': df_raw_rvi.loc[best_idx, 'date'],
                    'raw_NDVI': row_ndvi['raw_NDVI'],
                    'raw_EVI': row_ndvi['raw_EVI'],
                    'raw_RVI': df_raw_rvi.loc[best_idx, 'raw_RVI'],
                    'time_diff_days': (d1 - df_raw_rvi.loc[best_idx, 'date']).days
                })
                
        if not pairs:
            continue
            
        df_pairs = pd.DataFrame(pairs).drop_duplicates(subset=['date', 'rvi_date'])
        
        # Drop raw continuous DW matrices
        df_daily_features = df_daily.drop(columns=['raw_NDVI', 'raw_EVI', 'raw_RVI', 'interp_RVI'] + dw_cols)
        df_aligned = df_pairs.merge(df_daily_features, on='date', how='left')
        all_aligned_dfs.append(df_aligned)
        processed_count += 1
        
    if all_aligned_dfs:
        final_df = pd.concat(all_aligned_dfs, ignore_index=True)
        
        cols_order = [
            'task_id', 'date', 'rvi_date', 'time_diff_days', 'latitude', 'longitude',
            'raw_NDVI', 'raw_EVI', 'raw_RVI',
            'is_kharif', 'is_rabi', 'is_zaid', 'doy_sin', 'doy_cos',
            'Rainfall_15d_sum', 'MaxTemp_7d_avg', 'MinTemp_7d_avg',
            'dominant_land_cover_class',
            'RVI_lag_12', 'RVI_lag_6', 'RVI_lead_6', 'RVI_lead_12',
            'RVI_velocity_6', 'RVI_velocity_12', 'RVI_velocity_lead_6', 'RVI_velocity_lead_12'
        ]
        
        final_df = final_df[[c for c in cols_order if c in final_df.columns]]
        final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
        final_df['rvi_date'] = final_df['rvi_date'].dt.strftime('%Y-%m-%d')
        
        print(f"Saving velocity upgraded matrix data ({len(final_df)} rows) to: {output_file}")
        final_df.to_csv(output_file, index=False)
    else:
        print("Error: Dataset generation empty.")

if __name__ == "__main__":
    main()
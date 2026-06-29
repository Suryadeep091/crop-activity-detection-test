import os
import pickle
import pandas as pd
import numpy as np
import time

def compute_centroid(coords):
    if not coords or len(coords) == 0:
        return None, None
    # coords is [[lon, lat], ...]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return sum(lats) / len(lats), sum(lons) / len(lons)

def main():
    pickle_dir = r"C:\Users\Suryadeep Singh\Downloads\pickle_files"
    output_dir = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "raw_observations_dataset.csv")
    
    print(f"Reading pickle files from: {pickle_dir}")
    if not os.path.exists(pickle_dir):
        print(f"Error: {pickle_dir} directory does not exist.")
        return
        
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith(".pkl") or f.endswith(".pickle")]
    print(f"Found {len(pickle_files)} pickle files.")
    
    all_dfs = []
    processed_count = 0
    start_time = time.time()
    
    dw_cols = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    
    for idx, f_name in enumerate(pickle_files):
        f_path = os.path.join(pickle_dir, f_name)
        try:
            with open(f_path, "rb") as f:
                data = pickle.load(f)
                
            task_id = data.get("task_id", f_name.replace("_full.pkl", "").replace("_full_2024.pkl", "").replace("_full_2025.pkl", ""))
            coords = data.get("coords", [])
            lat, lon = compute_centroid(coords)
            
            raw_veg = data.get("raw_vegetation_indices", [])
            dw_probs = data.get("land_cover_probs", [])
            weather_data_dict = data.get("weather_data", {})
            weather_daily = weather_data_dict.get("daily_weather_data", []) if isinstance(weather_data_dict, dict) else []
            
            if not raw_veg:
                # Fallback to vegetation_indices if raw_vegetation_indices is missing, but filter for non-null
                continue
                
            df_raw = pd.DataFrame(raw_veg)
            if df_raw.empty:
                continue
                
            # Rename columns to show they are raw
            rename_dict = {}
            for col in ['NDVI', 'EVI', 'RVI']:
                if col in df_raw.columns:
                    rename_dict[col] = f"raw_{col}"
            df_raw = df_raw.rename(columns=rename_dict)
            
            # Convert date to datetime
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            
            # Check if we have at least one valid index
            index_cols = [c for c in ['raw_NDVI', 'raw_EVI', 'raw_RVI'] if c in df_raw.columns]
            if not index_cols:
                continue
            df_raw = df_raw.dropna(subset=index_cols, how='all')
            if df_raw.empty:
                continue
                
            # Clean and prepare DW probabilities
            df_dw = pd.DataFrame(dw_probs)
            if not df_dw.empty:
                df_dw['date'] = pd.to_datetime(df_dw['date'])
                df_dw = df_dw.drop_duplicates(subset=['date'])
                
            # Clean and prepare Weather data
            df_weather = pd.DataFrame(weather_daily)
            if not df_weather.empty:
                # Check column name case (can be 'Date' or 'date')
                date_col = 'Date' if 'Date' in df_weather.columns else 'date'
                df_weather['date'] = pd.to_datetime(df_weather[date_col])
                if date_col != 'date':
                    df_weather = df_weather.drop(columns=[date_col])
                if 'Month_DT' in df_weather.columns:
                    df_weather = df_weather.drop(columns=['Month_DT'])
                df_weather = df_weather.drop_duplicates(subset=['date'])
                
            # Merge and interpolate DW probabilities to raw observation dates
            min_date = min(df_raw['date'].min(), df_dw['date'].min() if not df_dw.empty else df_raw['date'].min())
            max_date = max(df_raw['date'].max(), df_dw['date'].max() if not df_dw.empty else df_raw['date'].max())
            all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            daily_df = pd.DataFrame({'date': all_dates})
            
            if not df_dw.empty:
                # Keep only DW probability columns and date
                cols_to_keep = ['date'] + [c for c in dw_cols if c in df_dw.columns]
                df_dw_clean = df_dw[cols_to_keep]
                daily_df = daily_df.merge(df_dw_clean, on='date', how='left')
                
                # Fill missing DW columns
                for c in dw_cols:
                    if c not in daily_df.columns:
                        daily_df[c] = np.nan
                        
                # Linearly interpolate probability gaps daily
                daily_df[dw_cols] = daily_df[dw_cols].interpolate(method='linear', limit_direction='both').fillna(0)
            else:
                # If no DW data, fill with zeros
                for c in dw_cols:
                    daily_df[c] = 0.0
                    
            # Merge interpolated DW back to raw observation dates
            df_raw = df_raw.merge(daily_df, on='date', how='left')
            
            # Merge Weather back to raw observation dates
            if not df_weather.empty:
                df_raw = df_raw.merge(df_weather, on='date', how='left')
            else:
                # Add empty weather columns
                for w_col in ['Rainfall_mm', 'Max_temp_celsius', 'Min_temp_celsius']:
                    df_raw[w_col] = np.nan
                    
            # Add spatial context
            df_raw['task_id'] = task_id
            df_raw['latitude'] = lat
            df_raw['longitude'] = lon
            
            # Reorder columns nicely
            cols_order = ['task_id', 'date', 'latitude', 'longitude'] + \
                         [f'raw_{c}' for c in ['NDVI', 'EVI', 'RVI']] + \
                         dw_cols + \
                         ['Rainfall_mm', 'Max_temp_celsius', 'Min_temp_celsius']
            
            # Ensure all output columns exist (in case raw_EVI etc were completely missing from the dict)
            for col in cols_order:
                if col not in df_raw.columns:
                    df_raw[col] = np.nan
                    
            df_raw = df_raw[cols_order]
            all_dfs.append(df_raw)
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(pickle_files)} files...")
                
        except Exception as e:
            print(f"Error processing {f_name}: {e}")
            
    if all_dfs:
        print("Combining all parcel dataframes...")
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Convert date to string format for CSV
        final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
        
        print(f"Saving compiled dataset of {len(final_df)} rows to: {output_file}")
        final_df.to_csv(output_file, index=False)
        
        elapsed = time.time() - start_time
        print(f"Dataset compilation completed successfully in {elapsed:.2f} seconds.")
        print(f"Total processed files: {processed_count}")
        print(f"Total rows in dataset: {len(final_df)}")
        print(f"Columns in dataset: {list(final_df.columns)}")
    else:
        print("No data frames could be compiled.")

if __name__ == "__main__":
    main()

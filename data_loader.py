import os
import ee
import pandas as pd
import datetime
import numpy as np
from collections import defaultdict
from functools import reduce
import scipy.signal
import google.auth
from scipy.signal import savgol_filter, find_peaks
from shapely.geometry import Polygon
import math, time, requests

try:
    # Use the default service account credentials provided by Cloud Run
    source_creds, project_id = google.auth.default()
    
    # Initialize Earth Engine with the credentials
    ee.Initialize(
        source_creds, 
        project=project_id, 
        opt_url='https://earthengine-highvolume.googleapis.com' # Best for APIs
    )
    print("DEBUG: Earth Engine initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Earth Engine failed to initialize: {e}")


def initialize_ee():
    try:
        service_account = os.getenv("GEE_SERVICE_ACCOUNT")
        json_path = "/secrets/gee-key.json/GOOGLE_APPLICATION_CREDENTIALS" 
        project_id = "advarisk" 
        
        if os.path.exists(json_path) and service_account:
            # PROD: Uses Secret Manager mount
            credentials = ee.ServiceAccountCredentials(service_account, json_path)
            ee.Initialize(credentials, project=project_id)
            print(f"✅ GEE Initialized via Secret Manager: {service_account}")
        elif service_account:
            # FALLBACK: Uses Metadata Server
            credentials = ee.ServiceAccountCredentials(service_account, key_data=None)
            ee.Initialize(credentials, project=project_id)
            print(f"✅ GEE Initialized via Metadata Server: {service_account}")
        else:
            # LOCAL: Uses your gcloud auth
            ee.Initialize(project=project_id)
            print("✅ GEE Initialized via local default credentials")
    except Exception as e:
        print(f"❌ GEE Initialization Failed: {e}")

# Run the initialization
initialize_ee()



# Function to count valid NDVI peaks (crop cycles) using the thumb rule
# A valid peak: NDVI rises for >=21 days, stays high for >=35 days, falls for >=21 days (total ~77 days)





def detect_crop_cycles(df):
    if df.empty or len(df) < 30:
        return {"total_cycles": 0, "detected_seasons": [], "details": [], "type": "insufficient_data"}

    df = df.sort_values('date')
    
    # --- STEP 1: SIGNAL SMOOTHING ---
    window_rvi = 21 if len(df) > 21 else (len(df) // 2 * 2 + 1)
    window_ndvi = 11 if len(df) > 11 else (len(df) // 2 * 2 + 1)
    
    df['rvi_smooth'] = savgol_filter(df['RVI'].fillna(0), window_rvi, 2)
    df['ndvi_smooth'] = savgol_filter(df['NDVI'].fillna(0), window_ndvi, 2)

    # --- NEW: PERENNIAL/FOREST FILTER ---
    # If the NDVI never drops below 0.4, it's likely a forest or orchard, not a crop cycle
    annual_min_ndvi = df['ndvi_smooth'].min()
    annual_range = df['ndvi_smooth'].max() - annual_min_ndvi
    
    if annual_min_ndvi > 0.45 and annual_range < 0.2:
        return {
            "total_cycles": 0, 
            "detected_seasons": ["Perennial/Evergreen"], 
            "details": [],
            "note": "NDVI remains high year-round; likely forest or orchard."
        }
            
    detected_cycles = []
    
    # Helper to validate a peak's "Cycle Integrity"
    def is_valid_cycle(sub_df, peak_idx, index_col, min_amplitude=0.20):
        peak_val = sub_df.iloc[peak_idx][index_col]
        # Look for the minimum value BEFORE the peak in this subset
        baseline_val = sub_df.iloc[:peak_idx][index_col].min() 
        return (peak_val - baseline_val) >= min_amplitude

    # --- STEP 2: KHARIF DETECTION ---
    kharif_mask = (df['date'].dt.month >= 5) & (df['date'].dt.month <= 11)
    kharif_df = df[kharif_mask]
    
    if not kharif_df.empty:
        peaks_rvi, _ = find_peaks(kharif_df['rvi_smooth'], prominence=0.15, distance=30)
        for p in peaks_rvi:
            if is_valid_cycle(kharif_df, p, 'rvi_smooth', 0.20):
                peak_date = kharif_df.iloc[p]['date']
                detected_cycles.append({"season": "Kharif", "peak_date": peak_date, "index_used": "RVI"})
                break

    # --- STEP 3: RABI DETECTION ---
    rabi_mask = (df['date'].dt.month >= 10) | (df['date'].dt.month <= 4)
    rabi_df = df[rabi_mask].sort_values('date')
    
    if not rabi_df.empty:
        peaks_ndvi, _ = find_peaks(rabi_df['ndvi_smooth'], prominence=0.20, distance=45)
        for p in peaks_ndvi:
            # For Rabi, we check the baseline before the peak (Nov/Dec)
            if is_valid_cycle(rabi_df, p, 'ndvi_smooth', 0.25):
                peak_date = rabi_df.iloc[p]['date']
                detected_cycles.append({"season": "Rabi", "peak_date": peak_date, "index_used": "NDVI"})
                break

    # --- STEP 4: ZAID DETECTION ---
    zaid_mask = (df['date'].dt.month >= 3) & (df['date'].dt.month <= 6)
    zaid_df = df[zaid_mask]
    
    if not zaid_df.empty:
        peaks_zaid, _ = find_peaks(zaid_df['ndvi_smooth'], prominence=0.15, distance=20)
        if len(peaks_zaid) > 0:
            if is_valid_cycle(zaid_df, peaks_zaid[0], 'ndvi_smooth', 0.15):
                detected_cycles.append({"season": "Zaid", "peak_date": zaid_df.iloc[peaks_zaid[0]]['date'], "index_used": "NDVI"})

    return {
        "total_cycles": len(detected_cycles),
        "detected_seasons": [c['season'] for c in detected_cycles],
        "details": detected_cycles
    }


# Function to count valid NDVI peaks
def count_crop_cycles(ndvi_series, date_series):
    if len(ndvi_series) < 10:
        return 0
    # Smooth NDVI with a rolling mean (window=7)
    ndvi = pd.Series(ndvi_series).rolling(window=7, min_periods=1, center=True).mean().values
    
    # Calculate gradient (slope) to ensure we only pick peaks with a decent slope
    slope = np.gradient(ndvi)
    
    # Find peaks: at least 30 days apart, some prominence
    peaks, _ = scipy.signal.find_peaks(ndvi, distance=15, prominence=0.10)
    
    valid_cycles = 0
    for peak in peaks:
        # Check if the surrounding slope is acceptable (not just noise)
        max_up_slope = np.max(slope[max(0, peak-15):peak]) if peak > 0 else 0
        min_down_slope = np.min(slope[peak:min(len(slope), peak+15)]) if peak < len(slope) else 0
        
        # If the slope leading up to it and down from it is significant
        if max_up_slope > 0.01 and min_down_slope < -0.01:
            valid_cycles += 1
            
    return valid_cycles

def process_parcel_data(run_id, coordinates, end_str):
    """
    Process parcel data for given coordinates and save results in a run-specific folder.
    
    Args:
        run_id (str): Unique identifier for this run
        coordinates (list): List of coordinate pairs defining the parcel polygon
        end_str (str): End date in YYYY-MM-DD format. Start date will be automatically set to 2 years before.
    """
    try:
        initialize_ee()

        # Create data directory if it doesn't exist
        data_dir = os.path.join("data", run_id)
        if os.path.exists(data_dir):
            print(f"Run {run_id} already exists. Skipping processing.")
            return
        
        # Create directory for this run
        os.makedirs(data_dir, exist_ok=True)
        
        # Calculate start date as 1 year before end date
        end_date = pd.to_datetime(end_str)
        start_date = end_date - pd.DateOffset(years=1)
        start_str = start_date.strftime('%Y-%m-%d')
        
        # Print input parameters for debugging
        print(f"\nInput Parameters:")
        print(f"End date: {end_str}")
        print(f"Auto-calculated start date: {start_str} (1 year before end date)")
        print(f"Coordinates: {coordinates}")
        
        # Convert coordinates to Earth Engine polygon
        polygon = ee.Geometry.Polygon([coordinates])

        # Cloud masking for Sentinel-2
        def maskS2clouds(image):
            scl = image.select('SCL')
            mask = (
                scl.neq(3).And(scl.neq(8))
                .And(scl.neq(9)).And(scl.neq(10))
            )
            return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

        # Compute NDVI and EVI
        def compute_indices(img):
            ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI')
            evi = img.expression(
                '(2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1)))',
                {'NIR': img.select('B8'), 'RED': img.select('B4'), 'BLUE': img.select('B2')}
            ).rename('EVI')
            return img.addBands([ndvi,evi]).select(['NDVI','EVI'])
        days_to_keep = ee.List.sequence(1, 365, 6)
        
        # Helper to filter collection by the sequence
        def filter_by_stride(collection):
            return collection.filter(ee.Filter.dayOfYear(1, 365)).filter(
                ee.Filter.inList('system:time_start', 
                collection.filter(ee.Filter.listContains('day_list', days_to_keep)))
            )
            
        # A simpler way: Use a 'calendarRange' or just filter the existing collections
        # after they are created to pick every Nth image
        def limit_to_60(collection):
            list_col = collection.toList(collection.size())
            # Calculate the step (e.g., 175 / 60 ≈ 3)
            step = ee.Number(collection.size()).divide(60).ceil()
            indices = ee.List.sequence(0, collection.size().subtract(1), step)
            return ee.ImageCollection.fromImages(indices.map(lambda i: list_col.get(i)))
        # Get Sentinel-2 collection
        s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_str, end_str)
            .filterBounds(polygon)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .map(maskS2clouds)
            .map(compute_indices))
        s2_col = limit_to_60(s2_col)

        # Print collection sizes
        print("\nCollection sizes:")
        print(f"Sentinel-2 images: {s2_col.size().getInfo()}")
        
        # Compute RVI for Sentinel-1
        def compute_rvi(img):
            vv = ee.Image(10).pow(img.select('VV').divide(10))
            vh = ee.Image(10).pow(img.select('VH').divide(10))
            rvi = vh.multiply(4).divide(vv.add(vh)).rename('RVI')
            return img.addBands(rvi).select(['RVI'])

        # Get Sentinel-1 collection
        s1_col = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterDate(start_str, end_str)
            .filterBounds(polygon)
            .filter(ee.Filter.eq('instrumentMode','IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
            .filter(ee.Filter.eq('orbitProperties_pass','DESCENDING'))
            .map(compute_rvi)
        )
        print(f"Sentinel-1 images: {s1_col.size().getInfo()}")

        # Dynamic World bands
        DW_BANDS = ['water','trees','grass','flooded_vegetation',
             'crops','shrub_and_scrub','built','bare','snow_and_ice']

        # Get Dynamic World collection
        dw_col = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
            .filterDate(start_str, end_str)
            .filterBounds(polygon)
            .select(DW_BANDS))
        dw_col = limit_to_60(dw_col)
        print(f"Dynamic World images: {dw_col.size().getInfo()}")

        # Extract timeseries for given bands
        def extract_timeseries(collection, bands):
            def reducer(img):
                stats = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=polygon,
                    scale=10,
                    bestEffort=True,
                    maxPixels=1e13
                )
                props = {'date': img.date().format('YYYY-MM-dd')}
                for b in bands:
                    props[b] = stats.get(b)
                return ee.Feature(None, props)
            fc = collection.map(reducer).filter(ee.Filter.notNull(bands))
            info = fc.getInfo().get('features', [])
            rows = [f['properties'] for f in info]
            df = pd.DataFrame(rows)
            
            # Debug prints
            print(f"\nNumber of rows in DataFrame: {len(df)}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            if 'date' not in df.columns:
                print("Warning: 'date' column is missing!")
                print("Available columns:", df.columns.tolist())
                return pd.DataFrame()  # Return empty DataFrame if date is missing
            
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date')

        print(f"\nProcessing data for run {run_id}...")
        
        # Extract data from all collections (for predictions - full date range)
        print("Extracting Sentinel-2 data...")
        df_s2 = extract_timeseries(s2_col, ['NDVI','EVI'])
        print("Extracting Sentinel-1 data...")
        df_s1 = extract_timeseries(s1_col, ['RVI'])
        print("Extracting Dynamic World data...")
        df_dw = extract_timeseries(dw_col, DW_BANDS)
        
        # Calculate 6 months ago from end date for classification
        end_date = pd.to_datetime(end_str)
        twelve_months_ago = end_date - pd.DateOffset(months=12)
        classification_start = twelve_months_ago.strftime('%Y-%m-%d')
        
        print(f"\nFetching additional data for classification from {classification_start} to {end_str}")
        
        # Get separate collections for classification (last 6 months only)
        s2_col_class = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(classification_start, end_str)
            .filterBounds(polygon)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
            .map(maskS2clouds)
            .map(compute_indices)
        )
        
        s1_col_class = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterDate(classification_start, end_str)
            .filterBounds(polygon)
            .filter(ee.Filter.eq('instrumentMode','IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
            .filter(ee.Filter.eq('orbitProperties_pass','DESCENDING'))
            .map(compute_rvi)
        )
        
        dw_col_class = (
            ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
            .filterDate(classification_start, end_str)
            .filterBounds(polygon)
            .select(DW_BANDS)
        )
        
        # Extract classification data (last 6 months only)
        print("Extracting classification data (last 6 months)...")
        df_s2_class = extract_timeseries(s2_col_class, ['NDVI','EVI'])
        df_s1_class = extract_timeseries(s1_col_class, ['RVI'])
        df_dw_class = extract_timeseries(dw_col_class, DW_BANDS)
        
        # Debug prints for each DataFrame
        print("\nDataFrame shapes:")
        print(f"S2 shape: {df_s2.shape if not df_s2.empty else 'Empty'}")
        print(f"S1 shape: {df_s1.shape if not df_s1.empty else 'Empty'}")
        print(f"DW shape: {df_dw.shape if not df_dw.empty else 'Empty'}")
        
        # Only save if DataFrames are not empty
        if not df_s2.empty:
            df_s2.to_csv(os.path.join(data_dir, 's2_data.csv'), index=False)
            print("Saved S2 data")
        if not df_s1.empty:
            df_s1.to_csv(os.path.join(data_dir, 's1_data.csv'), index=False)
            print("Saved S1 data")
        if not df_dw.empty:
            df_dw.to_csv(os.path.join(data_dir, 'dw_data.csv'), index=False)
            print("Saved DW data")
        
        # Merge all dataframes - handle empty DataFrames properly
        if df_s2.empty and df_s1.empty and df_dw.empty:
            raise Exception("No data available for any of the collections")
        
        # Start with the first non-empty DataFrame
        df_all = None
        if not df_s2.empty:
            df_all = df_s2.copy()
        elif not df_s1.empty:
            df_all = df_s1.copy()
        elif not df_dw.empty:
            df_all = df_dw.copy()
        
        # Merge with other non-empty DataFrames
        if df_all is not None:
            if not df_s1.empty and df_s1 is not df_all:
                df_all = df_all.merge(df_s1, on='date', how='outer')
            if not df_dw.empty and df_dw is not df_all:
                df_all = df_all.merge(df_dw, on='date', how='outer')
            df_all = df_all.sort_values('date')
        else:
            raise Exception("No data available for any of the collections")
        
        # Add final_classification column based on max of crops, forest, grass
        for col in ['crops', 'trees', 'grass']:
            if col not in df_all.columns:
                df_all[col] = np.nan
        def classify_row(row):
            # Use all Dynamic World classes for comprehensive land cover classification
            all_dw_classes = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
            vals = {k: row.get(k, np.nan) for k in all_dw_classes}
            
            # Filter out NaN values
            valid_vals = {k: v for k, v in vals.items() if not pd.isna(v)}
            
            if not valid_vals:
                return np.nan
            
            # Return the class with the highest probability
            return max(valid_vals, key=valid_vals.get)
        # Create classification DataFrame from last 6 months data
        df_class_all = None
        if not df_s2_class.empty:
            df_class_all = df_s2_class.copy()
        elif not df_s1_class.empty:
            df_class_all = df_s1_class.copy()
        elif not df_dw_class.empty:
            df_class_all = df_dw_class.copy()
        
        # Merge classification DataFrames
        if df_class_all is not None:
            if not df_s1_class.empty and df_s1_class is not df_class_all:
                df_class_all = df_class_all.merge(df_s1_class, on='date', how='outer')
            if not df_dw_class.empty and df_dw_class is not df_class_all:
                df_class_all = df_class_all.merge(df_dw_class, on='date', how='outer')
            df_class_all = df_class_all.sort_values('date')
        
        # Add final_classification column for classification data
        if df_class_all is not None and not df_class_all.empty:
            # Ensure all Dynamic World classes are present in the DataFrame
            all_dw_classes = ['water', 'trees', 'grass', 'flooded_vegetation',
                            'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
            for col in all_dw_classes:
                if col not in df_class_all.columns:
                    df_class_all[col] = np.nan

            # ✅ Create final_classification if missing
            if 'final_classification' not in df_class_all.columns:
                df_class_all['final_classification'] = df_class_all.apply(classify_row, axis=1)

            # Initialize all DW classes with 0
            summary_dict = {}
            class_counts = df_class_all['final_classification'].value_counts(dropna=True)
            total = class_counts.sum()

            for cls in all_dw_classes:
                count = int(class_counts.get(cls, 0))
                percent = round((count / total * 100), 2) if total > 0 else 0.0
                summary_dict[cls] = {'count': count, 'percent': percent}

            print(f"\nClassification based on data from {classification_start} to {end_str}")
            print(f"Number of data points in last 12 months: {len(df_class_all)}")

        else:
            # Fallback to using all data if no classification data available
            print("Warning: No data available in last 12 months, using all available data for classification")

            all_dw_classes = ['water', 'trees', 'grass', 'flooded_vegetation',
                            'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
            for col in all_dw_classes:
                if col not in df_all.columns:
                    df_all[col] = np.nan

            # ✅ Create final_classification if missing
            if 'final_classification' not in df_all.columns:
                df_all['final_classification'] = df_all.apply(classify_row, axis=1)

            summary_dict = {}
            class_counts = df_all['final_classification'].value_counts(dropna=True)
            total = class_counts.sum()

            for cls in all_dw_classes:
                count = int(class_counts.get(cls, 0))
                percent = round((count / total * 100), 2) if total > 0 else 0.0
                summary_dict[cls] = {'count': count, 'percent': percent}

        
        print('Land type summary (based on last 12 months):', summary_dict)
        
        # Add debug info about the comprehensive classification
        if summary_dict:
            print(f"\n📊 Comprehensive Land Cover Classification Results:")
            print("=" * 55)
            for land_type, stats in sorted(summary_dict.items(), key=lambda x: x[1]['percent'], reverse=True):
                print(f"{land_type.replace('_', ' ').title():20s}: {stats['percent']:5.1f}% ({stats['count']:3d} data points)")
            print("=" * 55)
            print("ℹ️  Classification now includes all 9 Dynamic World land cover types")
        else:
            print("⚠️  No land cover classification data available")
        
        # Check if we have sufficient data for predictions
        min_data_points = 10  # Minimum number of data points needed for reliable predictions
        if len(df_all) < min_data_points:
            print(f"\n⚠️  WARNING: Insufficient data for predictions!")
            print(f"   - Available data points: {len(df_all)}")
            print(f"   - Minimum required: {min_data_points}")
            print(f"   - Date range selected: {start_str} to {end_str}")
            print(f"   - Recommendation: Increase the date range to get more data for agriculture activity analysis")
            
            # Create a warning flag in the summary
            summary_dict['_warning'] = {
                'insufficient_data': True,
                'available_points': len(df_all),
                'min_required': min_data_points,
                'recommendation': f"Increase date range from {start_str} to {end_str} to get more data for agriculture activity analysis"
            }
        
        # Save results (full dataset for predictions)
        output_file = os.path.join(data_dir, 'parcel_data.csv')
        df_all.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        

        # Calculate crop intensity (number of valid NDVI peaks)
        crop_intensity = 0
        if 'NDVI' in df_all.columns and 'date' in df_all.columns:
            crop_intensity = count_crop_cycles(df_all['NDVI'].values, df_all['date'].values)
        summary_dict['crop_intensity'] = crop_intensity
        print(f"Crop Intensity (number of valid NDVI peaks): {crop_intensity}")
        return df_all, summary_dict, df_dw
        
    except Exception as e:
        print(f"Error processing parcel data: {e}")
        return None

def create_test_data(data_dir, end_str=None):
    """
    Process all CSV files in the data directory and create a consolidated test dataset.
    
    Args:
    data_dir (str): Path to the directory containing the CSV files
    end_str (str): End date in YYYY-MM-DD format. Start date will be automatically set to 2 years before.
    """
    # Get all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith('parcel')]
    
    # Group files by label (everything before the last underscore)
    label_files = defaultdict(list)
    for file in csv_files:
        label = '_'.join(file.replace('.csv', '').split('_')[:-1])
        label_files[label].append(file)
    
    label_dfs = {}
    
    # Define features in the desired order for final output
    features = ['NDVI', 'EVI', 'RVI', 'crops']
    # Define all columns we need to read from CSV files
    all_columns = ['date', 'NDVI', 'EVI', 'RVI', 'crops', 'trees', 'grass']
    
    for label, files in label_files.items():
        dfs = []
        for file in files:
            file_path = os.path.join(data_dir, file)
            # Read all required columns
            df = pd.read_csv(file_path)
            # Ensure all required columns exist, fill with NaN if missing
            for col in all_columns:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[all_columns]  # Select only the columns we need
            df['date'] = pd.to_datetime(df['date'])
            dfs.append(df)
    
        # Concatenate all files of same label without aggregation
        combined_df = pd.concat(dfs, ignore_index=True)
    
        # Rename columns with label prefix
        combined_df.rename(columns={
            'NDVI': f'{label}_NDVI',
            'EVI': f'{label}_EVI',
            'RVI': f'{label}_RVI',
            'crops': f'{label}_crops',
            'trees': f'{label}_trees',
            'grass': f'{label}_grass'
        }, inplace=True)
    
        label_dfs[label] = combined_df
    
    # Merge all label DataFrames on 'date'
    all_dfs = list(label_dfs.values())
    consolidated_df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_dfs)
    
    # Sort by date
    consolidated_df['date'] = pd.to_datetime(consolidated_df['date'])
    consolidated_df.sort_values('date', inplace=True)
    
    # Filter by date range - automatically use 1 year before end date
    if end_str:
        end_date = pd.to_datetime(end_str)
        start_date = end_date - pd.DateOffset(years=1)
        start_str = start_date.strftime('%Y-%m-%d')
        temp_df = consolidated_df[(consolidated_df['date'] >= start_date) & (consolidated_df['date'] <= end_date)].copy()
        print(f"Filtered data from {start_str} to {end_str}: {len(temp_df)} rows")
    else:
        # Use all data if no end date provided
        temp_df = consolidated_df.copy()
        print(f"Using all available data: {len(temp_df)} rows")
    
    # Get all crops, trees, grass columns from all labels
    crops_cols = [col for col in temp_df.columns if col.endswith('_crops')]
    trees_cols = [col for col in temp_df.columns if col.endswith('_trees')]
    grass_cols = [col for col in temp_df.columns if col.endswith('_grass')]
    
    # Calculate max of crops, trees, grass for each row
    if crops_cols and trees_cols and grass_cols:
        temp_df['max_crops'] = temp_df[crops_cols].max(axis=1)
        temp_df['max_trees'] = temp_df[trees_cols].max(axis=1)
        temp_df['max_grass'] = temp_df[grass_cols].max(axis=1)
        temp_df['max_class'] = temp_df[['max_crops', 'max_trees', 'max_grass']].max(axis=1)
    else:
        # Fallback if some columns are missing
        temp_df['max_class'] = 0.0
    
    # Create the final test dataframe with features in the desired order
    test_df = pd.DataFrame()
    test_df['date'] = temp_df['date']  # Preserve the date column
    
    # Calculate mean values for each feature
    for feature in features:
        if feature == 'crops':
            # Use the max_class value for crops
            test_df[feature] = temp_df['max_class']
        else:
            # Calculate mean for other features
            feature_cols = [col for col in temp_df.columns if col.endswith(f'_{feature}')]
            if feature_cols:
                test_df[feature] = temp_df[feature_cols].mean(axis=1)
            else:
                test_df[feature] = np.nan
    
    # Set date as index for interpolation
    test_df.set_index('date', inplace=True)
    
    # Perform linear interpolation for all features
    test_df = test_df.interpolate(method='linear', limit_direction='both')
    
    # Reset index to get date back as a column
    test_df.reset_index(inplace=True)
    
    # Save interpolated test data
    test_data_path = os.path.join(data_dir, 'test_data.csv')
    test_df.to_csv(test_data_path, index=False)
    print(f"Interpolated test data saved to {test_data_path}")
    
    return test_df

# Location information retrieval function

from shapely.geometry import Polygon
import pandas as pd
import math, time, os, requests

# --- Haversine formula ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# --- Nearest water source finder using Overpass API ---
def find_nearest_water_source(lat, lon, radius=100000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})[natural=water];
      way(around:{radius},{lat},{lon})[natural=water];
      relation(around:{radius},{lat},{lon})[natural=water];

      node(around:{radius},{lat},{lon})[water=lake];
      way(around:{radius},{lat},{lon})[water=lake];
      relation(around:{radius},{lat},{lon})[water=lake];

      node(around:{radius},{lat},{lon})[water=pond];
      way(around:{radius},{lat},{lon})[water=pond];
      relation(around:{radius},{lat},{lon})[water=pond];

      node(around:{radius},{lat},{lon})[water=reservoir];
      way(around:{radius},{lat},{lon})[water=reservoir];
      relation(around:{radius},{lat},{lon})[water=reservoir];

      node(around:{radius},{lat},{lon})[waterway=river];
      way(around:{radius},{lat},{lon})[waterway=river];
      relation(around:{radius},{lat},{lon})[waterway=river];

      node(around:{radius},{lat},{lon})[waterway=stream];
      way(around:{radius},{lat},{lon})[waterway=stream];
      relation(around:{radius},{lat},{lon})[waterway=stream];
    );
    out center;
    """

    try:
        response = requests.get(overpass_url, params={'data': query}, timeout=30)
        
        # Check HTTP status first
        if response.status_code != 200:
            print(f"Overpass API HTTP error: {response.status_code}")
            return None, None, 0.0

        # Check if response body is empty
        if not response.text or response.text.strip() == "":
            print("Overpass API returned empty response")
            return None, None, 0.0

        data = response.json()
        elements = data.get("elements", [])
        if not elements:
            return None, None, 0.0

        nearest, min_dist = None, float("inf")

        for elem in elements:
            if "lat" in elem and "lon" in elem:
                lat_w, lon_w = elem["lat"], elem["lon"]
            elif "center" in elem:
                lat_w, lon_w = elem["center"]["lat"], elem["center"]["lon"]
            else:
                continue

            dist = haversine(lat, lon, lat_w, lon_w)
            if dist < min_dist:
                min_dist = dist
                nearest = {
                    "name": elem.get("tags", {}).get("name", "Unnamed water source"),
                    "coords": (lat_w, lon_w),
                    "distance_km": round(dist, 2)
                }

        if nearest:
            return nearest["name"], nearest["coords"], nearest["distance_km"]
        return None, None, 0.0

    except requests.exceptions.Timeout:
        print("Overpass API timed out")
        return None, None, 0.0
    except requests.exceptions.ConnectionError:
        print("Overpass API connection error")
        return None, None, 0.0
    except ValueError as e:
        print(f"Overpass API JSON parse error: {e} | Response: {response.text[:200]}")
        return None, None, 0.0
    except Exception as e:
        print(f"Overpass API unexpected error: {e}")
        return None, None, 0.0

# # --- Main function with Google + Mandi + Water ---
# def get_centroid_location(coords, mandi_csv="Mandi_Locations.csv"):
#     polygon = Polygon(coords)
#     if not polygon.is_valid:
#         return None, {"error": "Invalid polygon"}

#     # --- Centroid ---
#     centroid = polygon.centroid
#     lon, lat = centroid.x, centroid.y
#     api_key = "AIzaSyD32KadsacLx6ZRZfZHvsTVLsEh-NRmFIQ"  # 🔐 Replace with your actual API key

#     # --- Google Reverse Geocode ---
#     url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
#     try:
#         response = requests.get(url)
#         data = response.json()
#     except Exception as e:
#         return (lat, lon), {"latitude": lat, "longitude": lon, "error": str(e)}

#     if data.get("status") != "OK":
#         return (lat, lon), {"latitude": lat, "longitude": lon, "error": data.get("status", "Unknown error")}

#     result = data["results"][0] if data["results"] else {}
#     address_components = result.get("address_components", [])
#     formatted_address = result.get("formatted_address")

#     # --- Helper ---
#     def get_component(ctype):
#         for comp in address_components:
#             if ctype in comp["types"]:
#                 return comp["long_name"]
#         return None

#     # --- Nearest Mandi lookup ---
#     def find_nearest_mandi(lat_c, lon_c, csv_path=mandi_csv):
#         if not os.path.exists(csv_path):
#             return None, None
#         df = pd.read_csv(csv_path)
#         nearest_mandi, min_distance = None, float("inf")

#         for _, row in df.iterrows():
#             if row.get("Coordinates") in (None, "NA"):
#                 continue
#             try:
#                 lat_m, lon_m = map(float, row["Coordinates"].split(","))
#             except:
#                 continue
#             dist = haversine(lat_c, lon_c, lat_m, lon_m)
#             if dist < min_distance:
#                 min_distance, nearest_mandi = dist, row["Mandi_Name"]

#         return nearest_mandi, round(min_distance, 2) if min_distance != float("inf") else None

#     mandi, mandi_distance = find_nearest_mandi(lat, lon, mandi_csv)

#     # --- Nearest Water Source ---
#     water_name, water_coords, water_distance = find_nearest_water_source(lat, lon)

#     # --- Collect details ---
#     location_details = {
#         "latitude": lat,
#         "longitude": lon,
#         "house_number": get_component("street_number"),
#         "road": get_component("route"),
#         "neighbourhood": get_component("neighborhood"),
#         "sublocality": get_component("sublocality"),
#         "locality": get_component("locality"),
#         "district": get_component("administrative_area_level_2"),
#         "state": get_component("administrative_area_level_1"),
#         "postcode": get_component("postal_code"),
#         "country": get_component("country"),
#         "full_address": formatted_address,
#         # --- Nearest mandi ---
#         "Nearest Mandi": mandi,
#         "Mandi Distance (km)": mandi_distance,
#         # --- Nearest water ---
#         "Nearest Water Source": water_name if water_name else "Not found",
#         # "Water Source Coordinates": water_coords if water_coords else "Not found",
#         "Water Distance (km)": water_distance if water_distance else "Not found",
#     }

#     return (lat, lon), location_details


from shapely.geometry import Polygon
from geopy.geocoders import Nominatim
import pandas as pd
import os

# --- Nearest mandi lookup ---
def find_nearest_mandi(lat_c, lon_c, csv_path="Mandi_Locations.csv"):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    nearest_mandi, min_distance = None, float("inf")

    for _, row in df.iterrows():
        if row.get("Coordinates") in (None, "NA"):
            continue
        try:
            lat_m, lon_m = map(float, row["Coordinates"].split(","))
        except:
            continue
        dist = haversine(lat_c, lon_c, lat_m, lon_m)
        if dist < min_distance:
            min_distance, nearest_mandi = dist, row["Mandi_Name"]

    return nearest_mandi, round(min_distance, 2) if min_distance != float("inf") else None


# --- Main function using Geopy ---
def get_centroid_location(coords, mandi_csv="Mandi_Locations.csv"):
    polygon = Polygon(coords)
    if not polygon.is_valid:
        return None, {"error": "Invalid polygon"}

    # --- Centroid ---
    centroid = polygon.centroid
    lon, lat = centroid.x, centroid.y

    # --- Reverse geocode with geopy ---
    geolocator = Nominatim(user_agent="my_app")
    try:
        location = geolocator.reverse((lat, lon), language="en")
    except Exception as e:
        return (lat, lon), {"latitude": lat, "longitude": lon, "error": str(e)}

    if not location:
        return (lat, lon), {"latitude": lat, "longitude": lon, "error": "No result from Nominatim"}

    address = location.raw.get("address", {})
    formatted_address = location.address

    # --- Nearest Mandi ---
    mandi, mandi_distance = find_nearest_mandi(lat, lon, mandi_csv)

    # --- Nearest Water Source ---
    water_name, water_coords, water_distance = find_nearest_water_source(lat, lon)

    # --- Collect details ---
    location_details = {
        "latitude": lat,
        "longitude": lon,
        "house_number": address.get("house_number"),
        "road": address.get("road"),
        "neighbourhood": address.get("neighbourhood"),
        "suburb": address.get("suburb"),
        "village": address.get("village"),
        "hamlet": address.get("hamlet"),
        "locality": address.get("locality"),
        "town": address.get("town"),
        "city": address.get("city"),
        "municipality": address.get("municipality"),
        "district": address.get("district") or address.get("county"),
        "state_district": address.get("state_district"),
        "state": address.get("state"),
        "postcode": address.get("postcode"),
        "country": address.get("country"),
        # "country_code": address.get("country_code"),
        "full_address": location.address,
        # --- Nearest mandi ---
        "Nearest Mandi": mandi,
        "Mandi Distance (km)": mandi_distance,
        # --- Nearest water ---
        "Nearest Water Source": water_name if water_name else "Not found",
        "Water Distance (km)": water_distance if water_distance else "0.0",
    }

    return (lat, lon), location_details




import requests

def get_places_info(coords, radius=200000):
    """
    Get the nearest railway station and water source using Google Places API.
    
    Args:
        coords (tuple): (lat, lon)
        radius (int): Search radius in meters (default: 2,000,000)
    
    Returns:
        dict: Dictionary with nearest railway station and water source info
    """
    
    # 🔐 Replace with your actual Google Maps API key
    API_KEY = "AIzaSyBFrNDdn6wpjvVPeHa_aYsVYNPifp7MkF0"
    lat, lon = coords

    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    def search_place(keyword=None, type_filter=None):
        params = {
            'location': f'{lat},{lon}',
            'radius': radius,
            'key': API_KEY
        }
        if keyword:
            params['keyword'] = keyword
        if type_filter:
            params['type'] = type_filter

        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            result = data['results'][0]
            return {
                'name': result.get('name'),
                'address': result.get('vicinity'),
                'location': result.get('geometry', {}).get('location'),
                'rating': result.get('rating'),
                'place_id': result.get('place_id'),
                'types': result.get('types')
            }
        else:
            return None

    # Nearest railway station
    nearest_station = search_place(keyword='railway station', type_filter='transit_station')

   

    return {
        'nearest_railway_station': nearest_station,
    }






import ee
import pandas as pd
import os
import time
from datetime import datetime, timedelta




def initialize_ee():
    try:
        service_account = os.getenv("GEE_SERVICE_ACCOUNT")
        # This matches the 'Mount path' you set in the Cloud Run UI
        json_path = "/secrets/gee-key.json" 
        
        if os.path.exists(json_path) and service_account:
            # Use the mounted secret file for production authentication
            credentials = ee.ServiceAccountCredentials(service_account, json_path)
            ee.Initialize(credentials)
            print(f"✅ GEE Initialized via Secret Manager: {service_account}")
            
        elif service_account:
            # Fallback: Cloud Run internal metadata (only if not using a key file)
            credentials = ee.ServiceAccountCredentials(service_account, key_data=None)
            ee.Initialize(credentials)
            print(f"✅ GEE Initialized via Metadata Server: {service_account}")
            
        else:
            # Local development fallback
            ee.Initialize()
            print("✅ GEE Initialized via local default credentials")
            
    except Exception as e:
        # Don't let this crash the app, or you'll get the 'Port 8080' error again
        print(f"❌ GEE Initialization Failed: {e}")

initialize_ee()

def make_parcel_from_centroid(coords, buffer_km=5):
    """
    Take polygon coords, compute centroid, 
    then build a square buffer around it suitable for ERA5 (~10km).
    
    Args:
        coords (list): Polygon coordinates [[lon, lat], ...].
        buffer_km (int): Half-size of square buffer in km (default 5 → 10km box).
    
    Returns:
        ee.Geometry.Polygon
    """
    # Ensure polygon is closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    
    polygon = ee.Geometry.Polygon([coords])
    
    # Get centroid
    centroid = polygon.centroid()
    
    # Make a square buffer around centroid
    buffer_m = buffer_km * 1000
    parcel = centroid.buffer(buffer_m).bounds()  # square parcel
    
    return parcel


def get_daily_weather_data_optimized(start_date, end_date, geometry):
    """
    Ultra-fast function using ERA5-Land Daily Aggregated dataset.
    Works for any geometry passed as input.
    
    Parameters:
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
        geometry (ee.Geometry): Region of interest
    """
    
    print(f"Loading daily weather data from {start_date} to {end_date}...")
    
    # Use ERA5-Land Daily Aggregated dataset (daily resolution)
    daily_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry) \
        .select([
            'total_precipitation_sum',   # Daily total precipitation 
            'temperature_2m_max',        # Daily maximum temperature
            'temperature_2m_min'         # Daily minimum temperature
        ])
    
    print(f"Found {daily_collection.size().getInfo()} daily images")
    
    results = []
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_date = start_dt
    batch_size = 60  # Process 60 days at a time
    
    while current_date < end_dt:
        batch_end = min(current_date + timedelta(days=batch_size), end_dt)
        batch_start_str = current_date.strftime("%Y-%m-%d")
        batch_end_str = batch_end.strftime("%Y-%m-%d")
        
        print(f"Processing batch: {batch_start_str} to {batch_end_str}")
        
        try:
            batch_collection = daily_collection.filterDate(batch_start_str, batch_end_str)
            batch_size_actual = batch_collection.size().getInfo()
            
            if batch_size_actual == 0:
                print(f"  No data available for this batch")
                fill_date = current_date
                while fill_date < batch_end:
                    results.append({
                        "Date": fill_date.strftime("%Y-%m-%d"),
                        "Rainfall_mm": 0,
                        "Max_temp_celsius": 0,
                        "Min_temp_celsius": 0,
                    })
                    fill_date += timedelta(days=1)
            else:
                # Process each image
                def process_image(img):
                    stats = img.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=11132,  # ERA5-Land native resolution
                        maxPixels=1e13,
                        bestEffort=True
                    )
                    return ee.Feature(None, {
                        'date': img.get('system:time_start'),
                        'rainfall': stats.get('total_precipitation_sum'),
                        'max_temp': stats.get('temperature_2m_max'),
                        'min_temp': stats.get('temperature_2m_min')
                    })
                
                features = batch_collection.map(process_image)
                batch_results = features.getInfo()
                
                for feature in batch_results['features']:
                    props = feature['properties']
                    date_obj = datetime.utcfromtimestamp(props['date']/1000)
                    date_str = date_obj.strftime("%Y-%m-%d")
                    
                    rainfall_m = props.get('rainfall')
                    max_temp_k = props.get('max_temp')
                    min_temp_k = props.get('min_temp')
                    
                    rainfall_mm = rainfall_m * 1000 if rainfall_m is not None else 0
                    max_temp_c = max_temp_k - 273.15 if max_temp_k is not None else 0
                    min_temp_c = min_temp_k - 273.15 if min_temp_k is not None else 0
                    
                    results.append({
                        "Date": date_str,
                        "Rainfall_mm": round(rainfall_mm, 3) if rainfall_mm != 0 else 0,
                        "Max_temp_celsius": round(max_temp_c, 2) if max_temp_c != 0 else 0,
                        "Min_temp_celsius": round(min_temp_c, 2) if min_temp_c != 0 else 0,
                    })
                
                print(f"  ✓ Processed {len(batch_results['features'])} days")
                
        except Exception as e:
            print(f"  [Error] Batch {batch_start_str} to {batch_end_str}: {str(e)}")
            fill_date = current_date
            while fill_date < batch_end:
                results.append({
                    "Date": fill_date.strftime("%Y-%m-%d"),
                    "Rainfall_mm": 0,
                    "Max_temp_celsius": 0,
                    "Min_temp_celsius": 0,
                })
                fill_date += timedelta(days=1)
        
        current_date = batch_end
        time.sleep(0.5)  # avoid rate-limiting
    
    return results


def fill_missing_dates(results, start_date, end_date):
    """ Ensure all dates in range are present, filling missing ones with -9999 """
    results_dict = {item['Date']: item for item in results}
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    complete_results = []
    current_date = start_dt
    while current_date < end_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str in results_dict:
            complete_results.append(results_dict[date_str])
        else:
            complete_results.append({
                "Date": date_str,
                "Rainfall_mm": 0,
                "Max_temp_celsius": 0,
                "Min_temp_celsius": 0,
            })
        current_date += timedelta(days=1)
    return complete_results


def get_one_year_weather_data(geo, end_date_str):
    geometry = make_parcel_from_centroid(geo)
    """Fetch daily weather data for 1 year ending at given end_date."""
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    start_date = end_date - timedelta(days=365)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"Daily Weather Data Extraction")
    print(f"Date range: {start_date_str} → {end_date_str}")
    print(f"{'='*60}\n")

    # Extract
    results = get_daily_weather_data_optimized(start_date_str, end_date_str, geometry)
    complete_results = fill_missing_dates(results, start_date_str, end_date_str)

    # To DataFrame
    df = pd.DataFrame(complete_results)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df


def get_five_year_weather_data(geo, end_date_str):
    geometry = make_parcel_from_centroid(geo)

    # Define 5-year range (ending 1 year before end_date like your current logic)
    end_date = (datetime.strptime(end_date_str, "%Y-%m-%d")- timedelta(days=365)).replace(day=1)
    start_date = (end_date - timedelta(days=5*365)).replace(day=1)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"Monthly Weather Data Extraction (ERA5 Monthly)")
    print(f"Date range: {start_date_str} → {end_date_str}")
    print(f"{'='*60}\n")

    # ERA5 Monthly dataset
    monthly_collection = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
        .filterDate(start_date_str, end_date_str) \
        .filterBounds(geometry) \
        .select([
            'temperature_2m_min',  # Monthly min
            'temperature_2m_max',  # Monthly max
            'total_precipitation_sum'          # Monthly total precip
        ])

    print("DEBUG: Monthly collection size:", monthly_collection.size().getInfo())

    results = []

    def process_image(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=27830,  # native ERA5 monthly resolution (~30km)
            maxPixels=1e13,
            bestEffort=True
        )
        return ee.Feature(None, {
            'date': img.date().format('YYYY-MM'),
            'min_temp': stats.get('temperature_2m_min'),
            'max_temp': stats.get('temperature_2m_max'),
            'precip': stats.get('total_precipitation_sum')
        })

    features = monthly_collection.map(process_image).getInfo()
    print("DEBUG: Number of features fetched:", len(features.get("features", [])))

    for idx, f in enumerate(features.get('features', [])):
        props = f.get('properties', {})
        print(f"DEBUG: Feature {idx} properties:", props)

        month = props.get('date')
        min_temp = props.get('min_temp')
        max_temp = props.get('max_temp')
        precip = props.get('precip')

        results.append({
            "Month": month,
            "Min_temp_celsius": round(min_temp - 273.15, 2) if min_temp is not None else 0,
            "Max_temp_celsius": round(max_temp - 273.15, 2) if max_temp is not None else 0,
            "Precip_mm": round(precip * 1000, 2) if precip is not None else 0
        })
        print(f"Fetched data for month: {month}")

    print("DEBUG: Results collected:", len(results))

    df = pd.DataFrame(results)
    print("DEBUG: DataFrame columns:", df.columns.tolist())
    print("DEBUG: DataFrame head:\n", df.head())

    if "Month" not in df.columns:
        print("ERROR: 'Month' column missing in DataFrame!")

    df = df.sort_values('Month').reset_index(drop=True)

    return df




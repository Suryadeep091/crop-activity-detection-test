import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import io
import base64
import requests
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta
from google.auth import impersonated_credentials
from google.cloud import storage
import google.auth
import pickle
import numpy as np
import traceback
from analytics_engine import (
    apply_empirical_logic,
    summarize_crop_activity_table_data,
    summarize_indices_for_table,
    run_full_analytics_pipeline,
    compute_analytics_from_extracted,
    impute_missing_ndvi_using_ensemble,
    prepare_daily_modeling_dataframe,
    build_vegetation_indices_chart,
    compute_parcel_certainty,
    has_noncrop_dominance_veto,
    ACTIVE_ACTIVITY_THRESHOLD,
)
from data_loader import get_centroid_location, get_places_info
from rain_temp import get_one_year_weather_data, get_five_year_weather_data
from pdf_generator import generate_intelligence_report 
from location import get_static_map_b64
from data_loader import detect_crop_cycles

# Configuration
matplotlib.use('Agg')
app = FastAPI(title="TerraDrishti Unified Khasra Engine")
executor = ThreadPoolExecutor(max_workers=20)

# --- DATA LOADING ---
# Load CSV once at startup for speed


class GeometryRequest(BaseModel):
    task_id: str 
    kml_coordinates: str 
    end_date: str = datetime.now().strftime("%Y-%m-%d")

def parse_kml_string(kml_str: str):
    """Parses KML <coordinates> string into [[lon, lat], ...]"""
    points = kml_str.strip().split()
    coords = []
    for p in points:
        parts = p.split(',')
        if len(parts) >= 2:
            coords.append([float(parts[0]), float(parts[1])])
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords
# --- CORE UTILITIES ---

# def fetch_parcel_geojson(guid, state):
#     salt_key = "PAe17K1Rvfeij21TQPlq"
   
#     url = f"https://test-client.quantasip.com/api/parcelData?saltKey={salt_key}&guid={guid}&state={state}"
#     try:
#         response = requests.get(url, timeout=15)
#         if response.status_code == 500:
#              # Custom log for your Cloud Run console
#              print(f"EXTERNAL API CRASH for GUID: {guid}") 
#              return None
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         print(f"Network Error: {e}")
#         return None



def upload_private_to_gcs(data, destination_blob_name, content_type, is_file=False):
    bucket_name = "terradrishti"
    source_creds, project_id = google.auth.default()
    
    # Impersonation for V4 Signing
    target_principal = "413500342905-compute@developer.gserviceaccount.com"
    creds = impersonated_credentials.Credentials(
        source_credentials=source_creds,
        target_principal=target_principal,
        target_scopes=["https://www.googleapis.com/auth/devstorage.read_write"],
        lifetime=3600
    )
    
    client = storage.Client(credentials=creds, project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    if is_file:
        blob.upload_from_filename(data, content_type=content_type)
    else:
        blob.upload_from_string(data, content_type=content_type)
    
    # 7-Day Private Link
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(days=7),
        method="GET"
    )

def download_from_gcs(blob_name, bucket_name="terradrishti"):
    """
    Downloads a blob from GCS and returns it as bytes.
    Matches the pattern of your existing upload function.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            print(f"⚠️ Blob not found: {blob_name}")
            return None
            
        # Download the content as bytes
        data = blob.download_as_bytes()
        return data
    except Exception as e:
        print(f"❌ GCS Download Error for {blob_name}: {e}")
        return None
    
# --- WORKERS (Same as before) ---
def satellite_extraction_worker(task_id, coords, end_date):
    from data_loader import process_parcel_data
    return process_parcel_data(task_id, coords, end_date)

def location_worker(task_id, coords):
    """Fetches Address, Water Sources, and the Static Map Image."""
    # 1. Get standard metadata
    centroid, location_details = get_centroid_location(coords)
    places_info = get_places_info(centroid)
    location_details.update(places_info)
    
    # 2. Extract Khasra No from task_id (e.g., KH_123_timestamp -> 123)
    # Or pass it explicitly if available
    khasra_label = task_id.split('_')[1] if '_' in task_id else "N/A"
    
    # 3. Generate the Static Map Image as Base64
    google_api_key = "AIzaSyBFrNDdn6wpjvVPeHa_aYsVYNPifp7MkF0" # Use your real key
    map_b64 = get_static_map_b64(coords, khasra_label, google_api_key)
    
    # 4. Return combined dictionary
    location_details["map_image_b64"] = map_b64
    location_details["coordinates"] = [{"lat": lat, "lon": lon} for lon, lat in coords] # Convert back to lat/lon dicts
    return location_details

def fig_to_base64(fig, is_plotly=False):
    """Helper to convert plots to base64 strings without saving to disk."""
    buf = io.BytesIO()
    if is_plotly:
        # Scale 2 for high-quality PDF resolution
        pio.write_image(fig, buf, format='png', scale=2)
    else:
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def weather_worker(coords, end_date):
    df_daily = get_one_year_weather_data(coords, end_date)
    df_monthly = get_five_year_weather_data(coords, end_date)

    analysis_end = datetime.strptime(end_date, "%Y-%m-%d")
    analysis_start = analysis_end - pd.DateOffset(years=1)

    # --- PLOT 1: 1-Year Rainfall (Daily) ---
    # --- PLOT 1: 1-Year Rainfall (Daily) ---
    fig_rain_1y = go.Figure()
    fig_rain_1y.add_trace(go.Scatter(
        x=df_daily['Date'], 
        y=df_daily['Rainfall_mm'],
        fill='tozeroy',
        mode='lines',
        line=dict(color='#2563eb', width=1.5),
        fillcolor='rgba(59, 130, 246, 0.3)',
        name="Rainfall (mm)"
    ))
    fig_rain_1y.update_layout(
        
        xaxis=dict(type="date", range=[
                    analysis_start.strftime("%Y-%m-%d"),
                    analysis_end.strftime("%Y-%m-%d"),
                ], tickformat="%b %Y"), # Consistent 1-year window
        yaxis=dict(title="Rainfall (mm)"),
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=200,
        showlegend=False # Trend line is self-explanatory
    )
    rain_1y = fig_to_base64(fig_rain_1y, is_plotly=True)

    # --- PLOT 2: 1-Year Temperature (Monthly Aggregated from Daily) ---
    # --- PLOT 2: 1-Year Temperature (Monthly) ---
    df_daily['Month_DT'] = pd.to_datetime(df_daily['Date']).dt.to_period('M').dt.to_timestamp()
    df_1y_temp = df_daily.groupby('Month_DT').agg({'Max_temp_celsius':'mean', 'Min_temp_celsius':'mean'}).reset_index()

    fig_temp_1y = go.Figure()
    fig_temp_1y.add_trace(go.Scatter(x=df_1y_temp['Month_DT'], y=df_1y_temp['Max_temp_celsius'], name='Max', line=dict(color='#ef4444', width=2), marker=dict(size=6)))
    fig_temp_1y.add_trace(go.Scatter(x=df_1y_temp['Month_DT'], y=df_1y_temp['Min_temp_celsius'], name='Min', line=dict(color='#3b82f6', width=2), marker=dict(size=6)))

    fig_temp_1y.update_layout(
        
        xaxis=dict(type="date", range=[
                    analysis_start.strftime("%Y-%m-%d"),
                    analysis_end.strftime("%Y-%m-%d"),
                ], tickformat="%b %Y"),
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.6)"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=200
    )
    temp_1y = fig_to_base64(fig_temp_1y, is_plotly=True)

    # --- PLOT 3: 5-Year Rainfall (Monthly) ---
    # --- PLOT 3: 5-Year Rainfall (Monthly) ---
    fig_rain_5y = go.Figure()
    fig_rain_5y.add_trace(go.Bar(
        x=df_monthly['Month'], 
        y=df_monthly['Precip_mm'],
        marker_color='#60a5fa',
        opacity=0.8,
        name="Precipitation"
    ))
    fig_rain_5y.update_layout(
       
        xaxis=dict(type="date", tickformat="%Y"), # Focus on years for 5-year view
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=200,
        showlegend=False
    )
    rain_5y = fig_to_base64(fig_rain_5y, is_plotly=True)

    # --- PLOT 4: 5-Year Temperature (Monthly) ---
    # --- PLOT 4: 5-Year Temperature (Monthly) ---
    fig_temp_5y = go.Figure()
    fig_temp_5y.add_trace(go.Scatter(x=df_monthly['Month'], y=df_monthly['Max_temp_celsius'], name='Max', line=dict(color='#b91c1c', width=1.5)))
    fig_temp_5y.add_trace(go.Scatter(x=df_monthly['Month'], y=df_monthly['Min_temp_celsius'], name='Min', line=dict(color='#1d4ed8', width=1.5)))

    fig_temp_5y.update_layout(
      
        xaxis=dict(type="date", tickformat="%Y"),
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.6)"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=200
    )
    temp_5y = fig_to_base64(fig_temp_5y, is_plotly=True)

    return {
        "rain_1y_b64": rain_1y,
        "temp_1y_b64": temp_1y,
        "rain_5y_b64": rain_5y,
        "temp_5y_b64": temp_5y,
        "daily_weather_data": df_daily.to_dict(orient="records"),
        "monthly_weather_data": df_monthly.to_dict(orient="records")
    }


# --- UNIFIED ENDPOINT ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(PROJECT_DIR, "no_lim_S2_analysis")
os.makedirs(PICKLE_DIR, exist_ok=True)

@app.post("/test/accuracy")
async def test_accuracy_by_geometry(request: GeometryRequest):
    loop = asyncio.get_event_loop()
    coords = parse_kml_string(request.kml_coordinates)
    task_id = request.task_id

    try:
        # Step 1: Execute Extraction Workers
        b1 = loop.run_in_executor(executor, satellite_extraction_worker, task_id, coords, request.end_date)
        b2 = loop.run_in_executor(executor, location_worker, task_id, coords)
        b3 = loop.run_in_executor(executor, weather_worker, coords, request.end_date)

        extraction_res, loc_res, weather_res = await asyncio.gather(b1, b2, b3)
        if extraction_res is None:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Satellite extraction failed before producing a result. "
                    "Check the Cloud Run logs above this request for the GEE extraction error."
                ),
            )

        # Convert daily weather data back to DataFrame for ensemble imputation
        df_weather = pd.DataFrame(weather_res["daily_weather_data"])
        
        # Run local computations of the analytics pipeline (deferred computation)
        sat_res = compute_analytics_from_extracted(task_id, coords, request.end_date, extraction_res, df_weather)
        if sat_res is None:
            raise HTTPException(
                status_code=502,
                detail="Satellite analytics computation failed."
            )

        # Step 2: Extract ONLY raw signals for future model testing
        # We ignore 'prediction_stats' and 'seasonal_activity' here
        # Step 2: Extract ALL signals (including images) for exact replay
        raw_signals_payload = {
            "task_id": task_id,
            "coords": coords,
            "vegetation_indices": sat_res.get("timeseries_data", {}).get("vegetation_indices"),
            "raw_vegetation_indices": sat_res.get("timeseries_data", {}).get("raw_vegetation_indices"),
            "land_cover_probs": sat_res.get("timeseries_data", {}).get("land_cover_probs"),
            "location_data": loc_res,
            "weather_data": weather_res, # Save the full worker result (includes b64 strings)
            "images": sat_res.get("images", {}), # CRITICAL: Save the chart strings
            "seasonal_activity": sat_res.get("seasonal_activity", {}),
            "peak_analysis": sat_res.get("vegetation_peak_analysis", [])
        }
        
        # Step 3: Save to Local Project Directory
        pickle_bytes = pickle.dumps(raw_signals_payload)

        # 3. Upload to GCS (using your existing function)
        pickle_url = upload_private_to_gcs(
            data=pickle_bytes, 
            destination_blob_name=f"no_lim_S2_analysis/{task_id}_full_2023.pkl", 
            content_type="application/octet-stream", # Standard for binary files
            is_file=False
        )
            
        print(f"DEBUG: Saved test data to {pickle_url}")

        # Step 4: Generate PDF using current model (for side-by-side comparison)
        full_data = {
            "task_id": task_id,
            "satellite_analytics": sat_res,
            "location_details": loc_res,
            "map_details": loc_res,
            "weather_data": weather_res,
            "metadata": {"timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")}
        }
        
        local_pdf_path = await generate_intelligence_report(full_data)
        report_url = upload_private_to_gcs(local_pdf_path, f"Model_Test/test_{task_id}.pdf", "application/pdf", is_file=True)

        # Cleanup ephemeral PDF
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)

        stats = sat_res.get("crop_activity_prediction_stats", {})
        total_instances = stats.get("total", 1) # Default to 1 to avoid division by zero
        active_instances = stats.get("crop_days", 0)
        land_cover_probs = sat_res.get("timeseries_data", {}).get("land_cover_probs", [])
        
        # 2. Calculate the Activity Percentage
        activity_ratio = (active_instances / total_instances) * 100
        noncrop_veto, veto_info = has_noncrop_dominance_veto(land_cover_probs)
        
        # 3. Apply parcel-level activity threshold and non-crop dominance guard.
        if activity_ratio > ACTIVE_ACTIVITY_THRESHOLD and not noncrop_veto:
            agri_verdict = "Agri activity detected"
            is_active = True
        else:
            agri_verdict = "Low Agri activity detected"
            is_active = False

        # --- FINAL RETURN ---
        return {
            "status": "success",
            "task_id": task_id,
            "verdict": agri_verdict,
            "agri_activity": agri_verdict,
            "activity_score": f"{round(activity_ratio, 2)}%",
            "is_active": is_active,
            "noncrop_dominance_veto": noncrop_veto,
            "final_confidence_score": f"{round(stats.get('certainty_score', stats.get('overall_confidence', 0)), 2)}%",
            "certainty_tier": stats.get("certainty_tier", ""),
            "report_url": report_url,
            "local_pickle_path": pickle_url,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Test Pipeline Error: {str(e)}")
    
@app.post("/test/replay/{task_id}")
async def replay_test_from_pickle(task_id: str):
    try:
        # 1. Fetch and Load Pickle
        pickle_bytes = download_from_gcs(f"no_lim_S2_analysis/{task_id}_full_2024.pkl")
        if not pickle_bytes:
            raise HTTPException(status_code=404, detail="Pickle not found")
        raw_data = pickle.loads(pickle_bytes)

        # 2. Extract Raw Signals from Pickle
        veg_indices = raw_data.get("vegetation_indices") or [] 
        raw_veg_indices = raw_data.get("raw_vegetation_indices") or []
        lc_probs = raw_data.get("land_cover_probs") or []
        loc_res = raw_data.get("location_data") or {}

        # 3. Reconstruct DataFrame for Processing
        df_veg = pd.DataFrame(veg_indices)
        df_raw_veg = pd.DataFrame(raw_veg_indices)
        df_dw = pd.DataFrame(lc_probs)
        
        # Normalize and merge (Mirroring your analytics_engine pipeline)
        df_veg['date'] = pd.to_datetime(df_veg['date']).dt.normalize()
        if not df_raw_veg.empty and 'date' in df_raw_veg.columns:
            df_raw_veg['date'] = pd.to_datetime(df_raw_veg['date']).dt.normalize()
        df_dw['date'] = pd.to_datetime(df_dw['date']).dt.normalize()
        ############
        # 1. Define all Dynamic World classes
        all_dw_classes = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']

        # 2. Determine the dominant class for every single timestamp in the pickle data
        def get_dominant_class(row):
            # Filter only the valid DW class columns
            valid_vals = {k: row[k] for k in all_dw_classes if k in row and not pd.isna(row[k])}
            if not valid_vals: return "bare" # Fallback
            return max(valid_vals, key=valid_vals.get)

        # Apply classification to the dataframe reconstructed from pickle
        df_dw['final_classification'] = df_dw.apply(get_dominant_class, axis=1)

        # 3. Calculate counts and percentages for the table
        summary_dict = {}
        class_counts = df_dw['final_classification'].value_counts()
        total_points = len(df_dw)

        for cls in all_dw_classes:
            count = int(class_counts.get(cls, 0))
            summary_dict[cls] = {
                "count": count,
                "percent": round((count / total_points * 100), 2) if total_points > 0 else 0.0
            }
        
        if not df_raw_veg.empty and 'date' in df_raw_veg.columns:
            # Run ensemble imputation on raw data during replay
            coords = raw_data.get("coords") or []
            end_date_val = pd.to_datetime(df_raw_veg['date']).max().strftime('%Y-%m-%d')
            df_weather_raw = pd.DataFrame(raw_data.get("weather_data", {}).get("daily_weather_data", []))
            
            df_raw_veg_updated, df_imputed = impute_missing_ndvi_using_ensemble(
                df_raw_veg, df_dw, coords, end_date_val, df_weather=df_weather_raw
            )
            veg_source = df_raw_veg_updated
            already_smoothed = False
        elif not df_veg.empty and 'date' in df_veg.columns:
            veg_source = df_veg
            already_smoothed = True
        else:
            raise HTTPException(status_code=400, detail="Pickle has no vegetation index data")

        dataset_df, raw_points_df = prepare_daily_modeling_dataframe(
            veg_source, df_dw, already_smoothed=already_smoothed
        )
        if already_smoothed and df_raw_veg.empty:
            raw_points_df = pd.DataFrame(columns=['date', 'NDVI', 'EVI', 'RVI'])

        # 4. RUN YOUR ENGINE LOGIC
        # We no longer need aggressive rolling windows because Whittaker has effectively smoothed it perfectly
        dataset_df['NDVI_smooth_slope'] = dataset_df['NDVI']
        dataset_df['RVI_smooth_slope'] = dataset_df['RVI']
        
        # Pre-calculate slopes (Daily points = fine-grained derivatives)
        dataset_df['NDVI_slope'] = np.gradient(dataset_df['NDVI_smooth_slope'])
        dataset_df['RVI_slope'] = np.gradient(dataset_df['RVI_smooth_slope'])
        
        cycle_info = detect_crop_cycles(dataset_df)
        
        results = dataset_df.apply(lambda row: apply_empirical_logic(row, cycle_info["detected_seasons"]), axis=1)
        dataset_df['prediction'] = [r[0] for r in results]
        dataset_df['p1_crop_conf'] = [r[1] for r in results]
        dataset_df['p2_crop_conf'] = [r[2] for r in results]
        dataset_df['final_confidence'] = [r[3] for r in results]
        dataset_df['p1_nocrop_conf'] = 100 - dataset_df['p1_crop_conf']
        dataset_df['p2_nocrop_conf'] = 100 - dataset_df['p2_crop_conf']
        
        # YEAR-LONG ENVIRONMENTAL GUARDBAND
        dominant_classes = dataset_df[['trees', 'water', 'built', 'shrub_and_scrub', 'grass', 'crops', 'flooded_vegetation', 'bare', 'snow_and_ice']].idxmax(axis=1)
        tree_freq = (dominant_classes == 'trees').mean()
        water_freq = (dominant_classes == 'water').mean()
        built_freq = (dominant_classes == 'built').mean()
        snow_freq = (dominant_classes == 'snow_and_ice').mean()
        flooded_freq = (dominant_classes == 'flooded_vegetation').mean()
        shrub_freq = (dominant_classes == 'shrub_and_scrub').mean()
        crop_freq = (dominant_classes == 'crops').mean() + (dominant_classes == 'flooded_vegetation').mean()
        
        non_crop_freq = max(tree_freq, water_freq, built_freq, snow_freq, flooded_freq, shrub_freq)
        crop_days = int((dataset_df['prediction'] == "Crop-Activity").sum())
        total_days = len(dataset_df)
        activity_ratio = (crop_days / total_days) if total_days > 0 else 0

        # Absolute Veto logic triggers - Occams Razor: Protect cycles from AI blindness
        total_cycles = cycle_info['total_cycles']
        is_guardband_triggered = (tree_freq > 0.60 or water_freq > 0.60 or built_freq > 0.50 or snow_freq > 0.60 or flooded_freq > 0.60)
        if total_cycles == 0:
            is_guardband_triggered = is_guardband_triggered or (crop_freq < 0.10)
        
        # 1. Calculate the Average BioScore for the parcel (if cycles exist)
        avg_bioscore = np.mean([c['confidence'] for c in cycle_info['details']]) if total_cycles > 0 else 0

        # 2. Define the Tree Trap
        # Logic: If trees are the majority AND crops are scarce AND biological sync is mediocre
        is_tree_trap = (tree_freq > 0.50) and (crop_freq < 0.35) and (avg_bioscore < 70)
        is_shrub_trap = (shrub_freq > 0.55) and (crop_freq < 0.20)
        guardband_conflict = False

        # 3. Guardband & Penalty logic (Scenario A/B/C)
        if is_guardband_triggered:
            if non_crop_freq > 0.85 and activity_ratio > 0.0:
                # Scenario C: Extreme AI Domination
                guardband_conflict = True
                dataset_df['p2_crop_conf'] *= 0.10
                if total_cycles == 0: dataset_df['p1_crop_conf'] *= 0.10
            elif activity_ratio > 0.50 and non_crop_freq > 0.50:
                # Scenario B: Major Conflict
                guardband_conflict = True
                dataset_df['p2_crop_conf'] *= 0.30
                if total_cycles == 0: dataset_df['p1_crop_conf'] *= 0.30
            else:
                # Scenario A: Total Agreement (Standard Lockout)
                dataset_df['p2_crop_conf'] = 0.0
                if total_cycles == 0: dataset_df['p1_crop_conf'] = 0.0

            # Re-evaluate row-level confidence dynamically for penalized rows
            dataset_df['p1_nocrop_conf'] = 100.0 - dataset_df['p1_crop_conf']
            dataset_df['p2_nocrop_conf'] = 100.0 - dataset_df['p2_crop_conf']
            
            # Recalculate row predictions 
            avg_crop = (dataset_df['p1_crop_conf'] * 0.60) + (dataset_df['p2_crop_conf'] * 0.40)
            avg_nocrop = (dataset_df['p1_nocrop_conf'] * 0.60) + (dataset_df['p2_nocrop_conf'] * 0.40)
            
            # 1. Save FNs safely: Modified with the Tree Trap block (Prediction untouched)
            dataset_df['prediction'] = np.where(
                (dataset_df['p1_crop_conf'] > 50) & 
                (total_cycles > 0) & 
                (crop_freq > 0.07) & 
                (~is_tree_trap) & (~is_shrub_trap),
                "Crop-Activity",
                np.where(avg_crop > avg_nocrop, "Crop-Activity", "No Crop-Activity")
            )
            
            # 2. Confidence uses Geometric Purity
            p1_winning = np.where(dataset_df['prediction'] == "Crop-Activity", dataset_df['p1_crop_conf'], dataset_df['p1_nocrop_conf'])
            p2_winning = np.where(dataset_df['prediction'] == "Crop-Activity", dataset_df['p2_crop_conf'], dataset_df['p2_nocrop_conf'])
            
            base_confidence = np.sqrt(p1_winning * p2_winning)
            
            noise_cols = [c for c in ['trees', 'water', 'built', 'shrub_and_scrub'] if c in dataset_df.columns]
            noise_level = dataset_df[noise_cols].sum(axis=1) if noise_cols else 0.0
            purity = np.where(dataset_df['prediction'] == "Crop-Activity", np.clip(1.0 - (noise_level ** 2), 0.1, 1.0), 1.0)
            
            raw_conf = base_confidence * purity
            
            # Smoothstep stretching to push Highs higher and Lows lower
            nx = np.clip((raw_conf / 100.0) / 0.85, 0.0, 1.0)
            dataset_df['final_confidence'] = (nx * nx * (3.0 - 2.0 * nx)) * 100.0

        
        # 5. GENERATE SUMMARY OBJECTS (Using your helpers)
        analysis_end = dataset_df['date'].max()
        analysis_start = analysis_end - pd.DateOffset(years=1)
        
        # Get the Rabi/Kharif grid
        seasonal_act = summarize_crop_activity_table_data(dataset_df, analysis_start, analysis_end)
        
        # Get Peak Index Summary
        temp_df = dataset_df.set_index('date')
        peak_analysis, missing_periods = summarize_indices_for_table(temp_df)

        # 6. ASSEMBLE FINAL DATA FOR PDF
        crop_days = int((dataset_df['prediction']=="Crop-Activity").sum())
        total_days = len(dataset_df)
        # crop_days = total_days - no_crop_days
        
        activity_ratio = (crop_days / total_days * 100) if total_days > 0 else 0
        noncrop_veto, veto_info = has_noncrop_dominance_veto(df_dw)
        is_active = activity_ratio > ACTIVE_ACTIVITY_THRESHOLD and not noncrop_veto

        # Format list for Annexure table
        predictions_list = dataset_df.copy()
        # Regenerate activity chart from NEW predictions
        predictions_list['date_str'] = predictions_list['date'].dt.strftime('%Y-%m-%d')
        activity_binary = (dataset_df['prediction'] == "Crop-Activity").astype(int)

        fig = go.Figure()
        
        bar_colors = activity_binary.map({1: '#27ae60', 0: '#e74c3c'})
        
        # Calculate height: if it's a crop, use final_confidence. If no crop, use 100 - final_confidence.
        bar_heights = [conf if pred == 'Crop-Activity' else (100 - conf) 
                       for pred, conf in zip(dataset_df['prediction'], dataset_df['final_confidence'])]
        
        fig.add_trace(go.Bar(
            x=predictions_list["date_str"],
            y=bar_heights,
            marker_color=bar_colors,
            name="Confidence"
        ))
        
        fig.update_layout(
            title="Daily Classification Strength",
            yaxis=dict(
                title="Certainty (%)",
                range=[0, 105],
                gridcolor='rgba(0,0,0,0.05)'
            ),
            xaxis=dict(
                tickangle=-45,
            ),
            template="plotly_white",
            showlegend=False,
            margin=dict(l=50, r=20, t=50, b=50),
            height=400
        )
        activity_base64 = fig_to_base64(fig, is_plotly=True)

        fig_trend = build_vegetation_indices_chart(
            dataset_df, raw_points_df, (analysis_start, analysis_end)
        )
        ndvi_rvi_b64 = fig_to_base64(fig_trend, is_plotly=True)

        p1_crop_mean = dataset_df['p1_crop_conf'].mean()
        p1_nocrop_mean = dataset_df['p1_nocrop_conf'].mean()
        p2_crop_mean = dataset_df['p2_crop_conf'].mean()
        p2_nocrop_mean = dataset_df['p2_nocrop_conf'].mean()
        final_crop_score = (p1_crop_mean * 0.60) + (p2_crop_mean * 0.40)
        final_nocrop_score = (p1_nocrop_mean * 0.60) + (p2_nocrop_mean * 0.40)

        current_crop_days = int((dataset_df['prediction'] == "Crop-Activity").sum())
        current_activity_ratio = (current_crop_days / len(dataset_df) * 100) if len(dataset_df) > 0 else 0
        is_active_final = current_activity_ratio > ACTIVE_ACTIVITY_THRESHOLD and not noncrop_veto

        certainty = compute_parcel_certainty(
            dataset_df,
            raw_points_df,
            is_active_final,
            noncrop_veto,
            final_crop_score,
            final_nocrop_score,
            crop_freq,
            non_crop_freq,
            is_guardband_triggered,
            guardband_conflict=guardband_conflict,
            missing_periods=missing_periods,
            transitional_dominance=veto_info.get("transitional_dominance", False),
        )
        overall_conf = certainty["certainty_score"]

        full_data = {
            "task_id": task_id,
            "satellite_analytics": {
                "metadata": {"coords": raw_data.get("coords") or []},
                "land use/ land cover details": summary_dict,
                "crop_activity_prediction_stats": {
                    "crop_days": crop_days,
                    "total": total_days,
                    "overall_confidence": float(overall_conf),
                    "certainty_score": float(overall_conf),
                    "certainty_tier": certainty["certainty_tier"],
                    "certainty_components": certainty["components"],
                    "certainty_penalties": certainty["penalties_applied"],
                    "is_active": bool(is_active),
                    "noncrop_dominance_veto": bool(noncrop_veto)
                },
                "crop_activity_predictions_list": predictions_list.to_dict(orient="records"),
                 "images": {
                    **raw_data.get("images", {}),      # keep ndvi_b64 and dw_b64 from pickle
                    "activity_b64": activity_base64,
                    "ndvi_b64": ndvi_rvi_b64   # override only the activity chart
                }, # Use B64 images already in pickle
                "vegetation_peak_analysis": peak_analysis,
                "seasonal_activity": seasonal_act,
                "crop_cycles_count": cycle_info["total_cycles"],
                "detected_seasons": cycle_info["detected_seasons"],
                "cycle_details": cycle_info["details"]
            },
            "location_details": loc_res,
            "map_details": loc_res,
            "weather_data": raw_data.get("weather_data", {}),
            "metadata": {"timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")}
        }

        # 7. Generate PDF and Response
        local_pdf_path = await generate_intelligence_report(full_data)
        report_url = upload_private_to_gcs(local_pdf_path, f"dummy_run_24/test_{task_id}_2024.pdf", "application/pdf", is_file=True)
        
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
            
        return {
            "status": "success",
            "task_id": task_id,
            "verdict": "Agri activity detected" if is_active else "Low Agri activity detected",
            "agri_activity": "Agri activity detected" if is_active else "Low Agri activity detected",
            "activity_score": f"{round(activity_ratio, 2)}%",
            "is_active": is_active,
            "noncrop_dominance_veto": noncrop_veto,
            "report_url": report_url,
            "crop_cycles_count": cycle_info["total_cycles"],
            "detected_seasons": cycle_info["detected_seasons"],
            "p1_avg_conf": f"{round(dataset_df['p1_crop_conf'].mean(), 2)}%",
            "p2_avg_conf": f"{round(dataset_df['p2_crop_conf'].mean(), 2)}%",
            "p1_nocrop_avg_conf": f"{round(dataset_df['p1_nocrop_conf'].mean(), 2)}%",
            "p2_nocrop_avg_conf": f"{round(dataset_df['p2_nocrop_conf'].mean(), 2)}%",
            "final_confidence_score": f"{round(overall_conf, 2)}%",
            "certainty_tier": certainty["certainty_tier"],
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
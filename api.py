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
import traceback
from analytics_engine import (
    apply_empirical_logic,
    summarize_crop_activity_table_data, 
    summarize_indices_for_table
)
# Module Imports (Ensure these files are in your deployment directory)
from analytics_engine import run_full_analytics_pipeline
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
def satellite_worker(task_id, coords, end_date):
    return run_full_analytics_pipeline(task_id, coords, end_date)

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
PICKLE_DIR = os.path.join(PROJECT_DIR, "accuracy_tests")
os.makedirs(PICKLE_DIR, exist_ok=True)

@app.post("/test/accuracy")
async def test_accuracy_by_geometry(request: GeometryRequest):
    loop = asyncio.get_event_loop()
    coords = parse_kml_string(request.kml_coordinates)
    task_id = request.task_id

    try:
        # Step 1: Execute Extraction Workers
        b1 = loop.run_in_executor(executor, satellite_worker, task_id, coords, request.end_date)
        b2 = loop.run_in_executor(executor, location_worker, task_id, coords)
        b3 = loop.run_in_executor(executor, weather_worker, coords, request.end_date)

        sat_res, loc_res, weather_res = await asyncio.gather(b1, b2, b3)

        # Step 2: Extract ONLY raw signals for future model testing
        # We ignore 'prediction_stats' and 'seasonal_activity' here
        # Step 2: Extract ALL signals (including images) for exact replay
        raw_signals_payload = {
            "task_id": task_id,
            "coords": coords,
            "vegetation_indices": sat_res.get("timeseries_data", {}).get("vegetation_indices"),
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
            destination_blob_name=f"accuracy_tests/{task_id}_raw.pkl", 
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
        report_url = upload_private_to_gcs(local_pdf_path, f"dummy_report/{task_id}.pdf", "application/pdf", is_file=True)

        # Cleanup ephemeral PDF
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)

        stats = sat_res.get("crop_activity_prediction_stats", {})
        total_instances = stats.get("total", 1) # Default to 1 to avoid division by zero
        active_instances = stats.get("crop_days", 0)
        
        # 2. Calculate the Activity Percentage
        activity_ratio = (active_instances / total_instances) * 100
        
        # 3. Apply your 15% Threshold Logic
        if activity_ratio > 15:
            agri_verdict = "Crop activity detected"
            is_active = True
        else:
            agri_verdict = "Low/No Crop activity detected"
            is_active = False

        # --- FINAL RETURN ---
        return {
            "status": "success",
            "task_id": task_id,
            "verdict": agri_verdict,        # The text: "Crop activity detected"
            "activity_score": f"{round(activity_ratio, 2)}%",
            "is_active": is_active,
            "report_url": report_url,
            "local_pickle_path": pickle_url,
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Test Pipeline Error: {str(e)}")
    
@app.post("/test/replay/{task_id}")
async def replay_test_from_pickle(task_id: str):
    try:
        # 1. Fetch and Load Pickle
        pickle_bytes = download_from_gcs(f"accuracy_tests/{task_id}_raw.pkl")
        if not pickle_bytes:
            raise HTTPException(status_code=404, detail="Pickle not found")
        raw_data = pickle.loads(pickle_bytes)

        # 2. Extract Raw Signals from Pickle
        veg_indices = raw_data.get("vegetation_indices") or [] 
        lc_probs = raw_data.get("land_cover_probs") or []
        loc_res = raw_data.get("location_data") or {}

        # 3. Reconstruct DataFrame for Processing
        df_veg = pd.DataFrame(veg_indices)
        df_dw = pd.DataFrame(lc_probs)
        
        # Normalize and merge (Mirroring your analytics_engine pipeline)
        df_veg['date'] = pd.to_datetime(df_veg['date']).dt.normalize()
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
        
        ###################
        dataset_df = df_veg.merge(df_dw, on='date', how='outer').sort_values('date')
        cycle_info = detect_crop_cycles(dataset_df)

        # integrity = calculate_integrity(dataset_df, cycle_info) 
        # Fill gaps via interpolation
        cols_to_fix = [c for c in dataset_df.columns if c not in ['date', 'prediction']]
        dataset_df[cols_to_fix] = dataset_df[cols_to_fix].interpolate(method='linear', limit_direction='both').fillna(0)

        # 4. RUN YOUR ENGINE LOGIC
        # Apply the exact same empirical logic as the live run
        dataset_df['NDVI_slope'] = np.gradient(dataset_df['NDVI'])
        dataset_df['RVI_slope'] = np.gradient(dataset_df['RVI'])
        
        results = dataset_df.apply(lambda row: apply_empirical_logic(row, cycle_info["detected_seasons"]), axis=1)
        dataset_df['prediction'] = [r[0] for r in results]
        dataset_df['p1_crop_conf'] = [r[1] for r in results]
        dataset_df['p2_crop_conf'] = [r[2] for r in results]

        
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
        if activity_ratio > 15:
            is_active = True
        else:
            is_active = False

        # Format list for Annexure table
        predictions_list = dataset_df.copy()
        # Regenerate activity chart from NEW predictions
        predictions_list['date_str'] = predictions_list['date'].dt.strftime('%Y-%m-%d')
        activity_binary = (dataset_df['prediction'] == "Crop-Activity").astype(int)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions_list["date_str"],
            y=activity_binary,
            mode='lines',
            fill='tozeroy',
            name="Active Cycle",
            line=dict(color='green', width=0),
            fillcolor='rgba(46, 139, 87, 0.3)'
        ))
        fig.add_trace(go.Scatter(
            x=predictions_list["date_str"],
            y=activity_binary,
            mode='markers',
            name="Data Points",
            marker=dict(color=activity_binary.map({1: 'green', 0: 'red'}), size=6, symbol='circle')
        ))
        fig.update_layout(
            title="Agricultural Activity Cycles",
            yaxis=dict(
                tickvals=[0, 1],
                ticktext=["No Crop Activity", "Crop Activity"]  # Changed labels
            ),
            template="plotly_white",
            showlegend=False,  # Remove legend so graph covers full width
            margin=dict(l=50, r=20, t=50, b=50),
            height=400
        )
        activity_base64 = fig_to_base64(fig, is_plotly=True)

        # Then override images in full_data:
       

        full_data = {
            "task_id": task_id,
            "satellite_analytics": {
                "metadata": {"coords": raw_data.get("coords") or []},
                "land use/ land cover details": summary_dict,
                "crop_activity_prediction_stats": {
                    "crop_days": crop_days,
                    "total": total_days
                },
                "crop_activity_predictions_list": predictions_list.to_dict(orient="records"),
                 "images": {
                    **raw_data.get("images", {}),      # keep ndvi_b64 and dw_b64 from pickle
                    "activity_b64": activity_base64    # override only the activity chart
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
        report_url = upload_private_to_gcs(local_pdf_path, f"Cycle_Test_New/test_{task_id}.pdf", "application/pdf", is_file=True)
        
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
            
        return {
            "status": "success",
            "task_id": task_id,
            "agri_activity": "Agri activity detected" if activity_ratio > 15 else "Low Agri activity detected",
            "activity_score": f"{round(activity_ratio, 2)}%",
            "is_active": is_active,
            "report_url": report_url,
            "crop_cycles_count": cycle_info["total_cycles"],
            "detected_seasons": cycle_info["detected_seasons"],
            "p1_avg_conf": f"{round(dataset_df['p1_crop_conf'].mean(), 2)}%",
            "p2_avg_conf": f"{round(dataset_df['p2_crop_conf'].mean(), 2)}%"
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
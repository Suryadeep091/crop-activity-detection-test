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

# Module Imports (Ensure these files are in your deployment directory)
from analytics_engine import run_full_analytics_pipeline
from data_loader import get_centroid_location, get_places_info
from rain_temp import get_one_year_weather_data, get_five_year_weather_data
from pdf_generator import generate_intelligence_report 
from location import get_static_map_b64

# Configuration
matplotlib.use('Agg')
app = FastAPI(title="TerraDrishti Unified Khasra Engine")
executor = ThreadPoolExecutor(max_workers=20)

# --- DATA LOADING ---
# Load CSV once at startup for speed
try:
    df_khasra = pd.read_csv("Telangana_Tehsil_Master.csv")
except Exception as e:
    print(f"CRITICAL: Could not load CSV: {e}")

class KhasraRequest(BaseModel):
    state: str
    district: str
    tehsil: str
    village: str
    khasra_no: str
    end_date: str = datetime.now().strftime("%Y-%m-%d")

# --- CORE UTILITIES ---

def fetch_parcel_geojson(guid, state):
    """Fetches geometry and properties from the external Quantasip API."""
    salt_key = "PAe17K1Rvfeij21TQPlq"
    url = f"https://test-client.quantasip.com/api/parcelData?saltKey={salt_key}&guid={guid}&state={state}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()

def upload_report_to_gcs(local_file_path, task_id):
    bucket_name = "terradrishti"
    
    # 1. Get the base default credentials
    source_creds, project_id = google.auth.default()
    
    # 2. Create impersonated credentials specifically for signing
    # This uses the "Token Creator" role you already granted
    target_principal = "413500342905-compute@developer.gserviceaccount.com"
    creds = impersonated_credentials.Credentials(
        source_credentials=source_creds,
        target_principal=target_principal,
        target_scopes=["https://www.googleapis.com/auth/devstorage.read_write"],
        lifetime=3600
    )
    
    # 3. Use these special credentials for the storage client
    client = storage.Client(credentials=creds, project=project_id)
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"reports/{task_id}.pdf")
    blob.upload_from_filename(local_file_path)
    
    # 4. Generate the URL
    # Because 'creds' now has signing capabilities, this will work
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=60),
        method="GET"
    )
    return url

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
        "temp_5y_b64": temp_5y
    }


# --- UNIFIED ENDPOINT ---

@app.post("/analyze/khasra")
async def get_analysis_by_khasra(request: KhasraRequest):
    """The master endpoint: Lookup -> Fetch -> Analyze -> Report."""
    loop = asyncio.get_event_loop()
    
    # 1. Lookup GUID in the local CSV
    match = df_khasra[
        (df_khasra['state'].str.lower() == request.state.lower()) &
        (df_khasra['khasra_no'].astype(str) == str(request.khasra_no)) &
        (df_khasra['district'].str.lower() == request.district.lower()) &
        (df_khasra['tehsil'].str.lower() == request.tehsil.lower()) &
        (df_khasra['village'].str.lower() == request.village.lower())
    ]
    
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Khasra {request.khasra_no} not found.")
    
    guid = match.iloc[0]['guid']
    
    # 2. Fetch Live Geometry and Properties
    try:
        geo_data = fetch_parcel_geojson(guid, request.state)
        feature = geo_data["features"][0]
        coords = feature["geometry"]["coordinates"]
        properties = feature.get("properties", {})
        
        # Robust unnesting of coordinates
        while isinstance(coords[0][0], list):
            coords = coords[0]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
            
        task_id = f"KH_{request.khasra_no}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geometry Fetch Error: {str(e)}")

    # 3. Parallel Analytics Execution
    try:
        b1 = loop.run_in_executor(executor, satellite_worker, task_id, coords, request.end_date)
        b2 = loop.run_in_executor(executor, location_worker, task_id, coords)
        b3 = loop.run_in_executor(executor, weather_worker, coords, request.end_date)

        sat_res, loc_res, weather_res = await asyncio.gather(b1, b2, b3)

        if not sat_res:
            raise HTTPException(status_code=500, detail="Satellite branch failed.")

        # 4. Generate & Secure the Report
        full_data = {
            "task_id": task_id,
            "satellite_analytics": sat_res,
            "location_details": properties, # Using fetched live properties
            "map_details": loc_res,
            "weather_data": weather_res,
            "metadata": {"timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")}
        }

        local_pdf_path = await generate_intelligence_report(full_data)
        report_url = upload_report_to_gcs(local_pdf_path, task_id)

        # Cleanup
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)

        return {
            "status": "success",
            "task_id": task_id,
            "report_url": report_url,
            "village": properties.get("Village"),
            "district": properties.get("District")
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline Crash: {str(e)}")

if __name__ == "__main__":
    # Use environment PORT for Cloud Run compatibility
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
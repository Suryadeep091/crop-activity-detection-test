import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from datetime import datetime, timedelta
from location import get_static_map_b64
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objs as go
import plotly.io as pio
from google.cloud import storage
import google.auth

# Module Imports
from analytics_engine import run_full_analytics_pipeline
from data_loader import get_centroid_location, get_places_info
from rain_temp import get_one_year_weather_data, get_five_year_weather_data
from pdf_generator import generate_intelligence_report # Your new module

app = FastAPI(title="TerraDrishti 3-Branch Parallel Engine")
executor = ThreadPoolExecutor(max_workers=20)


class AnalysisRequest(BaseModel):
    task_id: str
    coords: List[List[float]]
    end_date: str
    properties: dict = {}


def upload_report_to_gcs(local_file_path, task_id):
    bucket_name = "terradrishti"
    
    # 1. Initialize credentials with the required scope for signing
    credentials, project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # 2. Pass these credentials to the client
    client = storage.Client(credentials=credentials, project=project_id)
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"reports/{task_id}.pdf")
    blob.upload_from_filename(local_file_path)
    
    service_account_email = "413500342905-compute@developer.gserviceaccount.com"

    # 3. Generate the URL (The library will now call the IAM API remotely)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=60),
        method="GET",
        service_account_email=service_account_email
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

@app.post("/analyze/summary")
async def get_combined_analysis(request: AnalysisRequest):
    loop = asyncio.get_event_loop()
    
    try:
        # 1. TRIGGER TRIPLE CONCURRENCY FOR DATA GATHERING
        b1 = loop.run_in_executor(executor, satellite_worker, request.task_id, request.coords, request.end_date)
        b2 = loop.run_in_executor(executor, location_worker, request.task_id, request.coords)
        b3 = loop.run_in_executor(executor, weather_worker, request.coords, request.end_date)

        # Wait for data processing to finish
        sat_res, loc_res, weather_res = await asyncio.gather(b1, b2, b3)

        if not sat_res:
            raise HTTPException(status_code=500, detail="Satellite Analytics branch returned no data.")

        # 2. CONSTRUCT FINAL DATA OBJECT
        full_data = {
            "task_id": request.task_id,
            "satellite_analytics": sat_res,
            "location_details": request.properties,
            "map_details": loc_res,
            "weather_data": weather_res,
            "metadata": {
                "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
            }
        }

        # 3. TRIGGER PDF GENERATION (In-Memory / Buffer Flow)
        # This happens on the backend before the response is sent
        local_pdf_path = await generate_intelligence_report(full_data)

        report_url = upload_report_to_gcs(local_pdf_path, request.task_id)

        # 4. CLEANUP LOCAL FILE (Save Disk Space)
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)

        return {
            "status": "success",
            "task_id": request.task_id,
            "report_url": report_url, # Now returning a URL instead of a massive Base64 string
            "satellite_analytics": sat_res,
            "location_details": request.properties,
            "map_details": loc_res
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
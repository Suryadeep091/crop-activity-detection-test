import streamlit as st
import requests
import json
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import pandas as pd
import os

# --- CONFIG & STYLING (Keep your existing CSS here) ---
st.set_page_config(layout="wide")

# --- DATA LOADING (Khasra JSON) ---
@st.cache_data
def load_khasra_data():
    with open("Khasra.json", "r") as f:
        khasra_json = json.load(f)
    features = []
    for collection in khasra_json:
        if "features" in collection:
            features.extend(collection["features"])
    return features

features = load_khasra_data()

# --- HELPER FUNCTIONS ---
def create_map(center=[17.0944, 79.7750]):
    return folium.Map(location=center, zoom_start=17, tiles='OpenStreetMap')

# --- MAIN UI ---
st.title("🌾 TerraDrishti API Dashboard")

st.sidebar.header("Input Parameters")
input_method = st.sidebar.radio("Input Method", ["Select Khasra No.", "Draw on Map"])

selected_khasra = None
coords = None

if input_method == "Select Khasra No.":
    khasra_options = [f["properties"]["Khasra_No"] for f in features]
    selected_khasra = st.sidebar.selectbox("Select Khasra No.", khasra_options)
    
    feature = next((f for f in features if f["properties"]["Khasra_No"] == selected_khasra), None)
    if feature:
        raw_coords = feature["geometry"]["coordinates"]
            
        # Your JSON is MultiPolygon but coords are directly inside
        # Normalize to a list of [lon, lat]
        if feature["geometry"]["type"] == "MultiPolygon":
            # Take first polygon ring
            coords = raw_coords  
            # If it's nested [[[x,y],[x,y],...]] → flatten
            if isinstance(coords[0][0], (list, tuple)):
                coords = coords[0]
        
        st.session_state["json_coords"] = coords
        
        # Preview Map
        m = create_map(center=[coords[0][1], coords[0][0]])
        folium.Polygon(locations=[[c[1], c[0]] for c in coords], color="green", fill=True).add_to(m)
        st_folium(m, width="100%", height=400, key="khasra_map")


# --- API TRIGGER ---
if st.sidebar.button("Run Intelligence Report"):
    active_coords = st.session_state.get("json_coords")
    
    if active_coords:
        if active_coords[0] != active_coords[-1]:
            active_coords.append(active_coords[0])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        generated_task_id = f"KH_{selected_khasra}_{timestamp}"

        payload = {
            "task_id": generated_task_id,
            "coords": active_coords,
            "end_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        start_time_dt = datetime.now()
        call_time_str = start_time_dt.strftime("%d %b %Y, %I:%M:%S %p")
        
        with st.spinner("Executing Parallel Intelligence Pipeline..."):
            try:
                response = requests.post("http://localhost:8000/analyze/summary", json=payload)
                response.raise_for_status()
                data = response.json()
                
                # --- 1. DATA EXTRACTION ---
                sat_analytics = data.get("satellite_analytics", {})
                summary = sat_analytics.get("summary", {})
                loc = data.get("location_details", {})
                # weather = data.get("weather_data", {})
                paths = sat_analytics.get("paths", {}) # Get image paths

                # --- 2. TOP ROW: GEOSPATIAL METADATA ---
                st.markdown("### 📍 Location & Connectivity")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("District", loc.get("district", "N/A"))
                c2.metric("Water Source", loc.get("Nearest Water Source", "N/A"))
                c3.metric("Water Distance", f"{loc.get('Water Distance (km)', 0)} km")
                c4.metric("Railway", loc.get("nearest_railway_station", {}).get("name", "N/A"))

                # --- 3. MIDDLE ROW: SATELLITE SUMMARY & AI ---
                st.markdown("---")
                col_sat1, col_sat2 = st.columns([2, 1])
                
                with col_sat1:
                    st.subheader("🌾 Land Cover Breakdown")
                    cols = st.columns(3)
                    # Filter and show percentages
                    for idx, (land_type, values) in enumerate(summary.items()):
                        if isinstance(values, dict) and values.get('percent', 0) > 0:
                            cols[idx % 3].info(f"**{land_type.replace('_', ' ').title()}**: {values['percent']}%")
                
                with col_sat2:
                    st.subheader("🤖 AI Prediction")
                    pred_stats = sat_analytics.get("prediction_stats", {})
                    st.metric("Predicted Crop Days", pred_stats.get("crop_days", 0))
                    st.metric("Crop Intensity", summary.get('crop_intensity', 0))

                # --- 4. VISUAL INTELLIGENCE (The Plots) ---
                st.markdown("---")
                st.subheader("📈 Temporal Analysis & Trends")
                
                # Display the three main plots generated by Branch 1
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    if paths.get("ndvi_plot") and os.path.exists(paths["ndvi_plot"]):
                        st.image(paths["ndvi_plot"], caption="Vegetation Indices (NDVI/EVI/RVI)")
                    
                    if paths.get("dw_plot") and os.path.exists(paths["dw_plot"]):
                        st.image(paths["dw_plot"], caption="Land Cover Probability Trends (Dynamic World)")

                with v_col2:
                    if paths.get("activity_plot") and os.path.exists(paths["activity_plot"]):
                        st.image(paths["activity_plot"], caption="XGBoost Crop Activity Prediction")
                    
                    # Optional: Provide a download button for the raw predictions
                    if paths.get("predictions_csv") and os.path.exists(paths["predictions_csv"]):
                        with open(paths["predictions_csv"], "rb") as file:
                            st.download_button(
                                label="📥 Download Prediction CSV",
                                data=file,
                                file_name=f"predictions_{generated_task_id}.csv",
                                mime="text/csv"
                            )
                
                # --- 5. WEATHER INSIGHTS ---
                # st.markdown("---")
                # st.subheader("🌦️ Weather Intelligence")
                # w_tab1, w_tab2 = st.tabs(["Rainfall (Daily)", "Temperature (Historical)"])
                
                # with w_tab1:
                #     daily_df = pd.DataFrame(weather.get("daily", []))
                #     if not daily_df.empty:
                #         st.line_chart(daily_df.set_index("Date")["Rainfall_mm"])
                
                # with w_tab2:
                #     monthly_df = pd.DataFrame(weather.get("monthly", []))
                #     if not monthly_df.empty:
                #         st.line_chart(monthly_df.set_index("Month")[["Min_temp_celsius", "Max_temp_celsius"]])
                pdf_report_path = data.get("pdf_report_path")
                st.markdown("---")
                st.subheader("📄 Official Intelligence Report")
                
                if pdf_report_path and os.path.exists(pdf_report_path):
                    with open(pdf_report_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Full PDF Report",
                            data=f,
                            file_name=f"TerraDrishti_Report_{generated_task_id}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.warning("PDF Report is still being finalized or path is inaccessible.")

                end_time_dt = datetime.now()
                loaded_time_str = end_time_dt.strftime("%d %b %Y, %I:%M:%S %p")
                
                # Calculate Duration
                duration = end_time_dt - start_time_dt
                
                # Print and Show on Dashboard
                print(f"Duration: {duration.total_seconds():.2f} seconds")
                st.info(f"⚡ Analysis completed in {duration.total_seconds():.2f} seconds")

                # --- 6. FOOTER PERFORMANCE ---
                duration = datetime.now() - start_time_dt
                st.caption(f"Pipeline Execution Time: {duration.total_seconds():.2f}s | Task ID: {generated_task_id}")

            except Exception as e:
                st.error(f"UI Error: {e}")
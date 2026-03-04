import streamlit as st
import requests
import json
import folium
from streamlit_folium import st_folium
from datetime import datetime
import pandas as pd
import os

# --- CONFIG & STYLING ---
st.set_page_config(layout="wide")

# --- DATA LOADING (CSV & JSON) ---
@st.cache_data
def load_reference_csv():
    # Load the CSV containing Khasra and GUID mapping
    # Based on your image_70371b.png
    return pd.read_csv("Telangana_Tehsil_Master.csv") # Ensure this file name matches yours

df_khasra = load_reference_csv()

# --- API HELPER ---
def fetch_parcel_data(guid, state):

    salt_key = "PAe17K1Rvfeij21TQPlq"
    url = f"https://test-client.quantasip.com/api/parcelData?saltKey={salt_key}&guid={guid}&state={state}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# --- MAIN UI ---
st.title("🌾 TerraDrishti API Dashboard")

st.sidebar.header("Search Parcel")

# 1. New Text Input for Khasra Number
search_khasra = st.sidebar.text_input("Enter Khasra No. (e.g., 15, 21, 73/91)")

active_coords = None
selected_details = None
details = None

if search_khasra:
    # 2. Check if Khasra exists in local CSV
    match = df_khasra[df_khasra['khasra_no'].astype(str) == str(search_khasra)]
    
    if not match.empty:
        st.sidebar.success(f"✅ Khasra {search_khasra} found in records.")
        
        # Get GUID and State from CSV
        row = match.iloc[0]
        guid = row['guid']
        state = row['state']
        
        # 3. Call the external API with the GUID
        with st.spinner("Fetching Live GeoJSON..."):
            api_response = fetch_parcel_data(guid, state)
            print(f"API Response: {json.dumps(api_response, indent=2)}")  # Debug print to check structure
            if api_response and "features" in api_response and len(api_response["features"]) > 0:
                feature = api_response["features"][0]
                properties = feature.get("properties", {})
                geometry = feature.get("geometry", {})
                raw_coords = geometry.get("coordinates", [])
                
                # --- ROBUST COORDINATE NORMALIZATION ---
                extracted_coords = None
                
                # The API response shows coordinates as a list of [lon, lat] pairs directly.
                # Check if the first element is a list (a coordinate pair)
                if isinstance(raw_coords[0], list):
                    # If it's [ [lon, lat], ... ]
                    if isinstance(raw_coords[0][0], (int, float)):
                        extracted_coords = raw_coords
                    # If it's [ [ [lon, lat], ... ] ]
                    elif isinstance(raw_coords[0][0], list):
                        extracted_coords = raw_coords[0]
                        # Handle deeper nesting if it's MultiPolygon [[[[lon, lat]]]]
                        if isinstance(extracted_coords[0][0], list):
                            extracted_coords = extracted_coords[0]

                if extracted_coords:
                    active_coords = extracted_coords
                    st.session_state["active_coords"] = extracted_coords
                    st.session_state["details"] = properties
                    # Preview Map
                    st.markdown(f"### 🗺️ Preview: Village {properties.get('Village')}, {properties.get('District')}")
                    
                    # Folium uses [lat, lon] for the center
                    center_lat = active_coords[0][1]
                    center_lon = active_coords[0][0]
                    
                    # ESRI Satellite Tile URL
                    satellite_tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                    attr = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'

                    m = folium.Map(
                        location=[center_lat, center_lon], 
                        zoom_start=16, 
                        tiles=satellite_tiles, 
                        attr=attr
                    )

                    # Your existing polygon code
                    folium.Polygon(
                        locations=[[c[1], c[0]] for c in active_coords], 
                        color="yellow",  # Yellow or white often pops better on dark satellite imagery
                        weight=2,
                        fill=True,
                        fill_opacity=0.4
                    ).add_to(m)

                    st_folium(m, width="100%", height=400, key="khasra_map")
                else:
                    st.sidebar.error("Failed to parse coordinate structure.")
            else:
                st.sidebar.error("No features found in API response.")
    else:
        st.sidebar.warning(f"⚠️ Khasra {search_khasra} not found in local database.")
# --- API TRIGGER ---
if st.sidebar.button("Run Intelligence Report"):
    active_coords = st.session_state.get("active_coords")
    details = st.session_state.get("details", {})
    
    if active_coords:
        if active_coords[0] != active_coords[-1]:
            active_coords.append(active_coords[0])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        generated_task_id = f"KH_{search_khasra}_{timestamp}"
        print(f"Generated Task ID: {generated_task_id}")
        print(f"Active Coords: {active_coords}")
        print(f"Details: {details}")
        print(f"End Date: {datetime.now().strftime('%Y-%m-%d')}")
        payload = {
            "task_id": generated_task_id,
            "coords": active_coords,
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "properties": details
        }
        
        start_time_dt = datetime.now()
        call_time_str = start_time_dt.strftime("%d %b %Y, %I:%M:%S %p")
        
        with st.spinner("Executing Parallel Intelligence Pipeline..."):
            try:
                response = requests.post("https://terradristi-crop-activity-413500342905.asia-south1.run.app/analyze/summary", json=payload)
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
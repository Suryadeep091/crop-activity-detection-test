import streamlit as st
import requests
import json
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

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
        # Ensure the polygon is closed for GEE
        if active_coords[0] != active_coords[-1]:
            active_coords.append(active_coords[0])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        generated_task_id = f"KH_{selected_khasra}_{timestamp}"

        payload = {
            "task_id": generated_task_id,
            "coords": active_coords, # Send the validated coordinates
            "end_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # LOGGING: Verify in your terminal what is actually being sent
        print(f"DEBUG: Sending Task {generated_task_id} with {len(active_coords)} points.")
        # Record the START time as a datetime object
        start_time_dt = datetime.now()
        call_time_str = start_time_dt.strftime("%d %b %Y, %I:%M:%S %p")
        
        with st.spinner("FastAPI is crunching Satellite Data..."):
            try:
                # 2. Call the FastAPI Backend
                
                response = requests.post("http://localhost:8000/analyze/summary", json=payload)
                response.raise_for_status()
                data = response.json()
                
                summary = data["summary"]
                
                # Record the END time as a datetime object
                end_time_dt = datetime.now()
                loaded_time_str = end_time_dt.strftime("%d %b %Y, %I:%M:%S %p")
                
                # Calculate Duration
                duration = end_time_dt - start_time_dt
                
                # Print and Show on Dashboard
                print(f"Duration: {duration.total_seconds():.2f} seconds")
                st.info(f"⚡ Analysis completed in {duration.total_seconds():.2f} seconds")

                # 3. Display the Land Type Summary (using your existing card logic)
                st.subheader("Land Type Summary")
                for land_type, values in summary.items():
                    if isinstance(values, dict) and 'percent' in values:
                        st.write(f"**{land_type.title()}**: {values['percent']}%")
                
                if 'crop_intensity' in summary:
                    st.metric("Crop Intensity (Peaks)", summary['crop_intensity'])

            except Exception as e:
                st.error(f"Backend Communication Error: {e}")
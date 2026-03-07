import os
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from data_loader import create_test_data, process_parcel_data
from test_model import predict_from_pickle
from datetime import datetime

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 100)      # Show more rows


#

def summarize_crop_activity_table_data(predictions, start_date, end_date):
                """
                Summarizes crop activity into a table-ready format for PDF.

                Args:
                    predictions (pd.DataFrame): Must have 'date' and 'prediction' columns.
                    start_date (datetime): Start date of the analysis.
                    end_date (datetime): End date of the analysis.

                Returns:
                    month_labels: List of month-year labels with Rabi/Kharif tags.
                    table_data: List of (label, [bool per month]) for Crop, No Crop, Missing.
                    start_month: Integer month of the first entry.
                    start_year: Integer year of the first entry.
                """
                df = predictions.copy()
                df["date"] = pd.to_datetime(df["date"])
                df["Month"] = df["date"].dt.month
                df["Year"] = df["date"].dt.year

                # Generate list of months from start_date to end_date
                month_range = pd.date_range(start=start_date, end=end_date, freq="MS")
                month_labels = []
                month_activity = {}

                for dt in month_range:
                    month_num = dt.month
                    year_num = dt.year

                    # Assign Rabi or Kharif
                    if month_num in [11, 12, 1, 2, 3]:
                        season = "Rabi"
                    elif month_num in [6, 7, 8, 9, 10]:
                        season = "Kharif"
                    else:
                        season = "Zaid"

                    # Label format: Jan-2025 (Rabi)
                    label = f"{dt.strftime('%b-%Y')} ({season})"
                    month_labels.append(label)

                    # Filter predictions for this month-year
                    month_df = df[(df["Month"] == month_num) & (df["Year"] == year_num)]
                    if len(month_df) == 0:
                        month_activity[label] = None
                    else:
                        crop_count = (month_df["prediction"] == "Crop-Activity").sum()
                        no_crop_count = (
                            month_df["prediction"] == "No Crop-Activity"
                        ).sum()

                        if crop_count > no_crop_count:
                            month_activity[label] = True
                        elif crop_count < no_crop_count:
                            month_activity[label] = False
                        else:
                            month_activity[label] = True  # Tie

                # Create table data for PDF
                # Inside summarize_crop_activity_table_data
                table_data = [
                    {
                        "label": "Crop Activity",
                        "values": [month_activity[label] is True for label in month_labels],
                    },
                    {
                        "label": "No Crop Activity",
                        "values": [month_activity[label] is False for label in month_labels],
                    },
                    {
                        "label": "Missing Data",
                        "values": [month_activity[label] is None for label in month_labels],
                    },
                ]

                return {
                    "labels": month_labels,
                    "rows": table_data,
                    "start_month": month_range[0].month,
                    "start_year": month_range[0].year
                }


def summarize_indices_for_table(df):
        peak_data = []
        missing_periods = []

        month_map = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }

        for index in ['NDVI', 'EVI', 'RVI']:
            if index in df.columns:
                grouped = df[index].groupby([df.index.year, df.index.month]).mean()
                if not grouped.empty:
                    (peak_year, peak_month) = grouped.idxmax()
                    peak_value = grouped.max()

                    if peak_value < 0.3:
                        note = "Low health vegetation"
                    elif 0.3 <= peak_value < 0.6:
                        note = "Moderately healthy vegetation"
                    else:
                        note = "Very healthy vegetation"

                    peak_data.append({
                        "index": index,
                        "full_name": {
                            "NDVI": "Normalized Difference Vegetation Index",
                            "EVI": "Enhanced Vegetation Index",
                            "RVI": "Radar Vegetation Index"
                        }[index],
                        "peak_period": f"{month_map[peak_month]} {peak_year}",
                        "peak_value": round(peak_value, 2),
                        "note": note
                    })

        # Missing NDVI/EVI periods
        if all(col in df.columns for col in ['NDVI', 'EVI']):
            missing_months = df[['NDVI', 'EVI']].isna()
            missing_dates = df.index[missing_months.any(axis=1)]
            missing_ym = sorted(set((d.year, d.month) for d in missing_dates))
            missing_periods = [f"{month_map[m]} {y}" for y, m in missing_ym]
        
        return peak_data, missing_periods

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

def apply_empirical_logic(row):
    # Extract signals and month
    date_val = pd.to_datetime(row.get('date'))
    month = date_val.month
    
    ndvi = row.get('NDVI', 0)
    rvi = row.get('RVI', 0)
    # Combine AI classes
    crops = row.get('crops', 0)
    flooded_veg = row.get('flooded_vegetation', 0)
    crop_prob = crops + flooded_veg
    tree_prob = row.get('trees', 0)
    built = row.get('built', 0)
    water = row.get('water', 0)
    bare = row.get('bare', 0)
    shrubs = row.get('shrub_and_scrub', 0)
    snow = row.get('snow_and_ice', 0)
    
    # Global Tree Filter
    if tree_prob > 0.50 or tree_prob > crop_prob:
        return "No Crop-Activity"
    if built > 0.50 or built > crop_prob:
        return "No Crop-Activity"
    if water > 0.50 or water > crop_prob:
        return "No Crop-Activity"
    if bare > 0.50 or bare > crop_prob:
        return "No Crop-Activity"
    if shrubs > 0.50 or shrubs > crop_prob:
        return "No Crop-Activity"
    if snow > 0.50 or snow > crop_prob:
        return "No Crop-Activity"

    # --- SOWING DETECTION (Catching the start of the cycle) ---
    # If the AI sees Flooded Veg (Paddy) or if we see early structure with some AI confidence
    # We use much lower NDVI thresholds here.
    if flooded_veg > 0.40 and rvi > 0.30:
        return "Crop-Activity" # Early Paddy sowing detected
    
    if crop_prob > 0.35 and (ndvi > 0.25 and rvi > 0.35):
        return "Crop-Activity" # Early Rabi/Zaid growth detected

    # --- FULL VEGETATIVE LOGIC (Your existing thresholds for peak) ---
    # June to October (Kharif)
    if 6 <= month <= 10:
        if (rvi > 0.45 and crop_prob > 0.40) or (ndvi > 0.35 and crop_prob > 0.50):
            return "Crop-Activity"

    # November to March (Rabi)
    elif month in [11, 12, 1, 2, 3]:
        if ndvi > 0.40 and (crop_prob > 0.40 or rvi > 0.35):
            return "Crop-Activity"

    # April to May (Zaid)
    elif month in [4, 5]:
        if ndvi > 0.45 and rvi > 0.40:
            return "Crop-Activity"

    return "No Crop-Activity"

def run_full_analytics_pipeline(task_id, coords, end_date_str):
    try:
        # 1. Extraction (GEE)
        extraction_result = process_parcel_data(task_id, coords, end_date_str)
        if not extraction_result:
            return None
        
        df_all, summary_dict, df_dw_raw = extraction_result
        # We still need data_dir for the CSV predictions (optional)
        data_dir = os.path.join("data", task_id)
        os.makedirs(data_dir, exist_ok=True)
        analysis_end = datetime.strptime(end_date_str, "%Y-%m-%d")
        analysis_start = analysis_end - pd.DateOffset(years=1)

        # --- 2. Dynamic World Time-Series Plot (Base64) ---
        fig_dw = go.Figure()

        # Define bands and a professional color palette
        bands = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
        colors = ['#1f77b4', '#2ca02c', '#94ce7b', '#17becf', '#ff7f0e', '#8c564b', '#7f7f7f', '#e377c2', '#bcbd22']

        # Add traces for each land class found in the dataframe
        for b, color in zip(bands, colors):
            if b in df_dw_raw.columns:
                fig_dw.add_trace(
                    go.Scatter(
                        x=df_dw_raw['date'],
                        y=df_dw_raw[b],
                        mode='lines',
                        name=b.replace('_', ' ').title(),
                        line=dict(color=color, width=1.5),
                        stackgroup='one' # Optional: Use this if you want a stacked area chart
                    )
                )

        # Update layout for consistency
        fig_dw.update_layout(

            xaxis_title="Date",
            yaxis_title="Probability Share of Classes",
            template="plotly_white",
            xaxis=dict(
                type="date",
                range=[
                    analysis_start.strftime("%Y-%m-%d"),
                    analysis_end.strftime("%Y-%m-%d"),
                ],
               
                tickformat="%b %Y",
                dtick="M1"
            ),
            yaxis=dict(range=[0, 1.05]), # Probabilities range from 0 to 1
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="lightgrey",
                borderwidth=1,
                font=dict(size=10)
            ),
            margin=dict(l=50, r=20, t=50, b=50),
            height=400
        )

        # Convert to Base64 for the PDF
        dw_base64 = fig_to_base64(fig_dw, is_plotly=True)

        # --- 2. Create Dataset & Apply Empirical Labels (SKIP PICKLE) ---
        # Merge DW raw data with S2/S1 data for the labeling logic
        # 'outer' keeps dates from BOTH dataframes even if they don't match
        # --- 2. Create Dataset & Apply Empirical Labels ---
        # Step A: Perform Outer Join to align all sensors
        # --- 2. Create Dataset & Apply Empirical Labels ---
        
        # A. Force both dataframes to have identical datetime formats for merging
        # --- 2. Create Dataset & Apply Empirical Labels ---
        
        # A. Normalize Dates
        df_all['date'] = pd.to_datetime(df_all['date']).dt.normalize()
        df_dw_raw['date'] = pd.to_datetime(df_dw_raw['date']).dt.normalize()

        # B. REMOVE DUPLICATES: Drop DW columns from df_all before merging
        # This prevents the creation of _x and _y columns
        dw_cols = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
        df_all_clean = df_all.drop(columns=[c for c in dw_cols if c in df_all.columns])

        # C. Perform Outer Join (Now creates a single 'crops' column)
        dataset_df = df_all_clean.merge(df_dw_raw, on='date', how='outer').sort_values('date')

        # D. Interpolate ALL critical bands
        indices_cols = ['NDVI', 'EVI', 'RVI']
        cols_to_fix = [c for c in dw_cols + indices_cols if c in dataset_df.columns]
        
        # This will now correctly fill the 0s in rows 1, 3, and 4
        dataset_df[cols_to_fix] = dataset_df[cols_to_fix].interpolate(
            method='linear', limit_direction='both'
        )

        # E. Final Sanitize & Apply Logic
        dataset_df = dataset_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Now the logic will find 'crops' and 'flooded_vegetation' successfully
        dataset_df['prediction'] = dataset_df.apply(apply_empirical_logic, axis=1)
        
        predictions = dataset_df.copy()
        test_df = dataset_df.copy()

        # --- 3. Rest of your existing plotting and summary logic ---
        # (Peak analysis, Plotly charts, etc. use the 'predictions' df created above)
        
        # Peak summary calculation
        temp_df = test_df.copy()
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        temp_df.set_index('date', inplace=True)
        peak_data, missing_periods = summarize_indices_for_table(temp_df)
        
        seasonal_summary = summarize_crop_activity_table_data(predictions, analysis_start, analysis_end)
        
        activity_binary = (predictions["prediction"] == "Crop-Activity").astype(int)
        predictions["date_str"] = pd.to_datetime(predictions["date"]).dt.strftime("%Y-%m-%d")
        predictions_list = predictions.to_dict(orient="records")

    
        activity_binary = (predictions["prediction"] == "Crop-Activity").astype(
            int
        )
        predictions["date_str"] = pd.to_datetime(
            predictions["date"]
        ).dt.strftime("%Y-%m-%d")
        zero_indices = [i for i, val in enumerate(activity_binary) if val == 0]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[
                    predictions["date_str"][i]
                    for i, val in enumerate(activity_binary)
                    if val == 1
                ],
                y=[1] * sum(activity_binary),
                name="Crop Activity",
                marker_color="green",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[predictions["date_str"][i] for i in zero_indices],
                y=[0] * len(zero_indices),
                mode="markers",
                name="No Crop Activity",
                line=dict(color="red", dash="dot", width=0.5),
                showlegend=True,
            )
        )
        fig.update_layout(
          
            xaxis_title="Date",
            yaxis_title="Activity (1 = Crop, 0 = No Crop)",
            template="plotly_white",
            xaxis_tickangle=-45,
            yaxis=dict(tickmode="array", tickvals=[0, 1], range=[0, 1]),
            xaxis=dict(
                range=[
                    analysis_start.strftime("%Y-%m-%d"),
                    analysis_end.strftime("%Y-%m-%d"),
                ],
                title="Date",
            ),
            showlegend=True,
            legend=dict(
                orientation="v",      # Vertical orientation
                yanchor="top",        # Anchor at the top of the legend box
                y=0.99,               # Positioned at the very top (1.0 is the top edge)
                xanchor="right",      # Anchor at the right of the legend box
                x=0.99,               # Positioned at the very right edge
                bgcolor="rgba(255, 255, 255, 0.5)" # Semi-transparent background
            ),
            margin=dict(l=50, r=20, t=50, b=50),
            bargap=0.1,
            height=400,
        )
        # ... (Scatter trace) ...
        activity_base64 = fig_to_base64(fig, is_plotly=True)

        # --- 5. Save NDVI/EVI/RVI Static Plot (Base64) ---
        fig_indices = go.Figure()

# Loop through your indices and add traces
        for col, color in [('NDVI', 'green'), ('EVI', 'blue'), ('RVI', 'orange')]:
            if col in test_df.columns:
                fig_indices.add_trace(
                    go.Scatter(
                        x=test_df['date'],
                        y=test_df[col],
                        mode='lines',
                        name=col,
                        line=dict(color=color, width=2)
                    )
                )

        # Update layout for top-right internal legend
        fig_indices.update_layout(

            xaxis_title="Date",
            yaxis_title="Index Value",
            template="plotly_white",
            xaxis=dict(
                type="date",
                range=[
                    analysis_start.strftime("%Y-%m-%d"),
                    analysis_end.strftime("%Y-%m-%d"),
                ],# Use the 1-year range derived earlier
                tickformat="%b %Y",
                dtick="M1"
            ),
            yaxis=dict(range=[-0.1, 1.0]), # Indices usually stay between -0.1 and 1
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.6)", # Slight transparency to see grid lines
                bordercolor="lightgrey",
                borderwidth=1
            ),
            margin=dict(l=50, r=20, t=50, b=50), # Expanded horizontal area
            height=400
        )

        # Convert to Base64 for the PDF
        ndvi_base64 = fig_to_base64(fig_indices, is_plotly=True)

        print(seasonal_summary)

        indices_data = dataset_df[['date', 'NDVI', 'EVI', 'RVI']].copy()
        indices_data['date'] = indices_data['date'].dt.strftime('%Y-%m-%d')
        # Convert to list of dicts: [{'date': '2025-01-01', 'NDVI': 0.45, ...}, ...]
        indices_raw_list = indices_data.to_dict(orient="records")

        # 2. Prepare Dynamic World (DW) Raw Data
        dw_raw_export = df_dw_raw.copy()
        dw_raw_export['date'] = pd.to_datetime(dw_raw_export['date']).dt.strftime('%Y-%m-%d')
        dw_raw_list = dw_raw_export.to_dict(orient="records")

        # --- UPDATED RETURN STATEMENT ---
        return {
            "land use/ land cover details": summary_dict,
            "vegetation_peak_analysis": peak_data,        # Added peak data
            "missing_data": missing_periods,
            "seasonal_activity": seasonal_summary,
            "crop_activity_predictions_list": predictions_list,
            "crop_activity_prediction_stats": {
                "total": len(predictions),
                "crop_days": int(sum(activity_binary))
            },
            "timeseries_data": {
                "vegetation_indices": indices_raw_list, # NDVI, EVI, RVI points
                "land_cover_probs": dw_raw_list         # Dynamic World probability points
            },
            "images": {
                "ndvi_b64": ndvi_base64,
                "dw_b64": dw_base64,
                "activity_b64": activity_base64
            },
        }
    except Exception as e:
            print(f"Engine Error: {e}")
            return None
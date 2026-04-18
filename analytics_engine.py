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
from data_loader import detect_crop_cycles

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

# def apply_empirical_logic(row, prev_ndvi=None):
#     date_val = pd.to_datetime(row.get('date'))
#     month = date_val.month
    
#     # 1. Extract Signals
#     ndvi = row.get('NDVI', 0)
#     evi = row.get('EVI', 0) # Added EVI for better crop detection
#     rvi = row.get('RVI', 0)
#     crops = row.get('crops', 0)
#     flooded_veg = row.get('flooded_vegetation', 0)
#     crop_prob = crops + flooded_veg

#     # --- NEW: INDEX OVERRIDE (PRIORITY CHECK) ---
#     # If indices are exceptionally high, we ignore DW dominance.
#     # High NDVI/EVI suggests active photosynthesis that DW might be mislabeling.
#     if ndvi > 0.55 or evi > 0.45:
#         # Check if shrubs, built, or flooded_veg are the dominant "noise"
#         dominant_noise = max(row.get('built', 0), row.get('shrub_and_scrub', 0), row.get('flooded_vegetation', 0))
#         if dominant_noise > crop_prob:
#             return "Crop-Activity (Index Override)"

#     # 2. Global Guardrails (Standard Logic)
#     # If trees, water, or bare soil are truly dominant and indices aren't in 'Override' range
#     for noise_class in ['trees', 'water', 'bare']:
#         val = row.get(noise_class, 0)
#         if val > 0.50 and val > crop_prob:
#             return "No Crop-Activity"

#     # --- MONSOON SWITCH (June - Sept: Heavy Rain / Kharif) ---
#     if 6 <= month <= 9:
#         if (rvi > 0.40 and crop_prob > 0.35) or (flooded_veg > 0.50 and rvi > 0.30):
#             return "Crop-Activity"
            
#     # --- WINTER / DRY (Oct - March: Rabi) ---
#     elif month in [10, 11, 12, 1, 2, 3]:
#         if (ndvi > 0.38 and crop_prob > 0.40) or (rvi > 0.45):
#             return "Crop-Activity"

#     # --- SUMMER (April - May: Zaid) ---
#     elif month in [4, 5]:
#         if ndvi > 0.42 and rvi > 0.40:
#             return "Crop-Activity"

#     # --- THE "TREND" CATCHER ---
#     if prev_ndvi is not None:
#         if (ndvi - prev_ndvi) > 0.08:
#              return "Crop-Activity"

#     return "No Crop-Activity"


def apply_empirical_logic(row, detected_seasons):
    """
    row: current data point
    detected_seasons: list of seasons found in the full year (e.g., ['Kharif', 'Rabi'])
    """
    date_val = pd.to_datetime(row.get('date'))
    month = date_val.month
    ndvi = row.get('NDVI', 0)
    ndvi_slope = row.get('NDVI_slope', 0)
    
    # Determine current season based on month
    current_season = None
    if 6 <= month <= 10: current_season = "Kharif"
    elif month in [11, 12, 1, 2, 3]: current_season = "Rabi"
    elif month in [4, 5]: current_season = "Zaid"
    
    # --- PIPELINE 1: CYCLE & SLOPE CONFIDENCE ---
    has_cycle = current_season in detected_seasons
    
    # Simple base index logic scaled to 100
    base_index = min(max(ndvi * 100, 0), 100) 
    
    # Abs slope scaling: cap at 0.01 (daily max biological capability) -> 100% influence.
    slope_mag = min((abs(ndvi_slope) / 0.01) * 100, 100)
    
    if has_cycle:
        # Peak season: Trust high indices or steep slopes to mean crop activity.
        p1_crop_conf = min(30 + 0.5 * base_index + 0.2 * slope_mag, 100)
        p1_nocrop_conf = 100 - p1_crop_conf
    else:
        # Non-peak season: Trust low indices and flat slopes
        # Cap P1 NoCrop penalty so it doesn't unconditionally block Pipeline 2
        p1_nocrop_conf = min(50 + 0.3 * (100 - base_index) + 0.2 * (100 - slope_mag), 85)
        p1_crop_conf = 100 - p1_nocrop_conf

    # --- PIPELINE 2: DYNAMIC WORLD PROBABILITY CONFIDENCE ---
    classes = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    probs = {c: row.get(c, 0) for c in classes}
    
    # Sort classes by probability descending
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_class, top_prob = sorted_probs[0]
    runner_class, runner_prob = sorted_probs[1]
    
    margin = top_prob - runner_prob
    
    if top_class in ['crops']:
        # Base linear confidence scaling for crops (Empowered to beat P1 caps)
        dw_confidence = min(60 + (margin * 80), 100)
        p2_crop_conf = dw_confidence
        p2_nocrop_conf = 100 - dw_confidence
    else:
        # AGGRESSIVE SCALING for No-Crop blockages (Trees, Water, etc.)
        # Reaches 100% confidence much faster (at just a 0.50 gap instead of 1.0)
        dw_confidence = min(50 + (margin * 100), 100)
        
        # Absolute Lockout Guardrail: If Tree/Water/Built is overwhelmingly high, crush P1.
        if top_class in ['trees', 'water', 'built', 'snow_and_ice', 'flooded_vegetation'] and top_prob > 0.65:
            # Sigmoid Scaling allows a strong NDVI signal to "fight back" against weak noise.
            scale_factor = (top_prob - 0.65) / (1.0 - 0.65)
            dw_confidence = min(dw_confidence + (100 - dw_confidence) * (scale_factor ** 0.5), 100)
            
        p2_nocrop_conf = dw_confidence
        p2_crop_conf = 100 - dw_confidence

    # --- RESOLUTION: CONSENSUS LOGIC ---    
    if p2_crop_conf > 80: # AI is certain
        final_crop = p2_crop_conf * 0.7 + p1_crop_conf * 0.3
    elif p1_crop_conf > 80: # Physics is certain
        final_crop = p1_crop_conf * 0.8 + p2_crop_conf * 0.2
    else:
        # If neither is certain, they must support each other
        # Merge Phase (Weighted 60% Physics, 40% AI)
        final_crop = (p1_crop_conf * 0.60) + (p2_crop_conf * 0.40)
    
    final_nocrop = 100 - final_crop
    
    prediction = "Crop-Activity" if final_crop > final_nocrop else "No Crop-Activity"
    return prediction, p1_crop_conf, p2_crop_conf, final_crop


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

        # --- D. WHITTAKER SMOOTHER & DAILY UPSAMPLING ---
        dataset_df.set_index('date', inplace=True)
        # Handle duplicated dates by taking mean
        dataset_df = dataset_df.groupby('date').mean(numeric_only=True)
        # Resample strictly to daily points to hit exactly 365 days
        dataset_df = dataset_df.resample('D').mean(numeric_only=True)
        dataset_df.reset_index(inplace=True)

        def whittaker_smooth(y, lambda_=100, d=2):
            import numpy as np
            from scipy.sparse import eye, diags
            from scipy.sparse.linalg import spsolve
            m = len(y)
            if m < d + 1: return y
            weight = np.ones(m)
            weight[np.isnan(y)] = 0.0
            W = diags(weight, 0, shape=(m, m), format='csc')
            D = eye(m, format='csc')
            for _ in range(d):
                D = D[:-1, :] - D[1:, :]
            y_solve = np.copy(y)
            if np.isnan(y_solve).all(): return y_solve
            y_solve[np.isnan(y_solve)] = 0.0 
            A = W + (lambda_ * D.T.dot(D))
            return spsolve(A, W.dot(y_solve))

        # Apply Adaptive Whittaker parameters
        if 'NDVI' in dataset_df.columns:
            # Aggressive short-cycle kernel for peak preservation
            dataset_df['NDVI_short_smooth'] = whittaker_smooth(dataset_df['NDVI'].values, lambda_=10)
            # Default smoothing for standard extraction
            dataset_df['NDVI'] = whittaker_smooth(dataset_df['NDVI'].values, lambda_=50)
            
        if 'EVI' in dataset_df.columns:
            dataset_df['EVI'] = whittaker_smooth(dataset_df['EVI'].values, lambda_=50)
            
        if 'RVI' in dataset_df.columns:
            # Stiffness mode to combat radar speckle
            dataset_df['RVI'] = whittaker_smooth(dataset_df['RVI'].values, lambda_=200)

        # For Dynamic World discrete classes, standard linear interpolation is sufficient
        cols_to_fix = [c for c in dw_cols if c in dataset_df.columns]
        dataset_df[cols_to_fix] = dataset_df[cols_to_fix].interpolate(method='linear', limit_direction='both').fillna(0)

        # E. Final Sanitize & Apply Logic
        dataset_df = dataset_df.replace([np.inf, -np.inf], np.nan).fillna(0)
                        
        # We no longer need aggressive rolling windows because Whittaker has effectively smoothed it perfectly
        dataset_df['NDVI_smooth_slope'] = dataset_df['NDVI']
        dataset_df['RVI_smooth_slope'] = dataset_df['RVI']
        
        # Pre-calculate slopes (Daily points = fine-grained derivatives)
        dataset_df['NDVI_slope'] = np.gradient(dataset_df['NDVI_smooth_slope'])
        dataset_df['RVI_slope'] = np.gradient(dataset_df['RVI_smooth_slope'])
        
        cycle_info = detect_crop_cycles(dataset_df)

        # Step 2: THEN apply logic using correct cycle info
        results = dataset_df.apply(
            lambda row: apply_empirical_logic(row, cycle_info['detected_seasons']), axis=1
        )
        dataset_df['prediction'] = [r[0] for r in results]
        dataset_df['p1_crop_conf'] = [r[1] for r in results]
        dataset_df['p2_crop_conf'] = [r[2] for r in results]
        dataset_df['final_confidence'] = [r[3] for r in results]
        dataset_df['p1_nocrop_conf'] = 100 - dataset_df['p1_crop_conf']
        dataset_df['p2_nocrop_conf'] = 100 - dataset_df['p2_crop_conf']
        
        # ============================================================
        # YEAR-LONG ENVIRONMENTAL GUARDBAND
        # If the geography averages massive non-crop noise probabilities over the full 365 days,
        # it is physically impossible to be an active arable crop farm.
        # ============================================================
        dominant_classes = dataset_df[['trees', 'water', 'built', 'shrub_and_scrub', 'grass', 'crops', 'flooded_vegetation', 'bare', 'snow_and_ice']].idxmax(axis=1)
        tree_freq = (dominant_classes == 'trees').mean()
        water_freq = (dominant_classes == 'water').mean()
        built_freq = (dominant_classes == 'built').mean()
        crop_freq = (dominant_classes == 'crops').mean() + (dominant_classes == 'flooded_vegetation').mean()
        non_crop_freq = max(tree_freq, water_freq, built_freq, snow_freq, flooded_freq)
        crop_days = int((dataset_df['prediction'] == "Crop-Activity").sum())
        total_days = len(dataset_df)
        activity_ratio = (crop_days / total_days) if total_days > 0 else 0

        # Absolute Veto logic triggers - Occams Razor: Protect cycles from AI blindness
        total_cycles = cycle_info['total_cycles']
        is_guardband_triggered = (tree_freq > 0.60 or water_freq > 0.60 or built_freq > 0.50 or snow_freq > 0.60 or flooded_freq > 0.60)
        if total_cycles == 0:
            is_guardband_triggered = is_guardband_triggered or (crop_freq < 0.10)
        
        if is_guardband_triggered:
            if non_crop_freq > 0.85 and activity_ratio > 0.0:
                # Scenario C: Extreme AI Domination
                dataset_df['p2_crop_conf'] *= 0.10
                if total_cycles == 0: dataset_df['p1_crop_conf'] *= 0.10
            elif activity_ratio > 0.50 and non_crop_freq > 0.50:
                # Scenario B: Major Conflict
                dataset_df['p2_crop_conf'] *= 0.30
                if total_cycles == 0: dataset_df['p1_crop_conf'] *= 0.30
            else:
                # Scenario A: Total Agreement (Standard Lockout)
                dataset_df['p2_crop_conf'] = 0.0
                if total_cycles == 0: dataset_df['p1_crop_conf'] = 0.0

            # Re-evaluate row-level confidence dynamically for penalized rows
            dataset_df['p1_nocrop_conf'] = 100.0 - dataset_df['p1_crop_conf']
            dataset_df['p2_nocrop_conf'] = 100.0 - dataset_df['p2_crop_conf']
            
            # Recalculate row predictions based on user instruction: "Whichever is higher"
            avg_crop = (dataset_df['p1_crop_conf'] * 0.60) + (dataset_df['p2_crop_conf'] * 0.40)
            avg_nocrop = (dataset_df['p1_nocrop_conf'] * 0.60) + (dataset_df['p2_nocrop_conf'] * 0.40)
            
            # Override baseline prediction if structural physics are sound (P1 > 50) and verified by peaks (cycles > 0)
            dataset_df['prediction'] = np.where(
                (dataset_df['p1_crop_conf'] > 50) & (total_cycles > 0),
                "Crop-Activity",
                np.where(avg_crop > avg_nocrop, "Crop-Activity", "No Crop-Activity")
            )
            dataset_df['final_confidence'] = np.where(avg_crop > avg_nocrop, avg_crop, avg_nocrop)

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
        
        # Determine bar colors based on binary classification
        bar_colors = activity_binary.map({1: '#27ae60', 0: '#e74c3c'})
        
        # Calculate height: if it's a crop, use final_confidence. If no crop, use 100 - final_confidence.
        bar_heights = [conf if pred == 'Crop-Activity' else (100 - conf) 
                       for pred, conf in zip(predictions['prediction'], predictions['final_confidence'])]
        
        fig.add_trace(go.Bar(
            x=predictions["date_str"],
            y=bar_heights,
            marker_color=bar_colors,
            name="Confidence"
        ))

        fig.update_layout(
            title="Verdict Certainty Over Time",
            yaxis=dict(
                title="Certainty (%)",
                range=[0, 105],
                gridcolor='rgba(0,0,0,0.05)'
            ),
            xaxis=dict(
                range=[
                    analysis_start.strftime("%Y-%m-%d"),
                    analysis_end.strftime("%Y-%m-%d"),
                ],
                title="Date",
                tickangle=-45,
            ),
            template="plotly_white",
            showlegend=False,
            margin=dict(l=50, r=20, t=50, b=50),
            height=400,
        )

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

        # 4-Axis Synthesis for Overall Parcel Confidence
        p1_crop_mean = dataset_df['p1_crop_conf'].mean()
        p1_nocrop_mean = dataset_df['p1_nocrop_conf'].mean()
        p2_crop_mean = dataset_df['p2_crop_conf'].mean()
        p2_nocrop_mean = dataset_df['p2_nocrop_conf'].mean()

        final_crop_score = (p1_crop_mean * 0.60) + (p2_crop_mean * 0.40)
        final_nocrop_score = (p1_nocrop_mean * 0.60) + (p2_nocrop_mean * 0.40)

        # Re-verify activity ratio after possible guardband prediction flips
        current_crop_days = int((dataset_df['prediction'] == "Crop-Activity").sum())
        current_activity_ratio = (current_crop_days / len(dataset_df) * 100) if len(dataset_df) > 0 else 0
        overall_conf = final_crop_score if current_activity_ratio >= 15 else final_nocrop_score

        # --- UPDATED RETURN STATEMENT ---
        return {
            "land use/ land cover details": summary_dict,
            "vegetation_peak_analysis": peak_data,        # Added peak data
            "missing_data": missing_periods,
            "seasonal_activity": seasonal_summary,
            "crop_activity_predictions_list": predictions_list,
            "crop_activity_prediction_stats": {
                "total": len(predictions),
                "crop_days": int(sum(activity_binary)),
                "overall_confidence": float(overall_conf),
                "p1_avg_conf": float(dataset_df['p1_crop_conf'].mean()),
                "p2_avg_conf": float(dataset_df['p2_crop_conf'].mean()),
                "p1_nocrop_avg_conf": float(dataset_df['p1_nocrop_conf'].mean()),
                "p2_nocrop_avg_conf": float(dataset_df['p2_nocrop_conf'].mean())
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
            "metadata": {
                "coords": coords 
            },
            "crop_cycles_count": cycle_info["total_cycles"],
            "detected_seasons": cycle_info["detected_seasons"],
            "cycle_details": cycle_info["details"],
        }
    except Exception as e:
            print(f"Engine Error: {e}")
            return None
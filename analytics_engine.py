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
import traceback
from data_loader import create_test_data, process_parcel_data
from test_model import predict_from_pickle
from datetime import datetime
from data_loader import detect_crop_cycles

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 100)      # Show more rows

ACTIVE_ACTIVITY_THRESHOLD = 30.0
NON_CROP_DOMINANCE_THRESHOLD = 60.0
LOW_CROP_PROBABILITY_THRESHOLD = 25.0
NON_CROP_DOMINANT_CLASSES = {
    'water', 'trees', 'grass', 'flooded_vegetation',
    'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
}
DW_COLS = [
    'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
    'shrub_and_scrub', 'built', 'bare', 'snow_and_ice',
]

def has_noncrop_dominance_veto(land_cover_df):
    if land_cover_df is None:
        return False
    if isinstance(land_cover_df, list):
        if not land_cover_df:
            return False
        land_cover_df = pd.DataFrame(land_cover_df)
    elif isinstance(land_cover_df, pd.DataFrame):
        if land_cover_df.empty:
            return False
    else:
        return False

    dw_cols = [
        'water', 'trees', 'grass', 'flooded_vegetation',
        'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
    ]
    available_cols = [c for c in dw_cols if c in land_cover_df.columns]
    if not available_cols:
        return False

    numeric_df = land_cover_df[available_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    dominant_classes = numeric_df.idxmax(axis=1)
    dominant_share = dominant_classes.value_counts(normalize=True)
    dominant_class = dominant_share.index[0] if not dominant_share.empty else None
    dominant_percent = float(dominant_share.iloc[0] * 100) if not dominant_share.empty else 0.0
    crop_probability_mean = float(numeric_df['crops'].mean() * 100) if 'crops' in numeric_df else 0.0

    if (
        dominant_class in NON_CROP_DOMINANT_CLASSES
        and dominant_percent >= NON_CROP_DOMINANCE_THRESHOLD
        and crop_probability_mean <= LOW_CROP_PROBABILITY_THRESHOLD
    ):
        return True

    if 'grass' in numeric_df.columns:
        grass_day_share = float((dominant_classes == 'grass').mean() * 100)
        if (
            grass_day_share >= NON_CROP_DOMINANCE_THRESHOLD
            and crop_probability_mean <= LOW_CROP_PROBABILITY_THRESHOLD
            and dominant_class not in ('trees', 'water')
        ):
            return True

    return False


def whittaker_smooth(y, lambda_=100, d=2):
    from scipy.sparse import eye, diags
    from scipy.sparse.linalg import spsolve
    m = len(y)
    if m < d + 1:
        return y
    weight = np.ones(m)
    weight[np.isnan(y)] = 0.0
    W = diags(weight, 0, shape=(m, m), format='csc')
    D = eye(m, format='csc')
    for _ in range(d):
        D = D[:-1, :] - D[1:, :]
    y_solve = np.copy(y)
    if np.isnan(y_solve).all():
        return y_solve
    y_solve[np.isnan(y_solve)] = 0.0
    A = W + (lambda_ * D.T.dot(D))
    return spsolve(A, W.dot(y_solve))


def apply_whittaker_smoothing(dataset_df):
    if 'NDVI' in dataset_df.columns:
        dataset_df['NDVI_short_smooth'] = whittaker_smooth(dataset_df['NDVI'].values, lambda_=10)
        dataset_df['NDVI'] = whittaker_smooth(dataset_df['NDVI'].values, lambda_=50)
    if 'EVI' in dataset_df.columns:
        dataset_df['EVI'] = whittaker_smooth(dataset_df['EVI'].values, lambda_=50)
    if 'RVI' in dataset_df.columns:
        dataset_df['RVI'] = whittaker_smooth(dataset_df['RVI'].values, lambda_=200)
    return dataset_df


def prepare_daily_modeling_dataframe(df_veg_sparse, df_dw_raw, already_smoothed=False):
    """Merge sparse vegetation with DW, daily resample, optional single Whittaker pass."""
    veg = df_veg_sparse.copy()
    dw = df_dw_raw.copy()
    veg['date'] = pd.to_datetime(veg['date']).dt.normalize()
    dw['date'] = pd.to_datetime(dw['date']).dt.normalize()

    index_cols = [c for c in ['NDVI', 'EVI', 'RVI'] if c in veg.columns]
    raw_points_df = (
        veg[['date'] + index_cols]
        .dropna(subset=index_cols, how='all')
        .drop_duplicates(subset=['date'])
        .sort_values('date')
    )

    veg_clean = veg.drop(columns=[c for c in DW_COLS if c in veg.columns])
    dataset_df = veg_clean.merge(dw, on='date', how='outer').sort_values('date')
    dataset_df.set_index('date', inplace=True)
    dataset_df = dataset_df.groupby('date').mean(numeric_only=True)
    dataset_df = dataset_df.resample('D').mean(numeric_only=True)
    dataset_df.reset_index(inplace=True)

    if not already_smoothed:
        apply_whittaker_smoothing(dataset_df)

    cols_to_fix = [c for c in DW_COLS if c in dataset_df.columns]
    if cols_to_fix:
        dataset_df[cols_to_fix] = (
            dataset_df[cols_to_fix]
            .interpolate(method='linear', limit_direction='both')
            .fillna(0)
        )
    dataset_df = dataset_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return dataset_df, raw_points_df


def build_vegetation_indices_chart(smoothed_df, raw_df, date_range):
    """Whittaker-smoothed daily lines plus sparse GEE scene markers."""
    analysis_start, analysis_end = date_range
    line_styles = {
        'NDVI': ('green', 'NDVI (smoothed)'),
        'EVI': ('blue', 'EVI (smoothed)'),
        'RVI': ('orange', 'RVI (smoothed)'),
    }
    marker_styles = {
        'NDVI': ('darkgreen', 'NDVI (observed)'),
        'EVI': ('darkblue', 'EVI (observed)'),
        'RVI': ('darkorange', 'RVI (observed)'),
    }

    fig = go.Figure()
    plot_df = smoothed_df.copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'])

    for col, (color, name) in line_styles.items():
        if col in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df['date'],
                    y=plot_df[col],
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                    connectgaps=False,
                )
            )

    if raw_df is not None and not raw_df.empty:
        raw_plot = raw_df.copy()
        raw_plot['date'] = pd.to_datetime(raw_plot['date'])
        for col, (color, name) in marker_styles.items():
            if col in raw_plot.columns:
                observed = raw_plot.dropna(subset=[col])
                if not observed.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=observed['date'],
                            y=observed[col],
                            mode='markers',
                            name=name,
                            marker=dict(
                                color=color,
                                size=5,
                                symbol='circle',
                                opacity=0.75,
                            ),
                        )
                    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Index Value",
        template="plotly_white",
        xaxis=dict(
            type="date",
            range=[
                pd.Timestamp(analysis_start).strftime("%Y-%m-%d"),
                pd.Timestamp(analysis_end).strftime("%Y-%m-%d"),
            ],
            tickformat="%b %Y",
            dtick="M1",
        ),
        yaxis=dict(range=[-0.1, 1.0]),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.6)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        height=400,
    )
    return fig


def certainty_tier_from_score(score):
    if score >= 80:
        return "Very High"
    if score >= 65:
        return "High"
    if score >= 50:
        return "Moderate"
    if score >= 35:
        return "Low"
    return "Very Low"


def _smoothstep_certainty(raw_score):
    nx = min((raw_score / 100.0) / 0.85, 1.0)
    return float((nx * nx * (3.0 - 2.0 * nx)) * 100.0)


def compute_parcel_certainty(
    dataset_df,
    raw_indices_df,
    is_active,
    noncrop_veto,
    final_crop_score,
    final_nocrop_score,
    crop_freq,
    non_crop_freq,
    is_guardband_triggered,
    guardband_conflict=False,
    missing_periods=None,
):
    if is_active:
        # For active parcels, we expect crops during peaks and no-crops during off-seasons.
        # So daily winning confidence is the confidence of the predicted state on that day.
        daily_winning = np.maximum(
            dataset_df['p1_crop_conf'] * 0.60 + dataset_df['p2_crop_conf'] * 0.40,
            dataset_df['p1_nocrop_conf'] * 0.60 + dataset_df['p2_nocrop_conf'] * 0.40
        )
        daily_losing = 100.0 - daily_winning
        winning_score = float(daily_winning.mean())
        losing_score = float(daily_losing.mean())
        verdict_margin = abs(winning_score - losing_score)
    else:
        # For inactive parcels, we expect no-crops all year round.
        winning_score = final_nocrop_score
        losing_score = final_crop_score
        verdict_margin = abs(winning_score - losing_score)

    pipeline_disagreement = float(dataset_df['p1_crop_conf'].sub(dataset_df['p2_crop_conf']).abs().mean())
    pipeline_agreement = max(0.0, 100.0 - pipeline_disagreement)

    if raw_indices_df is not None and not raw_indices_df.empty and 'NDVI' in raw_indices_df.columns:
        n_s2_scenes = int(raw_indices_df['NDVI'].notna().sum())
    else:
        n_s2_scenes = 0
    # Realistic data sufficiency: 20 scenes per year is 100% sufficient for Whittaker
    data_sufficiency = min(100.0, 100.0 * (n_s2_scenes / 20.0))
    if missing_periods:
        data_sufficiency *= max(0.5, 1.0 - (len(missing_periods) * 0.05))

    if is_active:
        # Scaled crop frequency: 25% crop dominance over the year is highly sufficient for an active land.
        land_alignment = min(100.0, (crop_freq / 0.25) * 80.0 + (1.0 - non_crop_freq) * 20.0)
    else:
        land_alignment = min(100.0, (non_crop_freq * 100.0) + (20.0 if noncrop_veto else 0.0))

    composite = (
        0.35 * verdict_margin
        + 0.25 * pipeline_agreement
        + 0.20 * data_sufficiency
        + 0.20 * land_alignment
    )

    penalties = []
    activity_ratio = (
        (dataset_df['prediction'] == "Crop-Activity").mean() * 100.0
        if len(dataset_df) > 0
        else 0.0
    )
    if abs(activity_ratio - ACTIVE_ACTIVITY_THRESHOLD) <= 5.0:
        composite *= 0.75
        penalties.append("borderline_activity")
    if noncrop_veto and activity_ratio > ACTIVE_ACTIVITY_THRESHOLD:
        composite *= 0.6
        penalties.append("noncrop_veto_conflict")
    if is_guardband_triggered and guardband_conflict:
        composite *= 0.7
        penalties.append("guardband_conflict")
    if n_s2_scenes < 12:
        composite *= 0.8
        penalties.append("sparse_optical")

    certainty_score = max(15.0, _smoothstep_certainty(composite))
    return {
        "certainty_score": certainty_score,
        "certainty_tier": certainty_tier_from_score(certainty_score),
        "components": {
            "verdict_margin": round(verdict_margin, 2),
            "pipeline_agreement": round(pipeline_agreement, 2),
            "data_sufficiency": round(data_sufficiency, 2),
            "land_alignment": round(land_alignment, 2),
            "n_s2_scenes": n_s2_scenes,
        },
        "penalties_applied": penalties,
    }


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
    
    # 1. Prediction remains strictly linear
    prediction = "Crop-Activity" if final_crop > final_nocrop else "No Crop-Activity"
    
    # 2. Confidence uses Geometric Purity
    import math
    p1_winning = p1_crop_conf if prediction == "Crop-Activity" else p1_nocrop_conf
    p2_winning = p2_crop_conf if prediction == "Crop-Activity" else p2_nocrop_conf
    
    base_confidence = math.sqrt(p1_winning * p2_winning)
    
    if prediction == "Crop-Activity":
        noise_cols = [c for c in ['trees', 'water', 'built', 'shrub_and_scrub'] if c in row.index]
        noise_level = sum([row.get(c, 0.0) for c in noise_cols]) if noise_cols else 0.0
        # Use Quadratic Purity: Small edge-noise (30%) only penalizes 9%, but heavy noise (80%) penalizes 64%
        purity = max(1.0 - (noise_level ** 2), 0.1)
    else:
        purity = 1.0
        
    raw_conf = base_confidence * purity
    
    # Smoothstep stretching to push Highs higher and Lows lower
    nx = min((raw_conf / 100.0) / 0.85, 1.0)
    final_confidence = (nx * nx * (3.0 - 2.0 * nx)) * 100.0
    
    return prediction, p1_crop_conf, p2_crop_conf, final_confidence


def run_full_analytics_pipeline(task_id, coords, end_date_str):
    try:
        # 1. Extraction (GEE)
        extraction_result = process_parcel_data(task_id, coords, end_date_str)
        if not extraction_result:
            return None
        
        df_all, summary_dict, df_dw_raw = extraction_result

        # --- CAPTURE TRULY RAW GEE-FETCHED DATA ---
        # df_all at this point contains the original S2 (NDVI/EVI) and S1 (RVI) rows
        # exactly as returned by Earth Engine, before any resampling, outer-join with DW,
        # or Whittaker smoothing. This is the earliest possible capture point.
        _raw_index_cols = [c for c in ['NDVI', 'EVI', 'RVI'] if c in df_all.columns]
        raw_indices_df = (
            df_all[['date'] + _raw_index_cols]
            .copy()
            .assign(date=lambda d: pd.to_datetime(d['date']).dt.normalize())
            .dropna(subset=_raw_index_cols, how='all')
            .drop_duplicates(subset=['date'])
            .sort_values('date')
        )

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

        df_all_clean = df_all.drop(columns=[c for c in DW_COLS if c in df_all.columns])
        dataset_df, _ = prepare_daily_modeling_dataframe(
            df_all_clean, df_dw_raw, already_smoothed=False
        )
                        
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
            title="Daily Classification Strength",
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

        fig_indices = build_vegetation_indices_chart(
            test_df, raw_indices_df, (analysis_start, analysis_end)
        )
        ndvi_base64 = fig_to_base64(fig_indices, is_plotly=True)

        print(seasonal_summary)

        raw_indices_export = raw_indices_df.copy()
        raw_indices_export['date'] = pd.to_datetime(raw_indices_export['date']).dt.strftime('%Y-%m-%d')
        raw_indices_list = raw_indices_export.to_dict(orient="records")

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
        noncrop_veto = has_noncrop_dominance_veto(df_dw_raw)
        is_active = current_activity_ratio > ACTIVE_ACTIVITY_THRESHOLD and not noncrop_veto
        certainty = compute_parcel_certainty(
            dataset_df,
            raw_indices_df,
            is_active,
            noncrop_veto,
            final_crop_score,
            final_nocrop_score,
            crop_freq,
            non_crop_freq,
            is_guardband_triggered,
            guardband_conflict=guardband_conflict,
            missing_periods=missing_periods,
        )
        overall_conf = certainty["certainty_score"]

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
                "certainty_score": float(certainty["certainty_score"]),
                "certainty_tier": certainty["certainty_tier"],
                "certainty_components": certainty["components"],
                "certainty_penalties": certainty["penalties_applied"],
                "is_active": bool(is_active),
                "noncrop_dominance_veto": bool(noncrop_veto),
                "p1_avg_conf": float(dataset_df['p1_crop_conf'].mean()),
                "p2_avg_conf": float(dataset_df['p2_crop_conf'].mean()),
                "p1_nocrop_avg_conf": float(dataset_df['p1_nocrop_conf'].mean()),
                "p2_nocrop_avg_conf": float(dataset_df['p2_nocrop_conf'].mean())
            },
            "timeseries_data": {
                "vegetation_indices": indices_raw_list,     # Smoothed daily NDVI, EVI, RVI points
                "raw_vegetation_indices": raw_indices_list, # Sparse GEE scene observations for raw markers
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
            print(traceback.format_exc())
            return None
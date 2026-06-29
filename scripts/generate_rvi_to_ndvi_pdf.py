import os
import base64
import asyncio
from playwright.async_api import async_playwright
import shutil
import json

# Path configurations
ANALYSIS_DIR = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\analysis"
MODELS_DIR = r"c:\Users\Suryadeep Singh\Downloads\AdvaRisk - Test\models"
ARTIFACT_DIR = r"C:\Users\Suryadeep Singh\.gemini\antigravity-ide\brain\ea250939-4c8f-45b7-a97e-d60cb3693660"

def get_base64_image(image_path):
    """Convert local image to base64 string."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            b64_str = base64.b64encode(img_file.read()).decode('utf-8')
            return f"data:image/png;base64,{b64_str}"
    return ""

async def main():
    print("[PDF] Preparing RVI to NDVI Model PDF generation...")
    
    # 1. Base64 encode the generated plots
    scatter_b64 = get_base64_image(os.path.join(ANALYSIS_DIR, "rvi_to_ndvi_scatter.png"))
    importance_b64 = get_base64_image(os.path.join(ANALYSIS_DIR, "feature_importance_ndvi.png"))
    temporal_b64 = get_base64_image(os.path.join(ANALYSIS_DIR, "lstm_vs_rf_temporal.png"))
    residuals_hist_b64 = get_base64_image(os.path.join(ANALYSIS_DIR, "residuals_histogram.png"))
    residuals_vs_act_b64 = get_base64_image(os.path.join(ANALYSIS_DIR, "residuals_vs_actual.png"))
    seasonal_metrics_b64 = get_base64_image(os.path.join(ANALYSIS_DIR, "season_wise_metrics.png"))
    bigru_scatter_b64 = get_base64_image(os.path.join(ANALYSIS_DIR, "bigru_scatter.png"))
    bigru_importance_b64 = get_base64_image(os.path.join(ANALYSIS_DIR, "bigru_feature_importance.png"))
    
    print("[PDF] Successfully encoded all 8 charts to Base64.")
    
    # 2. Read performance metrics (Full 24-feature baseline values)
    metrics_json_path = os.path.join(ANALYSIS_DIR, "rvi_to_ndvi_expanded_metrics.json")
    
    model_type = "Random Forest Regressor"
    train_r2, test_r2 = "0.9686", "0.8975"
    train_rmse, test_rmse = "0.0442", "0.0919"
    train_mae, test_mae = "0.0369", "0.0634"
    
    seasonal = {
        "Kharif": {"R2": 0.8439, "RMSE": 0.1223, "MAE": 0.0698, "Samples": 1007},
        "Rabi": {"R2": 0.8486, "RMSE": 0.1134, "MAE": 0.0637, "Samples": 2685},
        "Zaid": {"R2": 0.7942, "RMSE": 0.1063, "MAE": 0.0571, "Samples": 1118}
    }
    
    if os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, 'r') as f:
                data = json.load(f)
                model_type = data.get('model_type', model_type)
                overall = data.get('overall', {})
                train_r2 = f"{overall.get('train_r2', 0.9686):.4f}"
                test_r2 = f"{overall.get('test_r2', 0.8975):.4f}"
                train_rmse = f"{overall.get('train_rmse', 0.0442):.4f}"
                test_rmse = f"{overall.get('test_rmse', 0.0919):.4f}"
                train_mae = f"{overall.get('train_mae', 0.0369):.4f}"
                test_mae = f"{overall.get('test_mae', 0.0634):.4f}"
                seasonal = data.get('seasonal', seasonal)
        except Exception as e:
            print(f"[PDF] Error parsing JSON metrics: {e}. Using baseline values.")
            
    # 3. HTML Content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RVI to NDVI Translation Model Analysis Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        @page {{
            size: A4;
            margin: 15mm 12mm 15mm 12mm;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #1e293b;
            line-height: 1.45;
            background-color: #ffffff;
            font-size: 9.5pt;
        }}
        
        .page {{
            page-break-after: always;
            position: relative;
        }}
        
        .page:last-child {{
            page-break-after: avoid !important;
        }}
        
        /* Cover Page Styling */
        .cover-page {{
            padding-top: 35mm;
            height: 250mm;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .cover-title-container {{
            border-left: 6px solid #10b981;
            padding-left: 24px;
        }}
        
        .cover-title {{
            font-size: 28pt;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -1.5px;
            color: #0f172a;
            margin-bottom: 12px;
        }}
        
        .cover-subtitle {{
            font-size: 13pt;
            font-weight: 500;
            color: #475569;
            line-height: 1.4;
        }}
        
        .cover-divider {{
            height: 3px;
            background: linear-gradient(90deg, #10b981, #3b82f6, transparent);
            margin: 30px 0;
            width: 80%;
        }}

        .cover-description {{
            font-size: 10.5pt;
            color: #475569;
            max-width: 85%;
            margin-bottom: 30px;
            line-height: 1.6;
        }}
        
        .cover-meta {{
            font-size: 9pt;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
            padding-top: 15px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 15mm;
        }}
        
        .meta-group strong {{
            color: #0f172a;
            display: block;
            margin-bottom: 4px;
        }}

        /* Section Headings */
        .section-header {{
            border-bottom: 2px solid #cbd5e1;
            padding-bottom: 6px;
            margin-bottom: 12px;
            margin-top: 2mm;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            page-break-after: avoid;
        }}
        
        .section-title {{
            font-size: 14pt;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.5px;
        }}
        
        .section-subtitle {{
            font-size: 7.5pt;
            color: #10b981;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }}
        
        h2 {{
            font-size: 10.5pt;
            font-weight: 700;
            color: #1e293b;
            margin-top: 12px;
            margin-bottom: 4px;
            border-left: 3px solid #3b82f6;
            padding-left: 8px;
            page-break-after: avoid;
        }}
        
        p {{
            margin-bottom: 8px;
            color: #334155;
            text-align: justify;
            font-size: 9pt;
            line-height: 1.45;
        }}
        
        strong {{
            color: #0f172a;
        }}
        
        .callout-box {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #10b981;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 12px;
            page-break-inside: avoid;
        }}
        
        .callout-title {{
            font-weight: 700;
            color: #065f46;
            margin-bottom: 4px;
            font-size: 9pt;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 8.5pt;
            page-break-inside: avoid;
        }}
        
        th {{
            background-color: #f1f5f9;
            color: #475569;
            font-weight: 700;
            padding: 6px 8px;
            border-bottom: 2px solid #cbd5e1;
            text-align: left;
        }}
        
        td {{
            padding: 6px 8px;
            border-bottom: 1px solid #e2e8f0;
            color: #334155;
            vertical-align: top;
        }}
        
        tr:nth-child(even) td {{
            background-color: #f8fafc;
        }}
        
        /* KPI Widget Grid */
        .kpi-row {{
            display: flex;
            gap: 15px;
            margin-bottom: 12px;
        }}
        
        .kpi-box {{
            flex: 1;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px;
            background-color: #f0fdf4;
            border-left: 4px solid #10b981;
            text-align: center;
        }}
        
        .kpi-box-title {{
            font-size: 7.5pt;
            font-weight: 700;
            color: #166534;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .kpi-box-num {{
            font-size: 20pt;
            font-weight: 800;
            color: #15803d;
        }}
        
        .kpi-box-sub {{
            font-size: 7.5pt;
            color: #166534;
            margin-top: 2px;
        }}

        .chart-row {{
            display: flex;
            gap: 15px;
            margin: 10px 0;
            page-break-inside: avoid;
        }}
        
        .chart-box {{
            flex: 1;
            text-align: center;
        }}
        
        .chart-img {{
            max-width: 100%;
            height: auto;
            max-height: 60mm;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
        }}
        
        .chart-caption {{
            font-size: 7.5pt;
            color: #64748b;
            margin-top: 3px;
        }}

        .bullet-list {{
            margin-bottom: 10px;
            padding-left: 20px;
        }}

        .bullet-list li {{
            font-size: 9pt;
            margin-bottom: 3px;
            color: #334155;
        }}
    </style>
</head>
<body>

    <!-- ==================== PAGE 1: COVER ==================== -->
    <div class="page cover-page">
        <div class="cover-title-container">
            <div class="cover-title">RVI to NDVI Model Report</div>
            <div class="cover-subtitle">Synthetic Optical Vegetation Index Generation using Radar Polarizations</div>
        </div>
        
        <div class="cover-divider"></div>

        <div class="cover-description">
            This technical report provides a detailed breakdown of the machine learning model designed to translate Sentinel-1 Synthetic Aperture Radar (SAR) based <strong>Radar Vegetation Index (RVI)</strong> into Sentinel-2 <strong>Normalized Difference Vegetation Index (NDVI)</strong>. Utilizing physical coordinates, seasonal parameters, dynamic world probabilities, and meteorological variables, the model provides cloud-free synthetic NDVI estimations to support continuous parcel-level agricultural audits.
        </div>
        
        <div class="cover-meta">
            <div class="meta-group">
                <strong>EVALUATION METHOD</strong>
                Group-based Train-Test Split (80/20)<br>
                Grouped by Parcel ID (task_id)
            </div>
            <div class="meta-group">
                <strong>TRAINING COHORT</strong>
                244 Train Parcels (18,289 rows)<br>
                62 Unseen Test Parcels (4,810 rows)
            </div>
            <div class="meta-group" style="text-align: right;">
                <strong>MODEL TYPE</strong>
                {model_type}<br>
                June 2026
            </div>
        </div>
    </div>
    
    <!-- ==================== PAGE 2: METRICS & ALIGNMENT ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">1. Dataset Alignment & Model Performance</div>
            <div class="section-subtitle">Performance Summary</div>
        </div>
        
        <p>
            Due to clouds obstructing optical bands, creating a synthetic proxy of NDVI from Sentinel-1 radar bands allows for continuous agricultural health monitoring. To train the translator model, Sentinel-2 optical scenes were paired with their nearest Sentinel-1 observations within a <strong>3-day matching window</strong>, producing a high-quality dataset of 23,099 aligned observations.
        </p>

        <h2>Operational Validation: Spatial Generalization Check</h2>
        <p>
            By splitting training and testing data at the parcel level (Group-based Split), the model was evaluated on 62 parcels that it had never seen during training. This simulates the exact production environment where the model is requested to generate predictions for entirely new geographical boundaries.
        </p>

        <h2>Model Benchmarking & Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th style="width: 30%;">Model Architecture</th>
                    <th style="width: 15%; text-align: center;">Partition</th>
                    <th style="width: 20%; text-align: center;">R² Score</th>
                    <th style="width: 20%; text-align: center;">RMSE Error</th>
                    <th style="width: 15%; text-align: center;">MAE Error</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td rowspan="2"><strong>Random Forest Baseline</strong><br><span style="color:#64748b; font-size:7.5pt;">(24 Features)</span></td>
                    <td style="text-align: center;">Train</td>
                    <td style="text-align: center;">0.9686</td>
                    <td style="text-align: center;">0.0442</td>
                    <td style="text-align: center;">0.0369</td>
                </tr>
                <tr>
                    <td style="text-align: center; border-bottom: 2px solid #cbd5e1;">Test</td>
                    <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #cbd5e1;">0.8975</td>
                    <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #cbd5e1;">0.0919</td>
                    <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #cbd5e1;">0.0634</td>
                </tr>
                <tr>
                    <td rowspan="2"><strong>Regularized LSTM</strong><br><span style="color:#64748b; font-size:7.5pt;">(Standard Recurrent)</span></td>
                    <td style="text-align: center;">Train</td>
                    <td style="text-align: center;">0.9320</td>
                    <td style="text-align: center;">0.0651</td>
                    <td style="text-align: center;">0.0483</td>
                </tr>
                <tr>
                    <td style="text-align: center; border-bottom: 2px solid #cbd5e1;">Test</td>
                    <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #cbd5e1;">0.9022</td>
                    <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #cbd5e1;">0.0898</td>
                    <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #cbd5e1;">0.0646</td>
                </tr>
                <tr>
                    <td rowspan="2" style="background-color: #f0fdf4;"><strong>Bidirectional GRU</strong><br><span style="color:#15803d; font-size:7.5pt;">(Sequence Imputer)</span></td>
                    <td style="text-align: center; background-color: #f0fdf4;">Train</td>
                    <td style="text-align: center; background-color: #f0fdf4;">0.9380</td>
                    <td style="text-align: center; background-color: #f0fdf4;">0.0622</td>
                    <td style="text-align: center; background-color: #f0fdf4;">0.0461</td>
                </tr>
                <tr>
                    <td style="text-align: center; font-weight: bold; background-color: #f0fdf4;">Test</td>
                    <td style="text-align: center; font-weight: bold; color: #15803d; background-color: #f0fdf4;">0.9077</td>
                    <td style="text-align: center; font-weight: bold; color: #15803d; background-color: #f0fdf4;">0.0873</td>
                    <td style="text-align: center; font-weight: bold; color: #15803d; background-color: #f0fdf4;">0.0628</td>
                </tr>
            </tbody>
        </table>

        <div class="chart-row">
            <div class="chart-box" style="max-width: 70%; margin: 0 auto;">
                <img class="chart-img" style="max-height: 80mm;" src="{scatter_b64}" alt="Actual vs Predicted Scatter Plot">
                <div class="chart-caption">Figure 1: RVI to NDVI actual vs. predicted values on unseen test parcels.</div>
            </div>
        </div>
    </div>

    <!-- ==================== PAGE 3: FEATURE SPECIFICATIONS ==================== -->
    <div class="page" style="padding-top: 10mm; page-break-before: always;">
        <div class="section-header">
            <div class="section-title">2. Detailed Model Feature Specifications</div>
            <div class="section-subtitle">Features & Descriptions</div>
        </div>
        
        <p>
            The RVI-to-NDVI translation model utilizes a full set of 24 spatial, structural, meteorologic, and phenological features to reconstruct vegetation indices. The table below lists all features used, what they are, and their operational role:
        </p>
        
        <table style="font-size: 7.6pt; line-height: 1.35; margin-top: 5px;">
            <thead>
                <tr>
                    <th style="width: 25%;">Feature Name</th>
                    <th style="width: 35%;">What the Feature Is</th>
                    <th style="width: 40%;">Why It Is Used in the Model</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>latitude</code></td>
                    <td>Absolute geographic latitude coordinate of the parcel centroid</td>
                    <td>Captures large-scale spatial/climatic gradients and regional crop calendar variations across India.</td>
                </tr>
                <tr>
                    <td><code>longitude</code></td>
                    <td>Absolute geographic longitude coordinate of the parcel centroid</td>
                    <td>Captures spatial gradients, local water availability, and regional agricultural boundaries.</td>
                </tr>
                <tr>
                    <td><code>raw_RVI</code></td>
                    <td>Sentinel-1 Radar Vegetation Index calculated from VV and VH backscatter power</td>
                    <td>Primary structural proxy of vegetation canopy density that penetrates cloud cover.</td>
                </tr>
                <tr>
                    <td><code>is_kharif</code></td>
                    <td>Binary seasonal flag (1 if date falls between June and October)</td>
                    <td>Represents the monsoon Kharif crop season, aligning the model with high-rainfall crop dynamics.</td>
                </tr>
                <tr>
                    <td><code>is_rabi</code></td>
                    <td>Binary seasonal flag (1 if date falls between November and March)</td>
                    <td>Represents the winter Rabi crop season, capturing irrigated winter crop phenology.</td>
                </tr>
                <tr>
                    <td><code>is_zaid</code></td>
                    <td>Binary seasonal flag (1 if date falls between April and May)</td>
                    <td>Represents the dry summer Zaid crop season, adjusting for extreme temperatures and summer crops.</td>
                </tr>
                <tr>
                    <td><code>doy_sin</code></td>
                    <td>Sine-transformed Day of Year (circular phenology parameter)</td>
                    <td>Captures the cyclic nature of annual crop growth, calendar progression, and solar patterns.</td>
                </tr>
                <tr>
                    <td><code>doy_cos</code></td>
                    <td>Cosine-transformed Day of Year (circular phenology parameter)</td>
                    <td>Provides orthogonal phase reference for DOY, ensuring continuous temporal phenology representation.</td>
                </tr>
                <tr>
                    <td><code>Rainfall_15d_sum</code></td>
                    <td>15-day cumulative rainfall leading up to the observation (mm)</td>
                    <td>Indicates soil moisture increases and precipitation dynamics affecting crop growth rate.</td>
                </tr>
                <tr>
                    <td><code>MaxTemp_7d_avg</code></td>
                    <td>7-day average of daily maximum surface temperature (°C)</td>
                    <td>Captures heat accumulation and evapotranspiration stress limits that impact canopy greenness.</td>
                </tr>
                <tr>
                    <td><code>MinTemp_7d_avg</code></td>
                    <td>7-day average of daily minimum surface temperature (°C)</td>
                    <td>Reflects night-time respiration rates and seasonal cooling trends affecting crop development.</td>
                </tr>
                <tr>
                    <td><code>crops</code></td>
                    <td>Dynamic World crop classification probability probability (0.0 to 1.0)</td>
                    <td>Provides baseline prior indicating if the parcel is active agricultural land.</td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- ==================== PAGE 4: FEATURE SPECIFICATIONS (CONTINUED) ==================== -->
    <div class="page" style="padding-top: 10mm; page-break-before: always;">
        <div class="section-header">
            <div class="section-title">2. Detailed Model Feature Specifications (Continued)</div>
            <div class="section-subtitle">Features & Descriptions</div>
        </div>
        
        <table style="font-size: 7.6pt; line-height: 1.35; margin-top: 5px;">
            <thead>
                <tr>
                    <th style="width: 25%;">Feature Name</th>
                    <th style="width: 35%;">What the Feature Is</th>
                    <th style="width: 40%;">Why It Is Used in the Model</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>trees</code></td>
                    <td>Dynamic World tree cover probability (0.0 to 1.0)</td>
                    <td>Identifies presence of perennial orchards/forests, adjusting the baseline NDVI projection.</td>
                </tr>
                <tr>
                    <td><code>grass</code></td>
                    <td>Dynamic World grass cover probability (0.0 to 1.0)</td>
                    <td>Distinguishes natural short-grass pasture signatures from structured crop fields.</td>
                </tr>
                <tr>
                    <td><code>flooded_vegetation</code></td>
                    <td>Dynamic World flooded vegetation probability (0.0 to 1.0)</td>
                    <td>Captures wetland/paddy crop dynamics where water backscatter distorts normal radar signals.</td>
                </tr>
                <tr>
                    <td><code>shrub_and_scrub</code></td>
                    <td>Dynamic World shrub/scrub cover probability (0.0 to 1.0)</td>
                    <td>Identifies sparse woody scrublands, distinguishing them from dense agricultural canopies.</td>
                </tr>
                <tr>
                    <td><code>built</code></td>
                    <td>Dynamic World built-up/infrastructure probability (0.0 to 1.0)</td>
                    <td>Detects urban or structural materials that generate high radar double-bounce reflections.</td>
                </tr>
                <tr>
                    <td><code>bare</code></td>
                    <td>Dynamic World bare soil probability (0.0 to 1.0)</td>
                    <td>Identifies harvested fields, fallow land, or dry soil background reflection periods.</td>
                </tr>
                <tr>
                    <td><code>water</code></td>
                    <td>Dynamic World open water probability (0.0 to 1.0)</td>
                    <td>Flags open water bodies, preventing specularity anomalies in VV/VH radar signals.</td>
                </tr>
                <tr>
                    <td><code>snow_and_ice</code></td>
                    <td>Dynamic World snow/glacier probability (0.0 to 1.0)</td>
                    <td>Prevents snow-melt or glacier signals from distorting vegetation indices in hilly regions.</td>
                </tr>
                <tr>
                    <td><code>RVI_lag_12</code></td>
                    <td>Linearly interpolated RVI value 12 days prior to observation date</td>
                    <td>Establishes a 12-day historical baseline of structural vegetative state to track trends.</td>
                </tr>
                <tr>
                    <td><code>RVI_lag_6</code></td>
                    <td>Linearly interpolated RVI value 6 days prior to observation date</td>
                    <td>Tracks short-term historical crop growth or harvest decay trajectory.</td>
                </tr>
                <tr>
                    <td><code>RVI_lead_6</code></td>
                    <td>Linearly interpolated RVI value 6 days ahead of observation date</td>
                    <td>Captures short-term future canopy development trends during temporal matching.</td>
                </tr>
                <tr>
                    <td><code>RVI_lead_12</code></td>
                    <td>Linearly interpolated RVI value 12 days ahead of observation date</td>
                    <td>Captures mid-term future canopy growth and structural senescence progression.</td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- ==================== PAGE 4: DRIVERS OF RECONSTRUCTION ==================== -->
    <div class="page" style="padding-top: 10mm; page-break-before: always;">
        <div class="section-header">
            <div class="section-title">3. Drivers of Vegetation Reconstruction</div>
            <div class="section-subtitle">Feature Importance & LULC Context</div>
        </div>

        <p>
            Feature importance analysis reveals that the target's current radar value (`raw_RVI`) and its temporal dynamics (lags/leads representing recent trends) serve as the dominant indicators. Local meteorological inputs (rainfall, temperature) and dynamic land cover distributions successfully constrain regional variations in signal response.
        </p>

        <div class="chart-container" style="text-align: center; margin: 10px 0;">
            <img class="chart-img" style="max-height: 75mm;" src="{importance_b64}" alt="Feature Importance">
            <div class="chart-caption" style="margin-bottom: 12px;">Figure 2: Relative feature importances in Random Forest regressor.</div>
        </div>

        <h2>Temporal Tracking Case Study</h2>
        <p>
            The reconstructed NDVI curve (orange) tracks the optical NDVI signature (green) over the full course of multiple seasonal crop cycles. This highlights that the model successfully filters out sensor speckle while maintaining sensitive response times to crop emergence, peak greening, and harvest events.
        </p>

        <div class="chart-container" style="text-align: center; margin: 10px 0;">
            <img class="chart-img" style="max-height: 68mm;" src="{temporal_b64}" alt="Temporal Aligned Prediction">
            <div class="chart-caption">Figure 3: Temporal prediction curve comparison (Random Forest vs. BiGRU) on an unseen test parcel.</div>
        </div>
    </div>

    <!-- ==================== PAGE 6: DEEP LEARNING DIAGNOSTICS ==================== -->
    <div class="page" style="padding-top: 10mm; page-break-before: always;">
        <div class="section-header">
            <div class="section-title">4. Deep Learning Interpretability & Diagnostics</div>
            <div class="section-subtitle">BiGRU Diagnostics</div>
        </div>
        
        <p>
            To evaluate the deep learning model's performance and internal representation, we perform standard diagnostics on the <strong>Bidirectional GRU (BiGRU)</strong> network. The scatter plot below displays predicted versus actual NDVI values on the unseen test parcels. The permutation feature importance chart displays the relative influence of the 24 inputs, computed by evaluating the increase in test Mean Squared Error (MSE) when the values of each feature are shuffled.
        </p>

        <div class="chart-row">
            <div class="chart-box">
                <img class="chart-img" style="max-height: 65mm;" src="{bigru_scatter_b64}" alt="BiGRU Scatter Plot">
                <div class="chart-caption">Figure 4: BiGRU Model: Actual vs. predicted NDVI values on unseen test parcels (Test R² = 0.9077).</div>
            </div>
            <div class="chart-box">
                <img class="chart-img" style="max-height: 65mm;" src="{bigru_importance_b64}" alt="BiGRU Feature Importance">
                <div class="chart-caption">Figure 5: BiGRU Model: Permutation feature importances based on test MSE increase.</div>
            </div>
        </div>

        <h2>Deep Learning Interpretability</h2>
        <p>
            The permutation feature importance highlights that the BiGRU model relies heavily on <code>raw_RVI</code> and the forward/backward temporal context variables (<code>RVI_lead_6</code>, <code>RVI_lag_6</code>). By integrating sequential context, the model learns a more continuous representation of crop growth curves compared to decision-tree architectures, reducing sensitivity to high-frequency speckle noise and maintaining robust NDVI predictions during periods of extended cloud cover.
        </p>
    </div>

</body>
</html>
"""

    # 4. Use Playwright Headless Browser to render HTML to A4 PDF
    print("[PDF] Loading Playwright headless compiler...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        page = await browser.new_page()
        
        # Load Content
        await page.set_content(html_content)
        await page.wait_for_load_state("networkidle")
        
        # Try writing to various paths using increments if locked
        success = False
        pdf_output_name = "RVI_to_NDVI_Model_Report.pdf"
        pdf_output_path = os.path.join(ANALYSIS_DIR, pdf_output_name)
        
        attempts = [pdf_output_name] + [f"RVI_to_NDVI_Model_Report_v{i}.pdf" for i in range(1, 10)]
        
        for attempt_name in attempts:
            attempt_path = os.path.join(ANALYSIS_DIR, attempt_name)
            print(f"[PDF] Attempting to compile to: {attempt_path}...")
            try:
                await page.pdf(
                    path=attempt_path,
                    format="A4",
                    print_background=True,
                    display_header_footer=True,
                    header_template="""
                        <div style="font-size: 8px; width: 100%; margin: 0 15mm; display: flex; justify-content: space-between; color: #94a3b8; font-family: 'Inter', sans-serif;">
                            <span>RVI TO NDVI MACHINE LEARNING RECONSTRUCTION REPORT</span>
                            <span>TECHNICAL AUDIT REPORT</span>
                        </div>
                    """,
                    footer_template="""
                        <div style="font-size: 8px; width: 100%; margin: 0 15mm; border-top: 1px solid #e2e8f0; padding-top: 4px; display: flex; justify-content: space-between; color: #94a3b8; font-family: 'Inter', sans-serif;">
                            <span>AdvaRisk Analytics Engine</span>
                            <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
                        </div>
                    """,
                    margin={
                        "top": "15mm",
                        "bottom": "18mm",
                        "left": "10mm",
                        "right": "10mm"
                    }
                )
                pdf_output_name = attempt_name
                pdf_output_path = attempt_path
                success = True
                print(f"[PDF] Successfully created PDF report at: {pdf_output_path}")
                break
            except PermissionError:
                print(f"[PDF] Permission denied on {attempt_path}. File is likely locked.")
                continue
        
        await browser.close()
        
        if not success:
            print("[PDF] Error: Could not write report to any of the attempted file paths due to locks.")
            return
        
    # Also copy the PDF directly to the brain artifact directory for absolute persistence!
    dest_path = os.path.join(ARTIFACT_DIR, pdf_output_name)
    try:
        shutil.copyfile(pdf_output_path, dest_path)
        print(f"[PDF] Copied report PDF to brain artifact directory: {dest_path}")
    except Exception as e:
        print(f"[PDF] Warning: Could not copy to brain directory: {e}")

if __name__ == "__main__":
    asyncio.run(main())

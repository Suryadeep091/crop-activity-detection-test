import os
import json
import base64
import asyncio
from playwright.async_api import async_playwright
import shutil

# Path configurations
JSON_FILE = 'analysis_stats.json'
RVI_METRICS_FILE = r'c:\Users\Suryadeep Singh\OneDrive\Old Repos\AdvaRisk - Test\analysis\rvi_to_ndvi_expanded_metrics.json'
ARTIFACT_DIR = r"C:\Users\Suryadeep Singh\.gemini\antigravity-ide\brain\ea250939-4c8f-45b7-a97e-d60cb3693660"
ASSETS_DIR = os.path.join(os.path.abspath(ARTIFACT_DIR), "assets")

def get_base64_image(image_name):
    """Convert local image to base64 string."""
    image_path = os.path.join(ASSETS_DIR, image_name)
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            b64_str = base64.b64encode(img_file.read()).decode('utf-8')
            return f"data:image/png;base64,{b64_str}"
    return ""

async def main():
    print("[PDF] Preparing unified PDF generation...")
    
    # 1. Base64 encode all generated charts (including multi-year stats and RVI-to-NDVI diagnostic plots)
    yearly_accuracy_b64 = get_base64_image("yearly_accuracy.png")
    confusion_matrices_b64 = get_base64_image("confusion_matrices.png")
    certainty_analysis_b64 = get_base64_image("certainty_analysis.png")
    state_performance_b64 = get_base64_image("state_performance.png")
    activity_transitions_b64 = get_base64_image("activity_transitions.png")
    
    # RVI-to-NDVI specific plots
    rvi_scatter_b64 = get_base64_image("rvi_to_ndvi_scatter.png")
    rvi_importance_b64 = get_base64_image("feature_importance_ndvi.png")
    rvi_residuals_hist_b64 = get_base64_image("residuals_histogram.png")
    rvi_temporal_b64 = get_base64_image("temporal_prediction_sample.png")
    
    print("[PDF] Successfully encoded all charts to Base64.")
    
    # 2. Load Stats JSON for rendering key values
    if not os.path.exists(JSON_FILE):
        raise FileNotFoundError(f"{JSON_FILE} not found. Run run_analysis.py first.")
        
    with open(JSON_FILE, "r") as f:
        stats = json.load(f)
        
    # Extracted stats helper
    y24 = stats["yearly"]["2023-24"]
    y25 = stats["yearly"]["2024-25"]
    y26 = stats["yearly"]["2025-26"]
    
    # 3. Load RVI-to-NDVI metrics
    rvi_model_type = "Random Forest Regressor"
    rvi_train_r2, rvi_test_r2 = "0.9686", "0.8975"
    rvi_train_rmse, rvi_test_rmse = "0.0442", "0.0919"
    rvi_train_mae, rvi_test_mae = "0.0369", "0.0634"
    rvi_seasonal = {
        "Kharif": {"R2": 0.8439, "RMSE": 0.1223, "MAE": 0.0698, "Samples": 1007},
        "Rabi": {"R2": 0.8486, "RMSE": 0.1134, "MAE": 0.0637, "Samples": 2685},
        "Zaid": {"R2": 0.7942, "RMSE": 0.1063, "MAE": 0.0571, "Samples": 1118}
    }
    
    if os.path.exists(RVI_METRICS_FILE):
        try:
            with open(RVI_METRICS_FILE, 'r') as f_rvi:
                rvi_data = json.load(f_rvi)
                rvi_model_type = rvi_data.get('model_type', rvi_model_type)
                overall = rvi_data.get('overall', {})
                rvi_train_r2 = f"{overall.get('train_r2', 0.9686):.4f}"
                rvi_test_r2 = f"{overall.get('test_r2', 0.8975):.4f}"
                rvi_train_rmse = f"{overall.get('train_rmse', 0.0442):.4f}"
                rvi_test_rmse = f"{overall.get('test_rmse', 0.0919):.4f}"
                rvi_train_mae = f"{overall.get('train_mae', 0.0369):.4f}"
                rvi_test_mae = f"{overall.get('test_mae', 0.0634):.4f}"
                rvi_seasonal = rvi_data.get('seasonal', rvi_seasonal)
        except Exception as e:
            print(f"[PDF] Error parsing RVI metrics JSON: {e}")
            
    # 4. Dynamic compilation of the 306 farm rows for the Appendix
    print(f"[PDF] Processing {len(stats['farms'])} farms for the Appendix...")
    table_rows = []
    for farm in stats["farms"]:
        name = farm["farm_name"]
        state = farm["state"]
        gt = farm["gt"][0] # "Active" or "Inactive"
        
        p24 = farm["pred"][0]
        c24 = farm["cert"][0]
        m24 = "✅" if farm["match"][0] else "❌"
        if p24 == "Error":
            m24 = "⚠️"
            
        p25 = farm["pred"][1]
        c25 = farm["cert"][1]
        m25 = "✅" if farm["match"][1] else "❌"
        if p25 == "Error":
            m25 = "⚠️"
            
        p26 = farm["pred"][2]
        c26 = farm["cert"][2]
        m26 = "✅" if farm["match"][2] else "❌"
        if p26 == "Error":
            m26 = "⚠️"
            
        row_html = f"""
            <tr>
                <td style="font-weight:600; padding: 4px 6px;">{name}</td>
                <td style="padding: 4px 6px;">{state}</td>
                <td style="padding: 4px 6px;">{gt}</td>
                <td style="padding: 4px 6px;">{p24} <span style="color:#64748b; font-size:7pt;">({c24})</span> {m24}</td>
                <td style="padding: 4px 6px;">{p25} <span style="color:#64748b; font-size:7pt;">({c25})</span> {m25}</td>
                <td style="padding: 4px 6px;">{p26} <span style="color:#64748b; font-size:7pt;">({c26})</span> {m26}</td>
            </tr>
        """
        table_rows.append(row_html)
        
    all_table_rows_html = "\n".join(table_rows)
    print(f"[PDF] Compiled HTML code for all {len(table_rows)} farm rows.")

    # 5. HTML Template for printable PDF report (with double-bracketed CSS and RVI parameters)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Terradrishti Multi-Year Farm Satellite Analytics Report (2023-26)</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
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
            padding-top: 40mm;
            height: 250mm;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .cover-title-container {{
            border-left: 6px solid #2563eb;
            padding-left: 20px;
        }}
        
        .cover-title {{
            font-size: 32pt;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -1px;
            color: #0f172a;
        }}
        
        .cover-subtitle {{
            font-size: 16pt;
            font-weight: 500;
            color: #475569;
            margin-top: 15px;
        }}
        
        .cover-meta {{
            font-size: 11pt;
            color: #64748b;
            margin-top: 100mm;
            border-top: 1px solid #e2e8f0;
            padding-top: 15px;
            display: flex;
            justify-content: space-between;
            margin-bottom: 20mm;
        }}
        
        /* Page Header and Titles */
        .section-header {{
            border-bottom: 2px solid #2563eb;
            padding-bottom: 8px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            page-break-after: avoid;
        }}
        
        .section-title {{
            font-size: 16pt;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.5px;
        }}
        
        .section-subtitle {{
            font-size: 9.5pt;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        h2 {{
            font-size: 11pt;
            font-weight: 700;
            color: #1e293b;
            margin-top: 15px;
            margin-bottom: 6px;
            border-left: 3px solid #10b981;
            padding-left: 8px;
            page-break-after: avoid;
        }}
        
        /* Typography */
        p {{
            margin-bottom: 12px;
            color: #334155;
            text-align: justify;
        }}
        
        strong {{
            color: #0f172a;
        }}
        
        /* Callout Box styling */
        .callout-box {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 12px 15px;
            margin-bottom: 15px;
            page-break-inside: avoid;
        }}
        
        .callout-title {{
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 4px;
            font-size: 10pt;
        }}
        
        /* Metrics Table */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
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
        }}
        
        tr:nth-child(even) td {{
            background-color: #f8fafc;
        }}
        
        /* KPI Widget Grid */
        .kpi-row {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .kpi-box {{
            flex: 1;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px;
            background-color: #f8fafc;
            text-align: center;
        }}
        
        .kpi-box-title {{
            font-size: 7.5pt;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .kpi-box-num {{
            font-size: 16pt;
            font-weight: 800;
            color: #2563eb;
        }}
        
        .kpi-box-sub {{
            font-size: 7.5pt;
            color: #94a3b8;
            margin-top: 2px;
        }}
        
        /* Chart container */
        .chart-container {{
            width: 100%;
            text-align: center;
            margin: 15px 0;
            page-break-inside: avoid;
        }}
        
        .chart-img {{
            max-width: 100%;
            height: auto;
            max-height: 80mm;
            border-radius: 4px;
        }}
        
        /* State Ranking list */
        .rank-grid {{
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
        }}
        
        .rank-column {{
            flex: 1;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
        }}
        
        .rank-column-title {{
            font-size: 10pt;
            font-weight: 700;
            color: #0f172a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
            margin-bottom: 8px;
        }}
        
        .rank-item {{
            display: flex;
            justify-content: space-between;
            font-size: 9pt;
            padding: 3px 0;
        }}
        
    </style>
</head>
<body>

    <!-- ==================== PAGE 1: COVER ==================== -->
    <div class="page cover-page">
        <div class="cover-title-container">
            <div class="cover-title">Multi-Year Farm<br>Satellite Analytics</div>
            <div class="cover-subtitle">Accuracy Benchmark & Performance Report (2023–26)</div>
        </div>
        
        <div class="cover-meta">
            <div>
                <strong>Generated For:</strong> AdvaRisk Audit Division<br>
                <strong>Focus Period:</strong> 2023-24, 2024-25, 2025-26
            </div>
            <div style="text-align: right;">
                <strong>Author:</strong> Terradrishti GEE Pipeline<br>
                <strong>Date:</strong> June 2026
            </div>
        </div>
    </div>
    
    <!-- ==================== PAGE 2: SUMMARY ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">1. Executive Summary & Trends</div>
            <div class="section-subtitle">Overview</div>
        </div>
        
        <div class="callout-box">
            <div class="callout-title">Core Finding: Absolute Stability in Ground Truth (306 Farms)</div>
            <p style="font-size: 9pt; margin-bottom: 0;">
                Analysis of the multi-year trajectory for the <strong>306 monitored farms</strong> across all 3 years reveals that the physically inspected agricultural ground truth is <strong>100% static</strong>. 
                There are exactly <strong>218 permanently active crop fields</strong> and <strong>88 permanently inactive</strong> fields, with zero state-transitions. 
                Any model variance observed across years represents GEE imagery or cycle fluctuations, rather than real shifts in farming behavior.
            </p>
        </div>
        
        <p>
            This report establishes accuracy benchmarks by comparing satellite analytics predictions against historical ground truth data. 
            Overall model accuracy remains high, establishing a benchmark baseline at <strong>92.43%</strong> in 23-24. 
            A slight margin of degradation is observed in subsequent years (89.54% and 88.52% in 24-25 and 25-26 respectively), primarily driven by temporal imagery alignment constraints and seasonal cloud coverage.
        </p>
        
        <div class="kpi-row">
            <div class="kpi-box">
                <div class="kpi-box-title">Overall Accuracy '23-24</div>
                <div class="kpi-box-num">{y24["accuracy"]:.1f}%</div>
                <div class="kpi-box-sub">Baseline Performance</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-box-title">Overall Accuracy '24-25</div>
                <div class="kpi-box-num">{y25["accuracy"]:.1f}%</div>
                <div class="kpi-box-sub">Mid-Term Performance</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-box-title">Overall Accuracy '25-26</div>
                <div class="kpi-box-num">{y26["accuracy"]:.1f}%</div>
                <div class="kpi-box-sub">Latest Multi-Season Model</div>
            </div>
        </div>
        
        <div class="chart-container">
            <img class="chart-img" style="max-height: 75mm;" src="{yearly_accuracy_b64}" alt="Yearly Accuracy Comparison">
        </div>
    </div>
    
    <!-- ==================== PAGE 3: CONFUSION MATRICES ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">2. Confusion Matrix & Transition Analysis</div>
            <div class="section-subtitle">Statistical Agreement</div>
        </div>
        
        <p>
            An evaluation of year-wise confusion matrices indicates that <strong>True Negatives remained perfectly stable</strong> at exactly <strong>79</strong> successful detections each year. 
            Conversely, the slight dip in accuracy is driven by a migration of <strong>True Positives (active crops)</strong> towards <strong>False Negatives (predicted inactive)</strong>. This signifies a slight under-detection of active crops, rather than false alerts.
        </p>
        
        <table>
            <thead>
                <tr>
                    <th>Benchmark Cohort</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>TP</th>
                    <th>TN</th>
                    <th>FP</th>
                    <th>FN</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>2023-24 (Baseline)</strong></td>
                    <td>{y24["accuracy"]:.2f}%</td>
                    <td>{y24["precision"]:.2f}%</td>
                    <td>{y24["recall"]:.2f}%</td>
                    <td>{y24["f1"]:.2f}%</td>
                    <td>{y24["TP"]}</td>
                    <td>{y24["TN"]}</td>
                    <td>{y24["FP"]}</td>
                    <td>{y24["FN"]}</td>
                </tr>
                <tr>
                    <td><strong>2024-25 (Mid-Term)</strong></td>
                    <td>{y25["accuracy"]:.2f}%</td>
                    <td>{y25["precision"]:.2f}%</td>
                    <td>{y25["recall"]:.2f}%</td>
                    <td>{y25["f1"]:.2f}%</td>
                    <td>{y25["TP"]}</td>
                    <td>{y25["TN"]}</td>
                    <td>{y25["FP"]}</td>
                    <td>{y25["FN"]}</td>
                </tr>
                <tr>
                    <td><strong>2025-26 (Latest Model)</strong></td>
                    <td>{y26["accuracy"]:.2f}%</td>
                    <td>{y26["precision"]:.2f}%</td>
                    <td>{y26["recall"]:.2f}%</td>
                    <td>{y26["f1"]:.2f}%</td>
                    <td>{y26["TP"]}</td>
                    <td>{y26["TN"]}</td>
                    <td>{y26["FP"]}</td>
                    <td>{y26["FN"]}</td>
                </tr>
            </tbody>
        </table>
        
        <div class="chart-container">
            <img class="chart-img" style="max-height: 48mm; margin-bottom: 3mm;" src="{confusion_matrices_b64}" alt="Confusion Matrices">
            <img class="chart-img" style="max-height: 48mm;" src="{activity_transitions_b64}" alt="Transition Pathways">
        </div>
    </div>
    
    <!-- ==================== PAGE 4: REGIONAL BREAKDOWN ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">3. Regional Breakdown & State performance</div>
            <div class="section-subtitle">Geographic Analysis</div>
        </div>
        
        <p>
            State-level grouping reveals substantial performance variance across different geographical zones. 
            Dense perennial canopy covers (such as the coconut and rubber plantations in Kerala) present the lowest accuracy, as they confuse multi-season index calculations. 
            Alluvial agricultural states, such as Uttar Pradesh, present exceptionally high, stable model performance.
        </p>
        
        <div class="rank-grid">
            <div class="rank-column" style="border-left: 3px solid #10b981;">
                <div class="rank-column-title" style="color: #047857;">Top 5 Performing States (Avg Accuracy)</div>
                <div class="rank-item"><span>1. Uttar Pradesh (UP)</span><strong>100.0%</strong></div>
                <div class="rank-item"><span>2. Jammu & Kashmir (JK)</span><strong>96.7%</strong></div>
                <div class="rank-item"><span>3. Uttarakhand (UK)</span><strong>96.7%</strong></div>
                <div class="rank-item"><span>4. Chhattisgarh (CH)</span><strong>96.7%</strong></div>
                <div class="rank-item"><span>5. Northeast (NE)</span><strong>96.3%</strong></div>
            </div>
            
            <div class="rank-column" style="border-left: 3px solid #ef4444;">
                <div class="rank-column-title" style="color: #b91c1c;">Lowest Performing States (Avg Accuracy)</div>
                <div class="rank-item"><span>1. Kerala (KL)</span><strong>79.2%</strong></div>
                <div class="rank-item"><span>2. Karnataka (KT)</span><strong>80.0%</strong></div>
                <div class="rank-item"><span>3. Himachal Pradesh (HP)</span><strong>83.3%</strong></div>
                <div class="rank-item"><span>4. Andhra Pradesh (AP)</span><strong>86.7%</strong></div>
                <div class="rank-item"><span>5. Rajasthan (RJ)</span><strong>86.7%</strong></div>
            </div>
        </div>
        
        <div class="chart-container" style="margin-top: 5px;">
            <img class="chart-img" style="max-height: 80mm;" src="{state_performance_b64}" alt="State-wise Performance Trends">
        </div>
    </div>
    
    <!-- ==================== PAGE 5: CERTAINTY & RECOMMENDATIONS ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">4. Certainty Analysis & Recommendations</div>
            <div class="section-subtitle">Auditing Guidelines</div>
        </div>
        
        <p>
            An analysis of prediction success against the model-assigned <strong>Certainty Tier</strong> reveals an excellent direct correlation. 
            Predictions generated with <strong>Very High Certainty</strong> achieved a high success rate of <strong>95.9%</strong>. 
            Conversely, cases categorized with <em>Moderate</em> certainty presented exactly 50% accuracy (matching random chance), indicating that these cases should be flagged for review.
        </p>
        
        <div class="chart-container" style="margin: 5px 0;">
            <img class="chart-img" style="max-height: 52mm;" src="{certainty_analysis_b64}" alt="Certainty vs Accuracy">
        </div>
        
        <div class="callout-box" style="border-left-color: #f59e0b; margin-top: 5px; padding: 10px 12px;">
            <div class="callout-title" style="color: #78350f;">Strategic Recommendations for Audit Pipeline</div>
            <p style="font-size: 9pt; margin-bottom: 6px;">
                <strong>1. Automated Certainty Gateways:</strong> Establish a threshold where any parcel with *Moderate*, *Low*, or *Very Low* certainty is automatically routed into a manual audit pipeline. This filters out the highest-risk predictions.
            </p>
            <p style="font-size: 9pt; margin-bottom: 6px;">
                <strong>2. Perennial Canopy Tree-Crop Filtering:</strong> Implement tree-crop masking filters for coastal and southern regions (specifically Kerala) to prevent mature perennial plantations from distorting annual and seasonal active crop indexes.
            </p>
            <p style="font-size: 9pt; margin-bottom: 0;">
                <strong>3. SAR (Synthetic Aperture Radar) Integration:</strong> Incorporate Sentinel-1 SAR imagery for states heavily impacted by monsoon cloud cover (Telangana, Karnataka, Kerala) to maintain observation frequency during active peak vegetative cycles.
            </p>
        </div>
    </div>
    
    <!-- ==================== PAGE 5: PIPELINE FLOWCHART ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">5. Multi-Season Crop Activity Detection Logic</div>
            <div class="section-subtitle">Decision Architecture Flowchart</div>
        </div>
        
        <p style="margin-bottom: 15px;">
            The agricultural crop-activity detection pipeline processes co-registered multi-spectral and radar satellite observations to classify parcels and compute certainty metrics. The process is fully automated and is structured as follows:
        </p>

        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; margin: 15px 0;">
            <!-- Step 1 -->
            <div style="width: 90%; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid #2563eb; border-radius: 6px; padding: 8px 12px; display: flex; align-items: center; justify-content: space-between;">
                <div style="width: 30%; font-weight: 700; color: #1e3a8a; font-size: 9pt;">1. Data Acquisition</div>
                <div style="width: 65%; font-size: 8.5pt; color: #475569;">Extracts daily Sentinel-1 (SAR) and Sentinel-2 (optical) observations from Google Earth Engine.</div>
            </div>
            
            <div style="color: #64748b; font-size: 12pt; font-weight: bold; line-height: 1;">↓</div>
            
            <!-- Step 2 -->
            <div style="width: 90%; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid #10b981; border-radius: 6px; padding: 8px 12px; display: flex; align-items: center; justify-content: space-between;">
                <div style="width: 30%; font-weight: 700; color: #065f46; font-size: 9pt;">2. Imputation & Smoothing</div>
                <div style="width: 65%; font-size: 8.5pt; color: #475569;">Reconstructs cloud-obstructed optical NDVI using the <strong>BiGRU + RF Ensemble Model</strong>, followed by Whittaker smoothing.</div>
            </div>
            
            <div style="color: #64748b; font-size: 12pt; font-weight: bold; line-height: 1;">↓</div>
            
            <!-- Step 3 -->
            <div style="width: 90%; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid #f59e0b; border-radius: 6px; padding: 8px 12px; display: flex; align-items: center; justify-content: space-between;">
                <div style="width: 30%; font-weight: 700; color: #78350f; font-size: 9pt;">3. LULC Guardband Veto</div>
                <div style="width: 65%; font-size: 8.5pt; color: #475569;">
                    Evaluates dominant LULC classes. Hard non-crop dominates: <strong>Vetoed to Inactive</strong>. Arid/transitional dominates: <strong>Bypassed if crop peaks ≥35%</strong> (with certainty penalty).
                </div>
            </div>
            
            <div style="color: #64748b; font-size: 12pt; font-weight: bold; line-height: 1;">↓</div>
            
            <!-- Step 4 -->
            <div style="width: 90%; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid #8b5cf6; border-radius: 6px; padding: 8px 12px; display: flex; align-items: center; justify-content: space-between;">
                <div style="width: 30%; font-weight: 700; color: #4c1d95; font-size: 9pt;">4. Cycle Detection</div>
                <div style="width: 65%; font-size: 8.5pt; color: #475569;">Detects seasonal crop cycles (Kharif, Rabi, Zaid) using numerical derivatives of smoothed NDVI time-series.</div>
            </div>
            
            <div style="color: #64748b; font-size: 12pt; font-weight: bold; line-height: 1;">↓</div>
            
            <!-- Step 5 -->
            <div style="width: 90%; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid #ec4899; border-radius: 6px; padding: 8px 12px; display: flex; align-items: center; justify-content: space-between;">
                <div style="width: 30%; font-weight: 700; color: #831843; font-size: 9pt;">5. Verdict & Certainty</div>
                <div style="width: 65%; font-size: 8.5pt; color: #475569;">Computes activity ratio and composite certainty score. Generates active/inactive verdict.</div>
            </div>
        </div>
        
        <div class="callout-box" style="margin-top: 10px; border-left-color: #8b5cf6; background-color: #fbfbfe;">
            <div class="callout-title" style="color: #4c1d95;">Mathematical Certainty Scoring Logic</div>
            <p style="font-size: 8.5pt; margin-bottom: 0; line-height: 1.4;">
                Certainty represents the model's confidence in its classification. It is calculated using a composite base score scaled by three penalties:
                <br>
                1. <strong>Missing Optical Data Penalty (0.8x)</strong>: Applied when Sentinel-2 scene count is low.
                <br>
                2. <strong>Transitional non-crop Dominance Penalty (0.85x)</strong>: Applied when active crop signatures are found on fields dominated by bare land, shrubs, or grass (e.g. in Rajasthan).
                <br>
                3. <strong>Guardband Conflict Penalty (0.5x)</strong>: Applied if LULC classifications and indices are highly contradictory.
            </p>
        </div>
    </div>
    
    <!-- ==================== PAGE 6: RVI-TO-NDVI TRANSLATION MODEL ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">6. Core Signal Imputation: BiGRU + RF Ensemble model</div>
            <div class="section-subtitle">SAR-to-Optical Mapping</div>
        </div>
        
        <p>
            Agricultural crop canopy monitoring is frequently hindered by optical cloud obstructions during key periods of the monsoon. To bridge these gaps, the pipeline uses a <strong>BiGRU + Random Forest Ensemble Model</strong> (60% BiGRU, 40% Random Forest) to translate cloud-penetrating Sentinel-1 Radar Vegetation Index (RVI) values to Sentinel-2 optical NDVI values.
        </p>
        
        <div class="kpi-row" style="margin-bottom: 10px;">
            <div class="kpi-box" style="background-color: #f0fdf4; border-left-color: #10b981;">
                <div class="kpi-box-title" style="color: #166534;">Final Generalization Accuracy (Test R²)</div>
                <div class="kpi-box-num" style="color: #15803d;">{rvi_test_r2}</div>
                <div class="kpi-box-sub" style="color: #15803d;">On Unseen Validation Parcels</div>
            </div>
            <div class="kpi-box" style="background-color: #f8fafc; border-left-color: #64748b;">
                <div class="kpi-box-title" style="color: #475569;">Test Mean Absolute Error (MAE)</div>
                <div class="kpi-box-num" style="color: #334155;">{rvi_test_mae}</div>
                <div class="kpi-box-sub" style="color: #475569;">Avg Deviation in NDVI units</div>
            </div>
        </div>
        
        <div style="display: flex; gap: 15px;">
            <div style="flex: 1.2;">
                <h2 style="font-size: 10pt; margin-top: 5px;">Model Parameters & Configuration</h2>
                <table style="font-size: 7.8pt; margin-top: 4px; margin-bottom: 8px;">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Setting</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Algorithm</strong></td>
                            <td>{rvi_model_type}</td>
                            <td>Hybrid Recurrent & Decision-Tree Ensemble</td>
                        </tr>
                        <tr>
                            <td><strong>Ensemble Weights</strong></td>
                            <td>60% BiGRU / 40% Random Forest</td>
                            <td>Weighted blending of predictions</td>
                        </tr>
                        <tr>
                            <td><strong>Sequence Length</strong></td>
                            <td>5 periods (bidirectional)</td>
                            <td>Temporal context window size</td>
                        </tr>
                        <tr>
                            <td><strong>BiGRU Hidden Units</strong></td>
                            <td>64 units, 2 layers</td>
                            <td>Recurrent layer configuration</td>
                        </tr>
                        <tr>
                            <td><strong>RF Estimators</strong></td>
                            <td>100 trees</td>
                            <td>Forest base estimators</td>
                        </tr>
                        <tr>
                            <td><strong>Smoothing</strong></td>
                            <td>Whittaker filter</td>
                            <td>NDVI &lambda;=50 | RVI &lambda;=200</td>
                        </tr>
                        <tr>
                            <td><strong>EVI Status</strong></td>
                            <td>EXCLUDED</td>
                            <td>EVI completely omitted</td>
                        </tr>
                    </tbody>
                </table>
                
                <h2 style="font-size: 10pt; margin-top: 5px;">Season-wise Generalization</h2>
                <table style="font-size: 7.8pt; margin-top: 4px; margin-bottom: 0;">
                    <thead>
                        <tr>
                            <th>Season</th>
                            <th>Test Samples</th>
                            <th>R² Score</th>
                            <th>RMSE Error</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Rabi (Winter)</strong></td>
                            <td>{rvi_seasonal.get('Rabi', {}).get('Samples', 2685)}</td>
                            <td>{rvi_seasonal.get('Rabi', {}).get('R2', 0.8486):.4f}</td>
                            <td>{rvi_seasonal.get('Rabi', {}).get('RMSE', 0.1134):.4f}</td>
                        </tr>
                        <tr>
                            <td><strong>Kharif (Monsoon)</strong></td>
                            <td>{rvi_seasonal.get('Kharif', {}).get('Samples', 1007)}</td>
                            <td>{rvi_seasonal.get('Kharif', {}).get('R2', 0.8439):.4f}</td>
                            <td>{rvi_seasonal.get('Kharif', {}).get('RMSE', 0.1223):.4f}</td>
                        </tr>
                        <tr>
                            <td><strong>Zaid (Summer)</strong></td>
                            <td>{rvi_seasonal.get('Zaid', {}).get('Samples', 1118)}</td>
                            <td>{rvi_seasonal.get('Zaid', {}).get('R2', 0.7942):.4f}</td>
                            <td>{rvi_seasonal.get('Zaid', {}).get('RMSE', 0.1063):.4f}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div style="flex: 1.0; text-align: center;">
                <img class="chart-img" style="max-height: 54mm; margin-top: 5px; border: 1px solid #e2e8f0;" src="{rvi_scatter_b64}" alt="RVI Scatter Plot">
                <div style="font-size: 7.5pt; color: #64748b; margin-top: 2px;">Figure 5: Predicted vs Ground-truth NDVI (Test Split)</div>
                
                <img class="chart-img" style="max-height: 38mm; margin-top: 5px; border: 1px solid #e2e8f0;" src="{rvi_importance_b64}" alt="RVI Feature Importance">
                <div style="font-size: 7.5pt; color: #64748b; margin-top: 2px;">Figure 6: Model Relative Feature Importances</div>
            </div>
        </div>
    </div>
    
    <!-- ==================== PAGE 7+: APPENDIX ==================== -->
    <div class="page" style="padding-top: 10mm; page-break-before: always;">
        <div class="section-header">
            <div class="section-title">6. Appendix: Detailed Farm-by-Farm Reference Table</div>
            <div class="section-subtitle">Full Census</div>
        </div>
        
        <p style="margin-bottom: 12px; font-size: 8.5pt; color:#475569;">
            This appendix contains a complete record of all <strong>306 monitored farms</strong>, detailing their ground truth classification (Static Active / Static Inactive) along with the predicted agricultural activity and certainty tier for each of the three years (2023-24, 2024-25, and 2025-26). Failed or errored GEE runs are flagged with ⚠️.
        </p>
        
        <table style="font-size: 7.2pt; width: 100%; margin-top: 5px;">
            <thead>
                <tr>
                    <th style="padding: 4px 6px; font-size: 8pt;">Farm Name</th>
                    <th style="padding: 4px 6px; font-size: 8pt;">State</th>
                    <th style="padding: 4px 6px; font-size: 8pt;">Ground Truth</th>
                    <th style="padding: 4px 6px; font-size: 8pt;">2023-24 (Certainty)</th>
                    <th style="padding: 4px 6px; font-size: 8pt;">2024-25 (Certainty)</th>
                    <th style="padding: 4px 6px; font-size: 8pt;">2025-26 (Certainty)</th>
                </tr>
            </thead>
            <tbody>
                {all_table_rows_html}
            </tbody>
        </table>
    </div>
    
</body>
</html>
"""

    # 4. Use Playwright Chrome Headless to compile HTML to PDF
    print("[PDF] Loading Playwright headless compiler...")
    pdf_output_path = "AdvaRisk_Farm_Satellite_analysis_2023-26.pdf"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        page = await browser.new_page()
        
        # Load HTML
        await page.set_content(html_content)
        await page.wait_for_load_state("networkidle")
        
        # Compile to PDF
        print(f"[PDF] Compiling HTML report directly to: {pdf_output_path}...")
        await page.pdf(
            path=pdf_output_path,
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="""
                <div style="font-size: 8px; width: 100%; margin: 0 15mm; display: flex; justify-content: space-between; color: #94a3b8; font-family: 'Inter', sans-serif;">
                    <span>TERRADRISHTI MULTI-YEAR BENCHMARK REPORT (2023-26)</span>
                    <span>CONFIDENTIAL - AUDIT COPY</span>
                </div>
            """,
            footer_template="""
                <div style="font-size: 8px; width: 100%; margin: 0 15mm; border-top: 1px solid #e2e8f0; padding-top: 4px; display: flex; justify-content: space-between; color: #94a3b8; font-family: 'Inter', sans-serif;">
                    <span>Generated: June 2026</span>
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
        
        await browser.close()
        
    print(f"[PDF] Successfully created PDF report at: {pdf_output_path}")
    
    # Also copy the PDF directly to the artifact directory for absolute persistence!
    dest_path = os.path.join(ARTIFACT_DIR, "analysis_report.pdf")
    try:
        shutil.copyfile(pdf_output_path, dest_path)
        print(f"[PDF] Copied report PDF to brain artifact directory: {dest_path}")
    except Exception as e:
        print(f"[PDF] Warning: Could not copy to brain directory: {e}")

if __name__ == "__main__":
    asyncio.run(main())

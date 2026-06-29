import asyncio
from playwright.async_api import async_playwright
import os

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TerraDrishti Unified Khasra Engine: System Design & Technical Specification</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        @page {
            size: A4;
            margin: 20mm 15mm 20mm 15mm;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #1e293b;
            line-height: 1.6;
            background-color: #ffffff;
            font-size: 10.5pt;
        }
        
        /* Print layout splitting */
        .page {
            page-break-after: always;
            position: relative;
        }
        
        .page:last-child {
            page-break-after: avoid !important;
        }
        
        /* Cover Page Styling */
        .cover-page {
            padding-top: 40mm;
            height: 250mm;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .cover-title-container {
            border-left: 6px solid #2563eb;
            padding-left: 24px;
        }
        
        .cover-title {
            font-size: 30pt;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -1.5px;
            color: #0f172a;
            margin-bottom: 15px;
        }
        
        .cover-subtitle {
            font-size: 15pt;
            font-weight: 500;
            color: #475569;
            line-height: 1.4;
        }
        
        .cover-divider {
            height: 2px;
            background: linear-gradient(90deg, #2563eb, #10b981, transparent);
            margin: 40px 0;
            width: 80%;
        }

        .cover-description {
            font-size: 11pt;
            color: #64748b;
            max-width: 85%;
            margin-bottom: 40px;
            line-height: 1.7;
        }
        
        .cover-meta {
            font-size: 10pt;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
            padding-top: 20px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 20mm;
        }
        
        .meta-group strong {
            color: #0f172a;
            display: block;
            margin-bottom: 4px;
        }

        /* Section Headings */
        .section-header {
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
            margin-bottom: 20px;
            margin-top: 10mm;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            page-break-after: avoid;
        }
        
        .section-title {
            font-size: 16pt;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.5px;
        }
        
        .section-subtitle {
            font-size: 8.5pt;
            color: #2563eb;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        
        h2 {
            font-size: 12.5pt;
            font-weight: 700;
            color: #1e293b;
            margin-top: 24px;
            margin-bottom: 10px;
            border-left: 3px solid #10b981;
            padding-left: 10px;
            page-break-after: avoid;
        }

        h3 {
            font-size: 11pt;
            font-weight: 600;
            color: #334155;
            margin-top: 16px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }
        
        p {
            margin-bottom: 14px;
            color: #334155;
            text-align: justify;
            font-size: 10pt;
            line-height: 1.6;
        }
        
        strong {
            color: #0f172a;
        }
        
        /* Callout Box styling */
        .callout-box {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 15px 18px;
            margin-bottom: 20px;
            page-break-inside: avoid;
        }
        
        .callout-title {
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 6px;
            font-size: 10.5pt;
        }

        .callout-box.green-box {
            border-left-color: #10b981;
            background-color: #f0fdf4;
        }
        .callout-box.green-box .callout-title {
            color: #14532d;
        }

        .callout-box.amber-box {
            border-left-color: #f59e0b;
            background-color: #fffbeb;
        }
        .callout-box.amber-box .callout-title {
            color: #78350f;
        }
        
        /* Code blocks */
        pre {
            background-color: #0f172a;
            color: #e2e8f0;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 8.5pt;
            line-height: 1.5;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            margin-bottom: 18px;
            font-family: 'JetBrains Mono', Consolas, monospace;
            border: 1px solid #1e293b;
            page-break-inside: avoid;
        }
        
        code {
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-size: 9pt;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 5px;
            border-radius: 4px;
            font-weight: 500;
        }

        pre code {
            background-color: transparent;
            color: inherit;
            padding: 0;
            font-weight: normal;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 9pt;
            page-break-inside: avoid;
        }
        
        th {
            background-color: #f1f5f9;
            color: #475569;
            font-weight: 700;
            padding: 10px 12px;
            border-bottom: 2px solid #cbd5e1;
            text-align: left;
        }
        
        td {
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
            color: #334155;
            vertical-align: top;
        }
        
        tr:nth-child(even) td {
            background-color: #f8fafc;
        }

        /* Flow Diagram styling */
        .diagram-container {
            background-color: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 8.5pt;
            line-height: 1.45;
            white-space: pre;
            overflow-x: auto;
            color: #334155;
            page-break-inside: avoid;
        }

        .math-block {
            text-align: center;
            font-family: "Cambria Math", "Times New Roman", serif;
            font-size: 12pt;
            margin: 15px 0;
            padding: 10px;
            background-color: #f8fafc;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            color: #0f172a;
            page-break-inside: avoid;
        }

        .bullet-list {
            margin-bottom: 16px;
            padding-left: 20px;
        }

        .bullet-list li {
            font-size: 10pt;
            margin-bottom: 6px;
            color: #334155;
        }
        
    </style>
</head>
<body>

    <!-- ==================== PAGE 1: COVER ==================== -->
    <div class="page cover-page">
        <div class="cover-title-container">
            <div class="cover-title">TerraDrishti Unified Khasra Engine</div>
            <div class="cover-subtitle">Technical Specification, System Architecture & Algorithmic Design</div>
        </div>
        
        <div class="cover-divider"></div>

        <div class="cover-description">
            This document outlines the complete technical specifications, software technology stack, parallel execution flows, security design, and mathematical frameworks governing the <strong>TerraDrishti Unified Khasra Engine</strong>. Developed for the <strong>AdvaRisk Audit Division</strong>, this platform utilizes multi-sensor satellite imagery (Sentinel-1 and Sentinel-2) and machine learning classification (Dynamic World) to automate agricultural land audits, verify parcel crop cycles, and support ag-credit risk evaluation.
        </div>
        
        <div class="cover-meta">
            <div class="meta-group">
                <strong>PREPARED BY</strong>
                Antigravity AI Pair Programmer
            </div>
            <div class="meta-group">
                <strong>TARGET COHORTS</strong>
                2023-24, 2024-25, 2025-26
            </div>
            <div class="meta-group" style="text-align: right;">
                <strong>DOCUMENT VERSION</strong>
                v1.0.0 (Technical Release)<br>
                June 2026
            </div>
        </div>
    </div>
    
    <!-- ==================== PAGE 2: PROJECT OVERVIEW ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">1. Executive Summary & System Philosophy</div>
            <div class="section-subtitle">Introduction</div>
        </div>
        
        <p>
            In downstream Earth Observation (EO) applications, the core challenge is not simply downloading more satellite pixels, but converting raw spatial data into reliable, decision-ready answers that align directly with business workflows. The <strong>TerraDrishti Unified Khasra Engine</strong> is designed to address this challenge in the agricultural credit sector. It automates Khasra-level (land parcel) monitoring to provide financial lenders and audit teams with verifiable evidence of crop activity.
        </p>

        <div class="callout-box green-box">
            <div class="callout-title">The Downstream Value Chain Thesis</div>
            <p style="font-size: 9.5pt; margin-bottom: 0;">
                Traditional Earth Observation platforms operate primarily at the <strong>upstream</strong> layer (~20% of code), focusing on raw spectral band acquisition and pixel indexing. TerraDrishti operates at the <strong>downstream</strong> application layer (~80% of code), applying temporal interpolation, cloud-filtering, Whittaker smoothing, biological cycle verification, and environmental guardbands to synthesize raw values into a legally and financially audit-ready PDF report.
            </p>
        </div>

        <h2>Core Platform Goals</h2>
        <ul class="bullet-list">
            <li><strong>Automated Khasra Resolution:</strong> Map user-supplied land parcels (Khasra numbers) to unique geospatial IDs (GUIDs) to fetch raw boundary coordinates.</li>
            <li><strong>Parallel Data Acquisition:</strong> Query Google Earth Engine (Sentinel-2 optical, Sentinel-1 radar, and Dynamic World land cover datasets), local climatology repositories, and routing interfaces simultaneously.</li>
            <li><strong>Multi-Sensor Signal Fusion:</strong> Combine physics-based vegetation indices (NDVI, EVI, RVI) with machine-learning land use classifications to detect crop cycles and mitigate seasonal cloud obstructions.</li>
            <li><strong>Lender Risk Protection:</strong> Apply automated environmental guardbands to identify non-crop dominant land coverages (orchards, forests, water bodies) that mimic active crop signals.</li>
            <li><strong>Deterministic Audit Trails:</strong> Archive intermediate results in binary pickle formats to Google Cloud Storage to enable exact verification replays of any generated report.</li>
        </ul>

        <h2>Multi-Year Benchmarking Baseline</h2>
        <p>
            A multi-year evaluation across a benchmark cohort of <strong>306 monitored farms</strong> across multiple Indian states highlights the system's operational stability. The ground truth profile is 100% static, comprising exactly <strong>218 permanently active crop fields</strong> and <strong>88 permanently inactive fields</strong>. TerraDrishti's consensus engine achieves a baseline accuracy of <strong>92.43%</strong> in 2023-24, with minor adjustments to <strong>89.54%</strong> in 2024-25 and <strong>88.52%</strong> in 2025-26, driven by cloud cover and temporal imagery alignment constraints.
        </p>
    </div>
    
    <!-- ==================== PAGE 3: TECH STACK ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">2. Technology Stack & Component Functionality</div>
            <div class="section-subtitle">System Stack</div>
        </div>
        
        <p>
            The TerraDrishti platform is built using a modern, containerized Python stack designed for fast geospatial computations, parallel network I/O, and secure cloud delivery.
        </p>

        <table>
            <thead>
                <tr>
                    <th style="width: 25%;">Component Layer</th>
                    <th style="width: 25%;">Software / Stack</th>
                    <th style="width: 50%;">Specific Functional Role in Codebase</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Frontend Dashboard</strong></td>
                    <td>Streamlit, Folium</td>
                    <td>Provides the user interface for Khasra search, renders parcel geometry overlays on ESRI Satellite Tiles, and displays Cloud Run signed URL report links.</td>
                </tr>
                <tr>
                    <td><strong>Backend API Gateway</strong></td>
                    <td>FastAPI, Uvicorn</td>
                    <td>Exposes asynchronous REST API endpoints (<code>/test/accuracy</code> and <code>/test/replay</code>) to orchestrate analysis pipelines and trigger replay checks.</td>
                </tr>
                <tr>
                    <td><strong>Geospatial Processing</strong></td>
                    <td>Google Earth Engine (GEE) API</td>
                    <td>Performs cloud-masked reduction over target polygons to fetch Sentinel-2, Sentinel-1, and Dynamic World pixel distributions.</td>
                </tr>
                <tr>
                    <td><strong>Analytical Engine</strong></td>
                    <td>Pandas, NumPy, SciPy (Signal)</td>
                    <td>Performs linear interpolation, Whittaker smoothing, Savitzky-Golay filtering, peak detection, and consensus voting.</td>
                </tr>
                <tr>
                    <td><strong>Data Operations</strong></td>
                    <td>Shapely, GeoPandas</td>
                    <td>Used for geometry validation, polygon centroid computation, and coordinate coordinate-system parsing.</td>
                </tr>
                <tr>
                    <td><strong>Weather Analysis</strong></td>
                    <td>NASA POWER / Open-Meteo REST API</td>
                    <td>Fetches 1-year daily weather observations and 5-year monthly climatology trends for the parcel centroid coordinates.</td>
                </tr>
                <tr>
                    <td><strong>Report Compiler</strong></td>
                    <td>Playwright, Jinja2</td>
                    <td>Renders HTML Jinja2 templates (with inline Base64 data-viz charts) and compiles them into print-ready A4 PDF reports via Chromium.</td>
                </tr>
                <tr>
                    <td><strong>Cloud Storage & Security</strong></td>
                    <td>GCS API, Google Auth</td>
                    <td>Manages storage of generated PDFs and raw pickle data in private buckets, authorizing requests via impersonated service accounts and generating 7-day private signed URLs.</td>
                </tr>
                <tr>
                    <td><strong>Deployment</strong></td>
                    <td>Docker, Cloud Run</td>
                    <td>Containerizes the application on Python 3.10-slim with system libraries for headless Playwright execution on GCP.</td>
                </tr>
            </tbody>
        </table>

        <h2>Component Execution Responsibility</h2>
        <ul class="bullet-list">
            <li><code>main.py</code>: Streamlit app; manages interactive front-end, reference master CSV cache, and Folium map.</li>
            <li><code>api.py</code>: Backend API routes; handles requests using Python's <code>ThreadPoolExecutor</code>.</li>
            <li><code>data_loader.py</code>: Handles Earth Engine credential validation, collection definitions, cloud-masking algorithms, Overpass API queries, and centroid calculation.</li>
            <li><code>analytics_engine.py</code>: Contains Whittaker smoothing matrix equations, crop cycle algorithms, consensus voting logic, and environmental guardbands.</li>
            <li><code>pdf_generator.py</code>: Orchestrates async Playwright instances to render reports and save files.</li>
            <li><code>rain_temp.py</code>: Fetches and structures weather data from NASA/Open-Meteo sources.</li>
            <li><code>location.py</code>: Calls Google Maps Static API to capture satellite snapshots of the boundary box.</li>
        </ul>
    </div>
    
    <!-- ==================== PAGE 4: SYSTEM FLOWS ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">3. Data Pipeline & Architectural Flows</div>
            <div class="section-subtitle">System Orchestration</div>
        </div>
        
        <p>
            The system executes a multi-threaded parallel data pipeline to retrieve satellite observations, local coordinates, and weather statistics simultaneously. This design minimizes processing latencies by running network-bound operations in parallel.
        </p>

        <div class="diagram-container">
[ User Request: Khasra Search ]
              ↓
  Streamlit: Match local CSV master -> GUID & State
              ↓
  API client: Fetch GeoJSON boundaries from Quantasip API
              ↓
  Post Payload to FastAPI Backend (/test/accuracy)
              ↓
  FastAPI ThreadPoolExecutor: Spawn Asynchronous Workers
  ┌───────────────────────────────┼──────────────────────────────┐
  ▼ (Satellite Worker)            ▼ (Location Worker)            ▼ (Weather Worker)
  Google Earth Engine API         Reverse Geocoding (Nominatim)  NASA POWER API
  - Sentinel-2 (NDVI, EVI)        Overpass API (Water sources)   - 1-Year Daily Weather
  - Sentinel-1 (RVI)              Google Static Maps API         - 5-Year Climatology
  - Dynamic World LULC            - Centroid & Map B64 Image     - Precipitation & Temp
  └───────────────────────────────┼──────────────────────────────┘
                                  ▼
                     asyncio.gather() Synthesis
                                  ↓
                     Data Preprocessing & Interpolation
                                  ↓
                     Whittaker Smoothing (Daily Grid)
                                  ↓
                     Consensus Engine & Guardband Vetoes
                                  ↓
                     Matplotlib & Plotly Chart Generation
                                  ↓
                     Playwright HTML to PDF Compiler
                                  ↓
             Upload PDF & Raw Signals Pickle to GCS Buckets
                                  ↓
           Generate V4 Signed URLs (7-Day Private Expiration)
                                  ↓
       Return JSON Response -> Streamlit UI displays Download Links
        </div>

        <h2>Detailed Pipeline Step-by-Step Flow</h2>
        <p>
            <strong>1. Input and Lookup:</strong> The user searches a Khasra number in the sidebar of <code>main.py</code>. The application checks the local cache of <code>Telangana_Tehsil_master.csv</code> (or appropriate state CSV) to retrieve the geographic GUID and State.
        </p>
        <p>
            <strong>2. Geometry Retrieval:</strong> The application queries <code>https://test-client.quantasip.com/api/parcelData</code>. It extracts the Khasra features, parses nested geometry coordinates (handling MultiPolygon formats), and displays the parcel outline on a yellow-themed Folium map.
        </p>
        <p>
            <strong>3. Async Worker Launch:</strong> Clicking "Run Intelligence Report" posts the coordinates and properties payload to the Cloud Run API. The backend <code>api.py</code> uses <code>asyncio.get_event_loop().run_in_executor()</code> to execute the satellite, location, and weather queries across parallel threads.
        </p>
    </div>

    <!-- ==================== PAGE 5: ALGORITHMS PART 1 ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">4. Deep Dive: Preprocessing & Whittaker Smoothing</div>
            <div class="section-subtitle">Mathematics & Algorithms</div>
        </div>
        
        <h2>Spectral Band Calculations</h2>
        <p>
            Google Earth Engine processes Sentinel-2 Level-2A surface reflectance data. A cloud mask is applied using the Scene Classification Layer (SCL) to remove clouds, cloud shadows, and cirrus. Spectral indices are calculated for each pixel within the parcel polygon:
        </p>
        
        <div class="math-block">
            NDVI = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}} = \frac{NIR - RED}{NIR + RED}
        </div>
        <div class="math-block">
            EVI = 2.5 \cdot \frac{NIR - RED}{NIR + 6 \cdot RED - 7.5 \cdot BLUE + 1}
        </div>
        <p>
            For cloud-penetrating radar, Sentinel-1 Ground Range Detected (GRD) Dual-Pol imagery is queried. Radar backscatter (expressed in decibels) is converted to linear power to calculate the Radar Vegetation Index (RVI):
        </p>
        <div class="math-block">
            VV_{linear} = 10^{\frac{VV_{dB}}{10}}, \quad VH_{linear} = 10^{\frac{VH_{dB}}{10}}
        </div>
        <div class="math-block">
            RVI = \frac{4 \cdot VH_{linear}}{VV_{linear} + VH_{linear}}
        </div>

        <h2>Whittaker Smoothing formulation</h2>
        <p>
            Satellite observations are irregularly spaced due to orbit paths and cloud cover. To establish a uniform daily sequence, the data loader groups values by date, calculates the mean over the parcel coordinates, resamples to a daily grid (<code>'D'</code>), and interpolates missing dates.
        </p>
        <p>
            To remove sensor noise and fill temporal gaps, the engine applies Whittaker smoothing. This algorithm balances data fidelity with curve smoothness by minimizing a penalized least-squares objective function:
        </p>
        <div class="math-block">
            Q = \sum_{i=1}^{m} w_i (y_i - z_i)^2 + \lambda \sum_{i=1}^{m} (\Delta^d z_i)^2
        </div>
        <p>
            Where:
        </p>
        <ul class="bullet-list">
            <li><code>y</code>: Vector of raw observed index values.</li>
            <li><code>z</code>: Vector of smoothed output values.</li>
            <li><code>w</code>: Diagonal weight matrix. $w_i = 1.0$ for valid scene dates, $w_i = 0.0$ for interpolated dates.</li>
            <li>$\Delta^d$: Difference operator of order $d$ (system uses second-order differences, $d=2$).</li>
            <li>$\lambda$: Smoothing parameter. Higher values increase smoothing; lower values preserve raw data points.</li>
        </ul>
        <p>
            The objective function is solved by computing:
        </p>
        <div class="math-block">
            (W + \lambda D^T D) z = W y
        </div>
        <p>
            Where $W$ is the diagonal matrix of weights, $D$ is the sparse difference matrix representing the second-order derivative, and $z$ is solved using SciPy's sparse linear solver (<code>spsolve</code>).
        </p>
        <div class="callout-box amber-box">
            <div class="callout-title">Calibrated Smoothing Parameters (\lambda)</div>
            <p style="font-size: 9.5pt; margin-bottom: 0;">
                The smoothing coefficients are calibrated separately for each sensor to optimize noise reduction without smoothing out real agricultural cycles:
                <br>• <strong>NDVI:</strong> $\lambda = 50$ (moderate smoothing to capture vegetation greenness trends).
                <br>• <strong>EVI:</strong> $\lambda = 50$ (designed to match NDVI's temporal profile).
                <br>• <strong>RVI:</strong> $\lambda = 200$ (higher value to filter high-frequency radar speckle noise).
            </p>
        </div>
    </div>

    <!-- ==================== PAGE 6: ALGORITHMS PART 2 ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">5. Deep Dive: Cycle Detection & Consensus Engine</div>
            <div class="section-subtitle">Mathematics & Algorithms</div>
        </div>
        
        <h2>Crop Cycle & Peak Detection</h2>
        <p>
            Before applying classification rules, the system detects seasonal crop cycles. It applies a Savitzky-Golay filter (a local polynomial regression of degree 2) to smooth the daily curves:
        </p>
        <pre># Savitzky-Golay window configuration based on data density
df['rvi_smooth'] = savgol_filter(df['RVI'].fillna(0), window_rvi, 2)
df['ndvi_smooth'] = savgol_filter(df['NDVI'].fillna(0), window_ndvi, 2)
df['ndvi_short_smooth'] = savgol_filter(df['NDVI'].fillna(0), window_ndvi_short, 2)</pre>
        <p>
            The peak detector searches for local maxima using <code>scipy.signal.find_peaks</code> within state-specific crop season windows (Kharif: May–Nov; Rabi: Oct–Apr; Zaid: Mar–Jun). A candidate peak is validated using a biological amplitude rule:
        </p>
        <div class="math-block">
            Peak_{validated} \iff (\text{Peak Value} - \text{Minimum Baseline Value}) \ge \text{Prominence Threshold}
        </div>
        <p>
            Prominence thresholds: RVI $\ge 0.10$, NDVI $\ge 0.15$, NDVI-Short $\ge 0.10$.
        </p>

        <h2>Biological Cycle Plausibility Validation</h2>
        <p>
            For each detected cycle, the system calculates a weighted <strong>Biological Plausibility Score</strong>:
        </p>
        <ul class="bullet-list">
            <li><strong>Persistence (35%):</strong> Evaluates cycle duration. Measured as the number of days the smoothed index remains above 0.35, scaled against a standard 45-day threshold.</li>
            <li><strong>Signal Stability (15%):</strong> Evaluates signal noise. Measures the mean absolute residuals between raw observed values and the smoothed curve, penalizing high variance.</li>
            <li><strong>SAR/Optical Correlation (30%):</strong> Biological check. Calculates the Pearson correlation coefficient between NDVI and RVI during the 60-day peak window, verifying that structural growth aligns with greenness.</li>
            <li><strong>Growth Velocity (20%):</strong> Enforces biological limits. Penalizes cycles where daily growth rates exceed 0.05 NDVI units/day, which typically indicates cloud artifacts.</li>
        </ul>

        <h2>Consensus Engine & Guardband Vetoes</h2>
        <p>
            The final daily classification runs on a weighted consensus model:
        </p>
        <div class="diagram-container" style="font-size: 8pt;">
  Pipeline 1 (Physics Model): Indices & Slopes ───► P1 Crop Confidence (60% weight) ──┐
                                                                                       ├──► Combined Verdict
  Pipeline 2 (AI Model): Dynamic World Probs ─────► P2 Crop Confidence (40% weight) ──┘
        </div>
        <p>
            If either pipeline expresses high certainty ($\ge 80\%$), that pipeline is given higher weight.
        </p>
        <p>
            <strong>Environmental Guardband Vetoes:</strong> Overwhelming non-crop land cover can generate false crop cycles (e.g., evergreen forests showing high greenness). The system evaluates the annual class shares. If non-crop classes (trees, grass, water, built, bare) dominate the year with $\ge 60\%$ frequency, and crop probability remains low ($\le 25\%$), agricultural activity is vetoed.
        </p>
        <p>
            <strong>Certainty Score:</strong> Calculated as a composite metric of the classification margin, model agreement, data sufficiency, and land alignment. It is penalized for sparse optical scenes, borderline activity ratios, and guardband conflicts, and then scaled using a smoothstep transfer function:
        </p>
        <div class="math-block">
            nx = \min\left(\frac{\text{Composite Score}}{85.0}, 1.0\right), \quad \text{Certainty} = (nx^2 \cdot (3 - 2nx)) \cdot 100
        </div>
    </div>

    <!-- ==================== PAGE 7: SYSTEM DESIGN ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">6. System Design & Infrastructure Patterns</div>
            <div class="section-subtitle">Architecture & Security</div>
        </div>
        
        <h2>Asynchronous Multi-Threading & Concurrency</h2>
        <p>
            Geospatial API calls to Google Earth Engine and reverse-geocoding requests to Overpass or Nominatim APIs are heavily I/O-bound. Running these synchronously would block the web server and degrade response times.
        </p>
        <p>
            To resolve this, FastAPI runs in an asynchronous event loop, delegating worker threads via Python's <code>ThreadPoolExecutor</code>. Thread workers handle GEE data retrieval and NASA Power weather fetching concurrently, bypassing Python's Global Interpreter Lock (GIL) for network-bound tasks.
        </p>
        <pre># ThreadPoolExecutor configuration in api.py
executor = ThreadPoolExecutor(max_workers=20)

@app.post("/test/accuracy")
async def test_accuracy_by_geometry(request: GeometryRequest):
    loop = asyncio.get_event_loop()
    coords = parse_kml_string(request.kml_coordinates)
    task_id = request.task_id

    # Execute workers in parallel threads
    b1 = loop.run_in_executor(executor, satellite_worker, task_id, coords, request.end_date)
    b2 = loop.run_in_executor(executor, location_worker, task_id, coords)
    b3 = loop.run_in_executor(executor, weather_worker, coords, request.end_date)

    sat_res, loc_res, weather_res = await asyncio.gather(b1, b2, b3)</pre>

        <h2>Cloud Security Architecture & Signed Access</h2>
        <p>
            Due to privacy regulations surrounding land ownership and financial audits, generated reports and raw signal pickles must not be publicly accessible.
        </p>
        <ul class="bullet-list">
            <li><strong>Impersonated IAM Credentials:</strong> Instead of embedding long-lived keys, the application utilizes Application Default Credentials (ADC) to dynamically assume an identity with narrow write access (<code>413500342905-compute@developer.gserviceaccount.com</code>) to the <code>terradrishti</code> bucket.</li>
            <li><strong>v4 Signature Generation:</strong> Using the impersonated credentials, the system generates a secure v4 signed URL for the compiled PDF and binary pickle files.</li>
            <li><strong>Temporal Access Controls:</strong> Signed links are configured with an expiration window of exactly 7 days (<code>timedelta(days=7)</code>). The links authorize read-only (<code>GET</code>) access to that specific blob, protecting data access.</li>
        </ul>

        <h2>Docker Containerization</h2>
        <p>
            Headless PDF compilation requires browser binaries and system libraries. The <code>Dockerfile</code> uses a <code>python:3.10-slim</code> base image and installs dependencies like <code>libnss3</code>, <code>libatk-bridge2.0-0</code>, and <code>libgtk-3-0</code>. It installs Python dependencies via <code>requirements.txt</code>, configures Playwright Chromium, adds the application to <code>PYTHONPATH</code>, and launches Uvicorn on port 8080.
        </p>
    </div>

    <!-- ==================== PAGE 8: CLOUD BUILD & CLOUD RUN ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">7. Cloud Build, Container Image & Serverless Infrastructure</div>
            <div class="section-subtitle">CI/CD & Deployment</div>
        </div>

        <h2>Docker Container Construction (Image Architecture)</h2>
        <p>
            The project uses a containerized architecture to ensure identical execution environments across local development and Cloud Run. The container is defined in the <code>Dockerfile</code> and built upon the official <code>python:3.10-slim</code> base image:
        </p>
        <ul class="bullet-list">
            <li><strong>Operating System Layer:</strong> Uses Debian-slim to keep the container footprint small while supporting native Unix compilers. It installs <code>build-essential</code> and <code>pkg-config</code> to compile C/C++ extensions, and <code>libcairo2-dev</code> for vector rendering.</li>
            <li><strong>Browser Automation Dependencies:</strong> Installs <code>libnss3</code>, <code>libatk-bridge2.0-0</code>, <code>libgtk-3-0</code>, and <code>libasound2</code>. These libraries are required for running Playwright's headless Chromium browser.</li>
            <li><strong>Application Layers:</strong> Pip installs dependencies with <code>--no-cache-dir</code>. It then executes <code>playwright install --with-deps chromium</code> to download the Chromium browser binary and its dependencies directly into the image.</li>
            <li><strong>Execution Context:</strong> Sets <code>ENV PYTHONPATH=/app</code> so absolute imports resolve correctly. It exposes port 8080 and configures the entry point to run Uvicorn: <code>CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]</code>.</li>
        </ul>

        <h2>CI/CD Pipeline with Google Cloud Build</h2>
        <p>
            To automate deployment, the codebase is integrated with a Google Cloud Build pipeline. When a developer pushes commits to the repository, Cloud Build triggers the compilation:
        </p>
        <ul class="bullet-list">
            <li><strong>Build Verification:</strong> Cloud Build executes steps in the cloud, resolving python packages, running system checks, and building the container layers.</li>
            <li><strong>Artifact Storage:</strong> Pushes the compiled image to Google Artifact Registry (GAR) in the project region. Image tag format: <code>asia-south1-docker.pkg.dev/advarisk/terradrishti-repo/unified-engine:latest</code>.</li>
        </ul>

        <h2>Serverless Orchestration via Google Cloud Run</h2>
        <p>
            The container is deployed to Google Cloud Run, a managed serverless platform. Deployment properties:
        </p>
        <ul class="bullet-list">
            <li><strong>Regional Hosting (asia-south1):</strong> Hosting in Mumbai, India localizes data traffic and reduces network latencies when fetching Indian local datasets and master files.</li>
            <li><strong>Hardware Allocation:</strong> Configured with 2 vCPUs and 4 GiB Memory. Generating PDF files via Playwright Chromium requires significant memory, making smaller instances susceptible to Out-Of-Memory (OOM) failures.</li>
            <li><strong>Service Account Identity:</strong> Executes using the service identity <code>413500342905-compute@developer.gserviceaccount.com</code>. This service identity has specific IAM permissions for GEE, Secret Manager, and Google Cloud Storage (allowing it to generate signed URLs).</li>
            <li><strong>Auto-Scaling Dynamics:</strong> Scales down to 0 instances when idle to reduce costs. During parallel batch operations, it scales up to 100+ concurrent containers, distributing the GEE API queue load.</li>
        </ul>
    </div>

    <!-- ==================== PAGE 9: BENCHMARKING ==================== -->
    <div class="page" style="padding-top: 10mm;">
        <div class="section-header">
            <div class="section-title">8. Benchmarking Performance & Operational Insights</div>
            <div class="section-subtitle">Benchmarking Analysis</div>
        </div>
        
        <p>
            This section summarizes the performance benchmarks derived from running the 306-farm multi-year dataset, detailing model performance, regional characteristics, and audit recommendations.
        </p>

        <h2>Year-over-Year Performance Summary</h2>
        <p>
            The evaluation indicates stable baseline metrics, with a slight adjustment in recall over the three-year period.
        </p>
        <table>
            <thead>
                <tr>
                    <th>Benchmark Cohort Year</th>
                    <th>Overall Accuracy</th>
                    <th>Precision Score</th>
                    <th>Recall Score</th>
                    <th>F1 Quality Index</th>
                    <th>True Negatives</th>
                    <th>False Negatives</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>2023-24 (Baseline Year)</strong></td>
                    <td>92.43%</td>
                    <td>96.19%</td>
                    <td>93.09%</td>
                    <td>94.61%</td>
                    <td>79 / 88</td>
                    <td>15 / 218</td>
                </tr>
                <tr>
                    <td><strong>2024-25 (Mid-Term Year)</strong></td>
                    <td>89.54%</td>
                    <td>95.59%</td>
                    <td>89.45%</td>
                    <td>92.42%</td>
                    <td>79 / 88</td>
                    <td>23 / 218</td>
                </tr>
                <tr>
                    <td><strong>2025-26 (Latest Model)</strong></td>
                    <td>88.52%</td>
                    <td>95.98%</td>
                    <td>87.61%</td>
                    <td>91.61%</td>
                    <td>79 / 88</td>
                    <td>27 / 218</td>
                </tr>
            </tbody>
        </table>

        <h2>Regional Performance Analysis</h2>
        <p>
            State-level grouping reveals substantial performance variance across different geographical zones:
        </p>
        <ul class="bullet-list">
            <li><strong>Alluvial Agriculture Zones:</strong> States such as Uttar Pradesh (98.3% avg accuracy) and Bihar (96.3% avg accuracy) exhibit high performance. These regions have flat terrains, distinct crop cycles, and minimal canopy obstruction.</li>
            <li><strong>Coastal & Evergreen Zones:</strong> Kerala exhibits lower model performance (70.8% avg accuracy). Perennial canopy coverages (coconut and rubber plantations) maintain high NDVI/EVI values year-round, which can distort annual crop cycle calculations.</li>
            <li><strong>Arid & Hilly Zones:</strong> Rajasthan (83.3% accuracy) and Himachal Pradesh (83.3% accuracy) present moderate accuracy, influenced by dry soil reflections and rugged terrains.</li>
        </ul>

        <h2>Strategic Audit Pipeline Recommendations</h2>
        <p>
            <strong>1. Automated Certainty Gateways:</strong> Implement a system rule where any parcel flagged with <em>Moderate</em>, <em>Low</em>, or <em>Very Low</em> certainty (composite score &lt; 65) is automatically routed to a manual audit queue. Parcels in the Very High certainty tier achieved a 95.9% success rate, indicating they are suitable for automated processing.
        </p>
        <p>
            <strong>2. Perennial Tree-Crop Masking:</strong> Southern and coastal regions should use tree-crop masking filters. Excluding pixels classified as mature forests or orchards prevents evergreen canopies from generating false crop cycles.
        </p>
        <p>
            <strong>3. Sentinel-1 SAR Integration:</strong> For states affected by monsoon cloud cover (Telangana, Karnataka, Kerala), rely on Sentinel-1 SAR backscatter rather than optical indices to maintain observation frequency.
        </p>
    </div>

</body>
</html>
"""

async def main():
    # Write the html content to a temporary file
    temp_html_path = "project_spec_temp.html"
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        page = await browser.new_page()
        
        # Load the HTML page
        await page.goto(f"file://{os.path.abspath(temp_html_path)}")
        await page.wait_for_load_state("networkidle")
        
        pdf_output_path = "AdvaRisk_Project_Specification_and_System_Design.pdf"
        print(f"Compiling PDF to {pdf_output_path}...")
        
        # Compile to PDF
        await page.pdf(
            path=pdf_output_path,
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="""
                <div style="font-size: 8px; width: 100%; margin: 0 15mm; display: flex; justify-content: space-between; color: #94a3b8; font-family: 'Inter', sans-serif;">
                    <span>TERRADRISHTI UNIFIED KHASRA ENGINE - SYSTEM DESIGN</span>
                    <span>CONFIDENTIAL - TECHNICAL REFERENCE</span>
                </div>
            """,
            footer_template="""
                <div style="font-size: 8px; width: 100%; margin: 0 15mm; border-top: 1px solid #e2e8f0; padding-top: 4px; display: flex; justify-content: space-between; color: #94a3b8; font-family: 'Inter', sans-serif;">
                    <span>AdvaRisk Agriculture Analytics Platform</span>
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
    
    # Remove temp html
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)
        
    print(f"PDF generation complete: {pdf_output_path}")

if __name__ == "__main__":
    asyncio.run(main())

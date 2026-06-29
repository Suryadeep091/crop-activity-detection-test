import json
import os

stats_file = "analysis_stats.json"
output_file = "dashboard.html"

if not os.path.exists(stats_file):
    raise FileNotFoundError(f"{stats_file} not found. Please run run_analysis.py first.")

with open(stats_file, "r") as f:
    stats = json.load(f)

# Convert to JSON string for embedding
stats_json_str = json.dumps(stats, indent=4)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Year Farm Satellite Analytics Dashboard (2023-26)</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #080c14;
            --card-bg: rgba(17, 25, 40, 0.65);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.15);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.15);
            --warning: #f59e0b;
            --danger: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.15);
            --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: var(--font-main);
            min-height: 100vh;
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 50% 0%, rgba(16, 185, 129, 0.05) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(147, 51, 234, 0.08) 0px, transparent 50%);
            background-attachment: fixed;
            overflow-x: hidden;
        }}

        header {{
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            background: rgba(8, 12, 20, 0.8);
            border-bottom: 1px solid var(--card-border);
            padding: 1.25rem 2.5rem;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .logo-section h1 {{
            font-size: 1.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #3b82f6, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }}

        .logo-section p {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }}

        .nav-tabs {{
            display: flex;
            gap: 0.5rem;
        }}

        .tab-btn {{
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-secondary);
            padding: 0.6rem 1.2rem;
            font-size: 0.85rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.03);
        }}

        .tab-btn.active {{
            color: var(--text-primary);
            background: rgba(59, 130, 246, 0.1);
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.1);
        }}

        .container {{
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
        }}

        .tab-content {{
            display: none;
            animation: fadeIn 0.4s ease;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.3s ease;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        }}

        .kpi-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(59, 130, 246, 0.2);
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.05);
        }}

        .kpi-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}

        .kpi-title {{
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
        }}

        .kpi-icon {{
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
        }}

        .kpi-val {{
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 0.25rem;
        }}

        .kpi-sub {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}

        .blue-card .kpi-icon {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; }}
        .green-card .kpi-icon {{ background: rgba(16, 185, 129, 0.15); color: #34d399; }}
        .orange-card .kpi-icon {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; }}
        .red-card .kpi-icon {{ background: rgba(239, 68, 68, 0.15); color: #f87171; }}

        /* Main Dashboard Grid */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }}

        .card-title {{
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -0.3px;
        }}

        .card-subtitle {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }}

        /* Confusion Matrix CSS */
        .cm-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 1rem;
        }}

        .cm-cell {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }}

        .cm-cell-num {{
            font-size: 1.6rem;
            font-weight: 800;
            margin-bottom: 4px;
        }}

        .cm-cell-label {{
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}

        /* Table CSS */
        .table-controls {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }}

        .search-box {{
            flex: 1;
            min-width: 250px;
            position: relative;
        }}

        .search-box input {{
            width: 100%;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--card-border);
            color: var(--text-primary);
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            border-radius: 10px;
            font-family: var(--font-main);
            font-size: 0.85rem;
            outline: none;
            transition: all 0.3s ease;
        }}

        .search-box input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.1);
        }}

        .search-box::before {{
            content: "🔍";
            position: absolute;
            left: 0.85rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.9rem;
            opacity: 0.5;
        }}

        .select-filter {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--card-border);
            color: var(--text-primary);
            padding: 0.75rem 1.5rem 0.75rem 1rem;
            border-radius: 10px;
            font-family: var(--font-main);
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
            min-width: 150px;
        }}

        .select-filter option {{
            background-color: var(--bg-color);
            color: var(--text-primary);
        }}

        .table-container {{
            width: 100%;
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid var(--card-border);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.85rem;
        }}

        th {{
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--card-border);
        }}

        td {{
            padding: 1rem 1.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            color: var(--text-primary);
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.015);
        }}

        .badge {{
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }}

        .badge-active {{ background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); }}
        .badge-inactive {{ background: rgba(156, 163, 175, 0.12); color: #d1d5db; border: 1px solid rgba(156, 163, 175, 0.2); }}
        .badge-error {{ background: rgba(239, 68, 68, 0.12); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }}
        .badge-match {{ background: var(--success-glow); color: var(--success); }}
        .badge-mismatch {{ background: var(--danger-glow); color: var(--danger); }}

        .accuracy-pill {{
            padding: 0.35rem 0.7rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
        }}

        .accuracy-3 {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .accuracy-2 {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .accuracy-1 {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}

        .pagination {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1.5rem;
            padding: 0 0.5rem;
        }}

        .page-info {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .page-controls {{
            display: flex;
            gap: 0.5rem;
        }}

        .page-btn {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--card-border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }}

        .page-btn:hover:not(:disabled) {{
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--primary);
        }}

        .page-btn:disabled {{
            opacity: 0.35;
            cursor: not-allowed;
        }}

        /* State leaderboard styling */
        .leaderboard-list {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}

        .leaderboard-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 10px;
        }}

        .state-name-tag {{
            font-weight: 700;
            background: rgba(59, 130, 246, 0.1);
            color: #60a5fa;
            width: 38px;
            height: 38px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .state-progress-container {{
            flex: 1;
            margin: 0 1.25rem;
        }}

        .state-progress-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            margin-bottom: 4px;
        }}

        .state-progress-bar-bg {{
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            overflow: hidden;
        }}

        .state-progress-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #10b981);
            border-radius: 3px;
        }}

        .state-score-val {{
            font-weight: 700;
            font-size: 0.95rem;
        }}
    </style>
</head>
<body>

    <header>
        <div class="logo-section">
            <h1>Terradrishti Analytics</h1>
            <p>Multi-Year Farm Satellite Accuracy Benchmark (2023 - 26)</p>
        </div>
        <div class="nav-tabs">
            <button class="tab-btn active" onclick="switchTab('dashboard')">Performance Summary</button>
            <button class="tab-btn" onclick="switchTab('transitions')">Stability & Transitions</button>
            <button class="tab-btn" onclick="switchTab('states')">Regional Analysis</button>
            <button class="tab-btn" onclick="switchTab('explorer')">Farm Explorer</button>
        </div>
    </header>

    <div class="container">
        
        <!-- ==================== TAB 1: SUMMARY ==================== -->
        <div id="dashboard" class="tab-content active">
            <!-- KPI Cards -->
            <div class="kpi-grid">
                <div class="kpi-card blue-card">
                    <div class="kpi-header">
                        <span class="kpi-title">Total Analysis Parcels</span>
                        <span class="kpi-icon">🚜</span>
                    </div>
                    <div class="kpi-val">918</div>
                    <div class="kpi-sub">306 unique farms monitored over 3 years</div>
                </div>
                <div class="kpi-card green-card">
                    <div class="kpi-header">
                        <span class="kpi-title">Accuracy '23-24</span>
                        <span class="kpi-icon">📈</span>
                    </div>
                    <div class="kpi-val" id="acc-24">92.4%</div>
                    <div class="kpi-sub">Baseline cohort success rate</div>
                </div>
                <div class="kpi-card green-card">
                    <div class="kpi-header">
                        <span class="kpi-title">Accuracy '24-25</span>
                        <span class="kpi-icon">📈</span>
                    </div>
                    <div class="kpi-val" id="acc-25">89.5%</div>
                    <div class="kpi-sub">Mid-term cohort success rate</div>
                </div>
                <div class="kpi-card green-card">
                    <div class="kpi-header">
                        <span class="kpi-title">Accuracy '25-26</span>
                        <span class="kpi-icon">📈</span>
                    </div>
                    <div class="kpi-val" id="acc-26">88.5%</div>
                    <div class="kpi-sub">Latest multi-season model schema</div>
                </div>
                <div class="kpi-card orange-card">
                    <div class="kpi-header">
                        <span class="kpi-title">Ground Truth Profile</span>
                        <span class="kpi-icon">⚖️</span>
                    </div>
                    <div class="kpi-val">218 vs 88</div>
                    <div class="kpi-sub">Active Crops vs Inactive/No-Crops</div>
                </div>
            </div>

            <!-- Dashboard Grid -->
            <div class="dashboard-grid">
                <div class="card">
                    <div class="card-header">
                        <div>
                            <h2 class="card-title">Year-over-Year Model Performance Trends</h2>
                            <p class="card-subtitle">Tracking precision, recall, and F1 index stability</p>
                        </div>
                    </div>
                    <div style="height: 320px; position: relative;">
                        <canvas id="trendChart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div>
                            <h2 class="card-title">Prediction vs Certainty Correlation</h2>
                            <p class="card-subtitle">AI accuracy segregated by confidence level tiers</p>
                        </div>
                    </div>
                    <div style="height: 320px; position: relative;">
                        <canvas id="certaintyChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="dashboard-grid" style="grid-template-columns: 1fr 1fr 1fr;">
                <div class="card">
                    <h2 class="card-title" style="margin-bottom: 1rem;">Confusion Matrix '23-24</h2>
                    <div class="cm-grid" id="cm-23-24">
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #60a5fa;">79</div><div class="cm-cell-label">TN (True Inact)</div></div>
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #ef4444;">8</div><div class="cm-cell-label">FP (False Act)</div></div>
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #ef4444;">15</div><div class="cm-cell-label">FN (False Inact)</div></div>
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #10b981;">202</div><div class="cm-cell-label">TP (True Act)</div></div>
                    </div>
                </div>
                <div class="card">
                    <h2 class="card-title" style="margin-bottom: 1rem;">Confusion Matrix '24-25</h2>
                    <div class="cm-grid" id="cm-24-25">
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #60a5fa;">79</div><div class="cm-cell-label">TN</div></div>
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #ef4444;">9</div><div class="cm-cell-label">FP</div></div>
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #ef4444;">23</div><div class="cm-cell-label">FN</div></div>
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #10b981;">195</div><div class="cm-cell-label">TP</div></div>
                    </div>
                </div>
                <div class="card">
                    <h2 class="card-title" style="margin-bottom: 1rem;">Confusion Matrix '25-26</h2>
                    <div class="cm-grid" id="cm-25-26">
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #60a5fa;">79</div><div class="cm-cell-label">TN</div></div>
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #ef4444;">8</div><div class="cm-cell-label">FP</div></div>
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #ef4444;">27</div><div class="cm-cell-label">FN</div></div>
                        <div class="cm-cell"><div class="cm-cell-num" style="color: #10b981;">191</div><div class="cm-cell-label">TP</div></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ==================== TAB 2: TRANSITIONS ==================== -->
        <div id="transitions" class="tab-content">
            <div class="dashboard-grid" style="grid-template-columns: 1fr 1fr;">
                <div class="card">
                    <div class="card-header">
                        <div>
                            <h2 class="card-title">Ground Truth Crop Stability Analysis</h2>
                            <p class="card-subtitle">Tracing real crop persistence over the 3-year timeline</p>
                        </div>
                    </div>
                    <div style="padding: 1rem 0;">
                        <p style="font-size: 0.9rem; line-height: 1.6; color: var(--text-secondary); margin-bottom: 1.5rem;">
                            A key analytical discovery in the ground truth dataset is that <strong>100% of all 306 parcels are physically static</strong> over three years. 
                            There are zero physical crop transitions (e.g. crop fields turning permanently barren, or barren fields turning agricultural).
                        </p>
                        <div class="cm-grid" style="grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                            <div class="cm-cell" style="padding: 1.5rem; background: rgba(59, 130, 246, 0.05); border-color: rgba(59, 130, 246, 0.15);">
                                <span style="font-size: 2.2rem;">🟢</span>
                                <div class="cm-cell-num" style="margin-top: 10px; color: #60a5fa;">218 Farms</div>
                                <div class="cm-cell-label" style="font-size: 0.8rem; margin-top: 4px;">Permanently Active (Crop)</div>
                                <p style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 8px;">Consistent agricultural crop activity observed across 2023, 2024, and 2025.</p>
                            </div>
                            <div class="cm-cell" style="padding: 1.5rem; background: rgba(156, 163, 175, 0.05); border-color: rgba(156, 163, 175, 0.15);">
                                <span style="font-size: 2.2rem;">⚫</span>
                                <div class="cm-cell-num" style="margin-top: 10px; color: #d1d5db;">88 Farms</div>
                                <div class="cm-cell-label" style="font-size: 0.8rem; margin-top: 4px;">Permanently Inactive (No-Crop)</div>
                                <p style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 8px;">Persistent barren state or non-crop dominant land coverage over all 36 months.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div>
                            <h2 class="card-title">AI Prediction Agreement on Stability Pathways</h2>
                            <p class="card-subtitle">How well the model captures multi-year stability</p>
                        </div>
                    </div>
                    <div style="height: 320px; position: relative;">
                        <canvas id="transitionChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- ==================== TAB 3: REGIONAL ==================== -->
        <div id="states" class="tab-content">
            <div class="dashboard-grid">
                <div class="card">
                    <div class="card-header">
                        <div>
                            <h2 class="card-title">State-wise Analytics Performance</h2>
                            <p class="card-subtitle">Year-over-Year accuracy trends in different regions</p>
                        </div>
                    </div>
                    <div style="height: 380px; position: relative;">
                        <canvas id="stateChart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div>
                            <h2 class="card-title">Regional Accuracy Leaderboard</h2>
                            <p class="card-subtitle">Average precision rank by state (Top 10)</p>
                        </div>
                    </div>
                    <div class="leaderboard-list" id="state-leaderboard">
                        <!-- Filled dynamically -->
                    </div>
                </div>
            </div>
        </div>

        <!-- ==================== TAB 4: EXPLORER ==================== -->
        <div id="explorer" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Farm-by-Farm Data Explorer</h2>
                        <p class="card-subtitle">Search, filter, and inspect AI predictions vs Ground Truth for all 306 farms (including failures/errors)</p>
                    </div>
                </div>

                <div class="table-controls">
                    <div class="search-box">
                        <input type="text" id="searchInput" placeholder="Search by farm name (e.g. UP1, RJ05)..." onkeyup="filterFarms()">
                    </div>
                    <select class="select-filter" id="stateFilter" onchange="filterFarms()">
                        <option value="ALL">All States</option>
                        <!-- Populated dynamically -->
                    </select>
                    <select class="select-filter" id="stabilityFilter" onchange="filterFarms()">
                        <option value="ALL">All Prediction Matches</option>
                        <option value="3">3/3 Years Correct (Perfect Match)</option>
                        <option value="2">2/3 Years Correct</option>
                        <option value="1">1/3 Years Correct</option>
                        <option value="0">0/3 Years Correct (Fully Mismatched)</option>
                    </select>
                </div>

                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Farm Name</th>
                                <th>State</th>
                                <th>Ground Truth</th>
                                <th>2023-24 (Certainty)</th>
                                <th>2024-25 (Certainty)</th>
                                <th>2025-26 (Certainty)</th>
                                <th>Success Rate</th>
                            </tr>
                        </thead>
                        <tbody id="farmTableBody">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>

                <div class="pagination">
                    <div class="page-info" id="pageInfo">Showing 1-15 of 306 entries</div>
                    <div class="page-controls">
                        <button class="page-btn" id="prevBtn" onclick="prevPage()" disabled>Previous</button>
                        <button class="page-btn" id="nextBtn" onclick="nextPage()">Next</button>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <!-- DATASET INTEGRATION -->
    <script>
        const STATS = {stats_json_str};

        // Navigation
        function switchTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            
            document.getElementById(tabId).classList.add('active');
            
            // Find tab button that has same onclick action
            const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
            if (btn) btn.classList.add('active');
        }}

        // Format percentage values for view
        document.getElementById('acc-24').innerText = STATS.yearly['2023-24'].accuracy.toFixed(1) + '%';
        document.getElementById('acc-25').innerText = STATS.yearly['2024-25'].accuracy.toFixed(1) + '%';
        document.getElementById('acc-26').innerText = STATS.yearly['2025-26'].accuracy.toFixed(1) + '%';

        // ------------------ CHARTS INITIALIZATION ------------------
        
        // 1. Performance Trend Chart
        const trendCtx = document.getElementById('trendChart').getContext('2d');
        new Chart(trendCtx, {{
            type: 'line',
            data: {{
                labels: ['2023-24', '2024-25', '2025-26'],
                datasets: [
                    {{
                        label: 'Accuracy',
                        data: [STATS.yearly['2023-24'].accuracy, STATS.yearly['2024-25'].accuracy, STATS.yearly['2025-26'].accuracy],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 3,
                        tension: 0.25,
                        fill: true
                    }},
                    {{
                        label: 'Precision',
                        data: [STATS.yearly['2023-24'].precision, STATS.yearly['2024-25'].precision, STATS.yearly['2025-26'].precision],
                        borderColor: '#10b981',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.25,
                        fill: false
                    }},
                    {{
                        label: 'Recall',
                        data: [STATS.yearly['2023-24'].recall, STATS.yearly['2024-25'].recall, STATS.yearly['2025-26'].recall],
                        borderColor: '#ff7f0e',
                        borderWidth: 2,
                        borderDash: [2, 2],
                        tension: 0.25,
                        fill: false
                    }},
                    {{
                        label: 'F1 Index',
                        data: [STATS.yearly['2023-24'].f1, STATS.yearly['2024-25'].f1, STATS.yearly['2025-26'].f1],
                        borderColor: '#9467bd',
                        borderWidth: 2.5,
                        tension: 0.25,
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e5e7eb', font: {{ family: 'Plus Jakarta Sans', weight: '600' }} }}
                    }}
                }},
                scales: {{
                    y: {{
                        min: 75,
                        max: 100,
                        grid: {{ color: 'rgba(255,255,255,0.05)' }},
                        ticks: {{ color: '#9ca3af', callback: val => val + '%' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }}
            }}
        }});

        // 2. Certainty analysis chart
        const certaintyCtx = document.getElementById('certaintyChart').getContext('2d');
        const certTiers = ['Very High', 'High', 'Moderate', 'Low', 'Very Low'];
        const certAccs = certTiers.map(t => {{
            const obj = STATS.certainty.find(c => c.certainty_tier === t);
            return obj ? obj.accuracy : 0;
        }});
        new Chart(certaintyCtx, {{
            type: 'bar',
            data: {{
                labels: certTiers,
                datasets: [{{
                    label: 'Accuracy',
                    data: certAccs,
                    backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'],
                    borderRadius: 8,
                    barThickness: 32
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        max: 100,
                        grid: {{ color: 'rgba(255,255,255,0.05)' }},
                        ticks: {{ color: '#9ca3af', callback: val => val + '%' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }}
            }}
        }});

        // 3. Stability Transition Chart
        const transCtx = document.getElementById('transitionChart').getContext('2d');
        const paths = ['Active -> Active -> Active', 'Inactive -> Inactive -> Inactive'];
        const pathCounts = paths.map(p => STATS.transitions[p] ? STATS.transitions[p].count : 0);
        const pathAccs = paths.map(p => STATS.transitions[p] ? STATS.transitions[p].accuracy : 0);
        new Chart(transCtx, {{
            type: 'bar',
            data: {{
                labels: ['Stable Crops', 'Stable Inactive'],
                datasets: [
                    {{
                        label: 'Total Farms',
                        data: pathCounts,
                        backgroundColor: 'rgba(59, 130, 246, 0.4)',
                        borderColor: '#3b82f6',
                        borderWidth: 1,
                        borderRadius: 6,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'AI Matching Accuracy',
                        data: pathAccs,
                        type: 'line',
                        borderColor: '#10b981',
                        backgroundColor: '#10b981',
                        borderWidth: 3,
                        pointRadius: 6,
                        yAxisID: 'yPercent'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e5e7eb' }}
                    }}
                }},
                scales: {{
                    y: {{
                        title: {{ display: true, text: 'Farms Count', color: '#9ca3af' }},
                        grid: {{ color: 'rgba(255,255,255,0.05)' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    yPercent: {{
                        position: 'right',
                        title: {{ display: true, text: 'Accuracy (%)', color: '#9ca3af' }},
                        min: 50,
                        max: 100,
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af', callback: val => val + '%' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }}
            }}
        }});

        // 4. Regional Performance State Bar Chart
        const stateCtx = document.getElementById('stateChart').getContext('2d');
        const stateLabels = Object.keys(STATS.state_counts).slice(0, 8); // Top 8 states
        const yearsList = ['2023-24', '2024-25', '2025-26'];
        
        const stateDatasets = yearsList.map((y, idx) => {{
            const colors = ['#4c78a8', '#f58518', '#e15759'];
            return {{
                label: y,
                data: stateLabels.map(s => {{
                    const match = STATS.states.find(st => st.state === s && st.year === y);
                    return match ? match.accuracy : 0;
                }}),
                backgroundColor: colors[idx],
                borderRadius: 4
            }};
        }});

        new Chart(stateCtx, {{
            type: 'bar',
            data: {{
                labels: stateLabels,
                datasets: stateDatasets
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#e5e7eb' }} }}
                }},
                scales: {{
                    y: {{
                        max: 100,
                        grid: {{ color: 'rgba(255,255,255,0.05)' }},
                        ticks: {{ color: '#9ca3af', callback: val => val + '%' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }}
            }}
        }});

        // Populate State Leaderboard
        const stateAccMap = {{}};
        STATS.states.forEach(st => {{
            if (!stateAccMap[st.state]) {{
                stateAccMap[st.state] = {{ sum: 0, count: 0 }};
            }}
            stateAccMap[st.state].sum += st.accuracy;
            stateAccMap[st.state].count += 1;
        }});
        
        const rankedStates = Object.keys(stateAccMap)
            .map(s => ({{ state: s, avgAcc: stateAccMap[s].sum / stateAccMap[s].count }}))
            .sort((a, b) => b.avgAcc - a.avgAcc)
            .slice(0, 10);

        const lbContainer = document.getElementById('state-leaderboard');
        rankedStates.forEach(rs => {{
            const itemHtml = `
                <div class="leaderboard-item">
                    <div class="state-name-tag">${{rs.state}}</div>
                    <div class="state-progress-container">
                        <div class="state-progress-labels">
                            <span style="color: var(--text-secondary)">Accuracy index</span>
                            <span style="color: var(--text-primary); font-weight:600">${{rs.avgAcc.toFixed(1)}}%</span>
                        </div>
                        <div class="state-progress-bar-bg">
                            <div class="state-progress-bar-fill" style="width: ${{rs.avgAcc}}%"></div>
                        </div>
                    </div>
                </div>
            `;
            lbContainer.insertAdjacentHTML('beforeend', itemHtml);
        }});

        // Populate State filter options
        const stateSelect = document.getElementById('stateFilter');
        const sortedStates = Object.keys(STATS.state_counts).sort();
        sortedStates.forEach(s => {{
            const opt = document.createElement('option');
            opt.value = s;
            opt.innerText = s;
            stateSelect.appendChild(opt);
        }});


        // ------------------ FARM TABLE EXPLORER PAGINATION ------------------
        let currentPage = 1;
        const rowsPerPage = 15;
        let filteredFarmsList = [...STATS.farms];

        function populateTable() {{
            const tbody = document.getElementById('farmTableBody');
            tbody.innerHTML = '';

            const startIndex = (currentPage - 1) * rowsPerPage;
            const endIndex = Math.min(startIndex + rowsPerPage, filteredFarmsList.length);
            
            const pageData = filteredFarmsList.slice(startIndex, endIndex);

            pageData.forEach(farm => {{
                // count matches (exclude errors from perfect correct matches counts)
                const matchCount = farm.match.filter((val, idx) => farm.pred[idx] !== "Error" && val).length;
                let accuracyClass = 'accuracy-3';
                if (matchCount === 2) accuracyClass = 'accuracy-2';
                if (matchCount <= 1) accuracyClass = 'accuracy-1';

                const getBadgeClass = (pred) => {{
                    if (pred === "Active") return 'badge-active';
                    if (pred === "Inactive") return 'badge-inactive';
                    return 'badge-error';
                }};

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight: 600;">${{farm.farm_name}}</td>
                    <td><span class="badge" style="background:rgba(255,255,255,0.05); color:#60a5fa">${{farm.state}}</span></td>
                    <td>
                        <span class="badge ${{farm.gt[0] === 'Active' ? 'badge-active' : 'badge-inactive'}}">
                            ${{farm.gt[0] === 'Active' ? 'Stable Crop' : 'Stable Inactive'}}
                        </span>
                    </td>
                    <td>
                        <span class="badge ${{getBadgeClass(farm.pred[0])}}">${{farm.pred[0]}}</span>
                        <span style="font-size:0.7rem; color:var(--text-secondary)">(${{farm.cert[0]}})</span>
                        <span class="badge ${{farm.match[0] ? 'badge-match' : 'badge-mismatch'}}">${{farm.pred[0] === 'Error' ? '⚠️' : farm.match[0] ? '✅' : '❌'}}</span>
                    </td>
                    <td>
                        <span class="badge ${{getBadgeClass(farm.pred[1])}}">${{farm.pred[1]}}</span>
                        <span style="font-size:0.7rem; color:var(--text-secondary)">(${{farm.cert[1]}})</span>
                        <span class="badge ${{farm.match[1] ? 'badge-match' : 'badge-mismatch'}}">${{farm.pred[1] === 'Error' ? '⚠️' : farm.match[1] ? '✅' : '❌'}}</span>
                    </td>
                    <td>
                        <span class="badge ${{getBadgeClass(farm.pred[2])}}">${{farm.pred[2]}}</span>
                        <span style="font-size:0.7rem; color:var(--text-secondary)">(${{farm.cert[2]}})</span>
                        <span class="badge ${{farm.match[2] ? 'badge-match' : 'badge-mismatch'}}">${{farm.pred[2] === 'Error' ? '⚠️' : farm.match[2] ? '✅' : '❌'}}</span>
                    </td>
                    <td>
                        <span class="accuracy-pill ${{accuracyClass}}">${{matchCount}}/3 Years</span>
                    </td>
                `;
                tbody.appendChild(tr);
            }});

            // Update page info
            document.getElementById('pageInfo').innerText = `Showing ${{filteredFarmsList.length === 0 ? 0 : startIndex + 1}} - ${{endIndex}} of ${{filteredFarmsList.length}} entries`;
            
            // Buttons
            document.getElementById('prevBtn').disabled = currentPage === 1;
            document.getElementById('nextBtn').disabled = endIndex >= filteredFarmsList.length;
        }}

        function filterFarms() {{
            const searchVal = document.getElementById('searchInput').value.toLowerCase().trim();
            const stateVal = document.getElementById('stateFilter').value;
            const stabilityVal = document.getElementById('stabilityFilter').value;

            filteredFarmsList = STATS.farms.filter(farm => {{
                const matchesSearch = farm.farm_name.toLowerCase().includes(searchVal);
                const matchesState = stateVal === 'ALL' || farm.state === stateVal;
                
                const matchCount = farm.match.filter((val, idx) => farm.pred[idx] !== "Error" && val).length;
                const matchesStability = stabilityVal === 'ALL' || matchCount == stabilityVal;

                return matchesSearch && matchesState && matchesStability;
            }});

            currentPage = 1;
            populateTable();
        }}

        function nextPage() {{
            if ((currentPage * rowsPerPage) < filteredFarmsList.length) {{
                currentPage++;
                populateTable();
            }}
        }}

        function prevPage() {{
            if (currentPage > 1) {{
                currentPage--;
                populateTable();
            }}
        }}

        // Initialize table
        populateTable();
    </script>
</body>
</html>
"""

with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Successfully compiled interactive glassmorphism dashboard in: {output_file}")

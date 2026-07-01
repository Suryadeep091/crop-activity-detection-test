import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f9f9f9',
    'grid.color': '#eeeeee',
    'grid.linewidth': 0.5
})

# Path configurations
JSON_FILE = '2023-26_analysis.json'
ARTIFACT_DIR = r"C:\Users\Suryadeep Singh\.gemini\antigravity-ide\brain\ea250939-4c8f-45b7-a97e-d60cb3693660"
ARTIFACT_DIR = os.path.abspath(ARTIFACT_DIR)
ASSETS_DIR = os.path.join(ARTIFACT_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# 1. Load Data
if not os.path.exists(JSON_FILE):
    raise FileNotFoundError(f"Error: {JSON_FILE} not found in workspace.")

with open(JSON_FILE, "r") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} total records.")

# 2. Parse Records
parsed_records = []
errors = []

# Gather all unique farm base names first to ensure all 306 are represented
all_base_names = set()
for entry in raw_data:
    task_id = entry.get("task_id", "")
    if task_id.startswith("24-"):
        base_name = task_id[3:]
    elif task_id.startswith("25-"):
        base_name = task_id[3:]
    else:
        base_name = task_id
    all_base_names.add(base_name)

print(f"Identified {len(all_base_names)} unique farm base names.")

for entry in raw_data:
    task_id = entry.get("task_id", "")
    status = entry.get("status", "")
    
    # Identify Year & Base Name
    if task_id.startswith("24-"):
        year = "2023-24"
        base_name = task_id[3:]
    elif task_id.startswith("25-"):
        year = "2024-25"
        base_name = task_id[3:]
    else:
        year = "2025-26"
        base_name = task_id
        
    # Extract State code (first 2 letters of base_name)
    state_match = re.match(r"^([A-Z]{2})", base_name)
    state = state_match.group(1) if state_match else "UNKNOWN"
    
    # Ground Truth: True if contains (Crop) else False if (No-Crop)
    ground_truth = True if "(Crop)" in base_name else False
    
    if status != "success":
        errors.append({
            "task_id": task_id,
            "year": year,
            "base_name": base_name,
            "state": state,
            "ground_truth": ground_truth,
            "error_msg": entry.get("error_detail") or entry.get("message") or "Unknown error"
        })
        # Add a record with is_active = None to preserve the row in pivot tables!
        parsed_records.append({
            "task_id": task_id,
            "year": year,
            "base_name": base_name,
            "state": state,
            "ground_truth": ground_truth,
            "is_active": None,
            "activity_score": "N/A",
            "confidence": 0.0,
            "certainty_tier": "N/A",
            "report_url": "",
            "crop_cycles_count": None,
            "detected_seasons": [],
            "status": "failed"
        })
        continue
        
    # Parse Confidence Score
    conf_str = entry.get("final_confidence_score", "0%")
    try:
        conf_val = float(conf_str.replace('%', '').strip())
    except (ValueError, AttributeError):
        conf_val = 0.0
        
    # Parse Certainty Tier
    certainty_tier = entry.get("certainty_tier")
    if not certainty_tier:
        if conf_val >= 80:
            certainty_tier = "Very High"
        elif conf_val >= 65:
            certainty_tier = "High"
        elif conf_val >= 50:
            certainty_tier = "Moderate"
        elif conf_val >= 35:
            certainty_tier = "Low"
        else:
            certainty_tier = "Very Low"
            
    parsed_records.append({
        "task_id": task_id,
        "year": year,
        "base_name": base_name,
        "state": state,
        "ground_truth": ground_truth,
        "is_active": entry.get("is_active", False),
        "activity_score": entry.get("activity_score", "0%"),
        "confidence": conf_val,
        "certainty_tier": certainty_tier,
        "report_url": entry.get("report_url", ""),
        "crop_cycles_count": entry.get("crop_cycles_count", None),
        "detected_seasons": entry.get("detected_seasons", []),
        "status": "success"
    })

df = pd.DataFrame(parsed_records)
print(f"Successfully processed {len(df)} records (includes failures as sentinel entries).")

# 3. Year-wise Statistical Metrics (excluding failed entries for mathematical fairness)
yearly_stats = {}
df_success = df[df["status"] == "success"]
for name, group in df_success.groupby("year"):
    tp = sum((group["ground_truth"] == True) & (group["is_active"] == True))
    fp = sum((group["ground_truth"] == False) & (group["is_active"] == True))
    tn = sum((group["ground_truth"] == False) & (group["is_active"] == False))
    fn = sum((group["ground_truth"] == True) & (group["is_active"] == False))
    
    total = len(group)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    yearly_stats[name] = {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "total": total
    }

print("\n--- Year-Wise Core Metrics (Success-only) ---")
for year, stats in yearly_stats.items():
    print(f"Year {year}: Accuracy = {stats['accuracy']:.2f}%, F1 = {stats['f1']:.2f}% (TP={stats['TP']}, TN={stats['TN']}, FP={stats['FP']}, FN={stats['FN']})")

# 4. State-wise Performance Analysis
state_stats = []
for (state, year), group in df_success.groupby(["state", "year"]):
    correct = sum(group["ground_truth"] == group["is_active"])
    total = len(group)
    state_stats.append({
        "state": state,
        "year": year,
        "accuracy": (correct / total) * 100,
        "total": total
    })
df_state = pd.DataFrame(state_stats)

# 5. Certainty Tier Analysis
cert_stats = []
for certainty, group in df_success.groupby("certainty_tier"):
    correct = sum(group["ground_truth"] == group["is_active"])
    total = len(group)
    cert_stats.append({
        "certainty_tier": certainty,
        "accuracy": (correct / total) * 100,
        "count": total
    })
df_cert = pd.DataFrame(cert_stats)

# 6. Farm Transitions Over 3 Years
pivot_gt = df.pivot(index="base_name", columns="year", values="ground_truth")
pivot_pred = df.pivot(index="base_name", columns="year", values="is_active")
pivot_cert = df.pivot(index="base_name", columns="year", values="certainty_tier")

# Transition pathways for all 306 farms (using exact Ground Truth from task naming structure)
transition_profiles = {}
for base in all_base_names:
    gt_val = True if "(Crop)" in base else False
    gt_path = (gt_val, gt_val, gt_val)
    
    # Check predictions
    pred_list = []
    for year in ["2023-24", "2024-25", "2025-26"]:
        val = pivot_pred.loc[base, year] if base in pivot_pred.index else None
        pred_list.append(val)
    pred_path = tuple(pred_list)
    
    path_name = " -> ".join(["Active" if x else "Inactive" for x in gt_path])
    
    if path_name not in transition_profiles:
        transition_profiles[path_name] = {"count": 0, "correct_matches": 0}
        
    transition_profiles[path_name]["count"] += 1
    # Match is True if all 3 predictions equal ground truth
    if pred_path == gt_path:
        transition_profiles[path_name]["correct_matches"] += 1

for path, info in transition_profiles.items():
    info["accuracy"] = (info["correct_matches"] / info["count"]) * 100

print(f"\nFarms represented in 3-year stability pathways: {sum(info['count'] for info in transition_profiles.values())}")
for path, info in sorted(transition_profiles.items(), key=lambda x: x[1]['count'], reverse=True):
    print(f"Path [{path}]: Farms = {info['count']}, Model Match Accuracy = {info['accuracy']:.2f}%")


# ==========================================
# GENERATE PLOTS
# ==========================================

# Chart 1: Yearly Accuracy, Precision, Recall, F1 Comparison
fig, ax = plt.subplots(figsize=(9, 5.5))
metrics_df = pd.DataFrame(yearly_stats).T[["accuracy", "precision", "recall", "f1"]]
metrics_df.plot(kind="bar", color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"], ax=ax, width=0.75)
ax.set_title("Multi-Year Satellite Analytics Core Performance Comparison", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Metric Value (%)", fontsize=12)
ax.set_xlabel("Cohort Year", fontsize=12)
ax.set_ylim(0, 105)
ax.legend(["Accuracy", "Precision", "Recall", "F1 Score"], loc="lower right", framealpha=0.9, facecolor="#ffffff")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1.5),
                ha='center', va='center', xytext=(0, 3), textcoords='offset points', fontsize=9, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "yearly_accuracy.png"), dpi=200)
plt.close(fig)

# Chart 2: Year-wise Confusion Matrices (Heatmaps side-by-side)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
years = ["2023-24", "2024-25", "2025-26"]
for i, year in enumerate(years):
    stats = yearly_stats[year]
    cm = np.array([[stats["TN"], stats["FP"]], 
                   [stats["FN"], stats["TP"]]])
    
    im = axes[i].imshow(cm, cmap="Blues", interpolation="nearest", vmin=0, vmax=max(df_success["ground_truth"].value_counts()))
    
    # Add text annotations inside cells
    for row in range(2):
        for col in range(2):
            cell_val = cm[row, col]
            color = "white" if cell_val > (cm.max() / 2) else "black"
            axes[i].text(col, row, str(cell_val), ha="center", va="center", fontsize=14, fontweight="bold", color=color)
            
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(["Inactive", "Active"], fontsize=10)
    axes[i].set_yticks([0, 1])
    axes[i].set_yticklabels(["Inactive", "Active"], fontsize=10)
    
    axes[i].set_title(f"Confusion Matrix: {year}", fontsize=12, fontweight="bold", pad=10)
    axes[i].set_xlabel("AI Predicted Label", fontsize=10)
    axes[i].set_ylabel("Ground Truth Label", fontsize=10)
    axes[i].grid(False)
    
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "confusion_matrices.png"), dpi=200)
plt.close(fig)

# Chart 3: Accuracy by Certainty Tier
fig, ax = plt.subplots(figsize=(8, 5))
tier_order = ["Very High", "High", "Moderate", "Low", "Very Low"]
df_cert['certainty_tier'] = pd.Categorical(df_cert['certainty_tier'], categories=tier_order, ordered=True)
df_cert = df_cert.sort_values('certainty_tier')

color_map = ["#2ca02c", "#7f7f7f", "#ff7f0e", "#d62728", "#9467bd"]
bars = ax.bar(df_cert["certainty_tier"], df_cert["accuracy"], color=color_map[:len(df_cert)], edgecolor="#444444", width=0.55)
ax.set_title("Prediction Accuracy vs. Model Certainty Tier", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_xlabel("Certainty Tier", fontsize=11)
ax.set_ylim(0, 105)
for bar, row in zip(bars, df_cert.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, 
            f"{row.accuracy:.1f}%\n(N={row.count})", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "certainty_analysis.png"), dpi=200)
plt.close(fig)

# Chart 4: State-wise Accuracy Comparison (Top 8 states with most farms)
top_states = df_success["state"].value_counts().head(8).index
df_state_top = df_state[df_state["state"].isin(top_states)]
pivot_state = df_state_top.pivot(index="state", columns="year", values="accuracy")

fig, ax = plt.subplots(figsize=(10, 5.5))
pivot_state.plot(kind="bar", color=["#4c78a8", "#f58518", "#e15759"], width=0.75, ax=ax)
ax.set_title("Year-Wise Accuracy Comparison Across Top 8 States", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Model Accuracy (%)", fontsize=11)
ax.set_xlabel("State", fontsize=11)
ax.set_ylim(0, 105)
ax.legend(title="Year", framealpha=0.9, facecolor="#ffffff")
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.0f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1.5),
                    ha='center', va='center', xytext=(0, 2), textcoords='offset points', fontsize=8, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "state_performance.png"), dpi=200)
plt.close(fig)

# Chart 5: Farm Activity Transition Profile counts
fig, ax = plt.subplots(figsize=(8, 5))
paths = list(transition_profiles.keys())
counts = [info["count"] for info in transition_profiles.values()]
accs = [info["accuracy"] for info in transition_profiles.values()]

# Sort by count
sorted_indices = np.argsort(counts)[::-1]
paths_sorted = [paths[i] for i in sorted_indices]
counts_sorted = [counts[i] for i in sorted_indices]
accs_sorted = [accs[i] for i in sorted_indices]

bars = ax.barh(paths_sorted, counts_sorted, color="#8da0cb", edgecolor="#555555", height=0.6)
ax.set_title("3-Year Farm Activity Transition Pathways (2023-26)", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Number of Farms (Total = 306)", fontsize=11)
ax.set_ylabel("Activity Pathway (GT: '23-24 -> '24-25 -> '25-26')", fontsize=11)

for bar, acc in zip(bars, accs_sorted):
    width = bar.get_width()
    ax.text(width + 2, bar.get_y() + bar.get_height()/2., f"{width} farms (AI match: {acc:.1f}%)", 
            ha="left", va="center", fontsize=8.5, fontweight="bold", color="#333333")
ax.set_xlim(0, max(counts_sorted) + 40)
plt.gca().invert_yaxis()  # Invert y-axis to show largest on top
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "activity_transitions.png"), dpi=200)
plt.close(fig)

# 7. Compile the complete table of all 306 farms with year-wise predicted activity and certainty tier!
farm_records = []
for base in sorted(all_base_names):
    # Ground Truth (Static)
    gt_val = "Active" if "(Crop)" in base else "Inactive"
    
    # Extract state from base
    state_match = re.match(r"^([A-Z]{2})", base)
    state = state_match.group(1) if state_match else "UNKNOWN"
    
    gt_list = [gt_val, gt_val, gt_val]
    pred_list = []
    cert_list = []
    match_list = []
    
    for year in ["2023-24", "2024-25", "2025-26"]:
        # Find item in df
        match_rows = df[(df["base_name"] == base) & (df["year"] == year)]
        if len(match_rows) > 0:
            row = match_rows.iloc[0]
            if row["status"] == "failed":
                pred_val = "Error"
                cert_val = "N/A"
                is_match = False
            else:
                pred_val = "Active" if row["is_active"] else "Inactive"
                cert_val = row["certainty_tier"]
                is_match = (gt_val == pred_val)
        else:
            pred_val = "Error"
            cert_val = "N/A"
            is_match = False
            
        pred_list.append(pred_val)
        cert_list.append(cert_val)
        match_list.append(is_match)
        
    farm_records.append({
        "farm_name": base,
        "state": state,
        "gt": gt_list,
        "pred": pred_list,
        "cert": cert_list,
        "match": match_list
    })

stats_data = {
    "yearly": yearly_stats,
    "states": df_state.to_dict(orient="records"),
    "certainty": df_cert.to_dict(orient="records"),
    "transitions": {path: {"count": info["count"], "accuracy": info["accuracy"]} for path, info in transition_profiles.items()},
    "errors": errors,
    "state_counts": df_success["state"].value_counts().to_dict(),
    "completed_farms_count": len(all_base_names),
    "farms": farm_records
}

with open("analysis_stats.json", "w") as f:
    json.dump(stats_data, f, indent=4)

print("\nAll analysis metrics computed successfully!")
print(f"Generated 5 premium-quality figures in: {ASSETS_DIR}")
print("Saved calculated data to: analysis_stats.json")

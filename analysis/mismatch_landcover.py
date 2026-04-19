import json
import os

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
results_file = os.path.join(downloads_path, "batch_results_300_Final.json")

def analyze_mismatches():
    with open(results_file, "r") as f:
        data = json.load(f)

    fps = []
    fns = []

    for entry in data:
        task_id = entry.get("task_id", "")
        if "(Crop)" in task_id:
            ground_truth = True
        elif "(No-Crop)" in task_id:
            ground_truth = False
        else:
            continue

        is_active = entry.get("is_active", False)
        
        if ground_truth != is_active:
            lc = entry.get("land use/ land cover details", {})
            trees = lc.get("trees", {}).get("percent", 0)
            built = lc.get("built", {}).get("percent", 0)
            crops = lc.get("crops", {}).get("percent", 0)
            water = lc.get("water", {}).get("percent", 0)
            grass = lc.get("grass", {}).get("percent", 0)
            flooded = lc.get("flooded_vegetation", {}).get("percent", 0)

            mismatch_data = {
                "id": task_id,
                "cycles": entry.get("crop_cycles_count", 0),
                "p1_crop": entry.get("p1_avg_conf", "0%"),
                "p2_crop": entry.get("p2_avg_conf", "0%"),
                "LC": f"Crops:{crops}% Trees:{trees}% Built:{built}% Water:{water}% Grass:{grass}% Flood:{flooded}%"
            }
            if ground_truth and not is_active:
                fns.append(mismatch_data)
            elif not ground_truth and is_active:
                fps.append(mismatch_data)

    print("--- FALSE POSITIVES (Truth=No-Crop, Pred=Crop) ---")
    for fp in fps:
        print(f"{fp['id']:<15} | Cyc:{fp['cycles']} | P1:{fp['p1_crop']:<6} | P2:{fp['p2_crop']:<6} | {fp['LC']}")

    print("\n--- FALSE NEGATIVES (Truth=Crop, Pred=No-Crop) ---")
    for fn in fns:
         print(f"{fn['id']:<15} | Cyc:{fn['cycles']} | P1:{fn['p1_crop']:<6} | P2:{fn['p2_crop']:<6} | {fn['LC']}")

if __name__ == "__main__":
    analyze_mismatches()

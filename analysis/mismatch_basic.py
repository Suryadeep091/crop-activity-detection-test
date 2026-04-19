import json
import os

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
results_file = os.path.join(downloads_path, "batch_results_300_Final.json")

def analyze_mismatches():
    with open(results_file, "r") as f:
        data = json.load(f)

    mismatches = []
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
            mismatch_data = {
                "id": task_id,
                "truth": "Crop" if ground_truth else "No-Crop",
                "pred": "Crop" if is_active else "No-Crop",
                "score": entry.get("activity_score", "0%"),
                "cycles": entry.get("crop_cycles_count", 0),
                "seasons": entry.get("detected_seasons", []),
                "p1_crop": entry.get("p1_avg_conf", "0%"),
                "p2_crop": entry.get("p2_avg_conf", "0%")
            }
            mismatches.append(mismatch_data)
            if ground_truth and not is_active:
                fns.append(mismatch_data)
            elif not ground_truth and is_active:
                fps.append(mismatch_data)

    print(f"Total Mismatches: {len(mismatches)}")
    print(f"False Positives (AI says Crop, Truth is No-Crop): {len(fps)}")
    for fp in fps:
        print(fp)
        
    print("\n")
    print(f"False Negatives (AI says No-Crop, Truth is Crop): {len(fns)}")
    for fn in fns:
        print(fn)

if __name__ == "__main__":
    analyze_mismatches()

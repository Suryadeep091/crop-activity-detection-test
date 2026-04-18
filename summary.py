import json
import os

# --- CONFIGURATION ---
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
results_file = os.path.join(downloads_path, "batch_results_300_Whittaker.json")
comparison_report = os.path.join(downloads_path, "accuracy_comparison_300_Whittaker.txt")

def generate_accuracy_report():
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {results_file} not found.")
        return

    print(f"📊 Analyzing Accuracy for {len(data)} parcels...")
    print("-" * 60)

    comparisons = []
    correct_count = 0

    for entry in data:
        task_id = entry.get("task_id", "")
        # Extract Ground Truth from Task ID (e.g., "UP4 (Crop)" -> "Crop")
        if "(Crop)" in task_id:
            ground_truth = "Active"
        elif "(No-Crop)" in task_id:
            ground_truth = "Inactive"
        else:
            ground_truth = "Unknown"

        # Map AI Verdict to binary
        is_active = entry.get("is_active", False)
        ai_result = "Active" if is_active else "Inactive"
        
        # Check for match
        match = (ground_truth == ai_result)
        if match:
            correct_count += 1

        comparisons.append({
            "task_id": task_id,
            "ground_truth": ground_truth,
            "ai_result": ai_result,
            "score": entry.get("activity_score", "0%"),
            "match": "✅ MATCH" if match else "❌ MISMATCH"
        })

    # Calculate Accuracy
    accuracy = (correct_count / len(data)) * 100 if data else 0

    # Print & Save Results
    with open(comparison_report, "w", encoding="utf-8") as f:
        header = f"{'TASK ID':<20} | {'TRUTH':<10} | {'AI':<10} | {'SCORE':<8} | {'STATUS'}"
        print(header)
        f.write(header + "\n" + "-"*70 + "\n")
        
        for c in comparisons:
            line = f"{c['task_id']:<20} | {c['ground_truth']:<10} | {c['ai_result']:<10} | {c['score']:<8} | {c['match']}"
            print(line)
            f.write(line + "\n")

        summary = f"\n{'='*30}\nOVERALL ACCURACY: {accuracy:.2f}%\n{'='*30}"
        print(summary)
        f.write(summary)

    print(f"\n📝 Detailed report saved to: {comparison_report}")

if __name__ == "__main__":
    generate_accuracy_report()
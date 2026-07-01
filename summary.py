import json
import os

# --- CONFIGURATION ---
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
results_file = os.path.join(downloads_path, "dummy_run_2.json")
comparison_report = os.path.join(downloads_path, "24-25_analysis.txt")

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
    review_count = 0

    for entry in data:
        task_id = entry.get("task_id", "")
        # Extract Ground Truth from Task ID (e.g., "UP4 (Crop)" -> "Crop")
        if "(Crop)" in task_id:
            ground_truth = "Active"
        elif "(No-Crop)" in task_id:
            ground_truth = "Inactive"
        else:
            ground_truth = "Unknown"

        # Preserve the internal Review state when the newer policy returns it.
        decision_label = str(entry.get("decision_label") or "").strip()
        if decision_label.lower() == "review":
            ai_result = "Review"
            review_count += 1
        else:
            is_active = entry.get("is_active", False)
            ai_result = "Active" if is_active else "Inactive"
        
        # Check for match
        match = (ground_truth == ai_result)
        if match:
            correct_count += 1

        final_conf_str = entry.get("final_confidence_score", "0%")
        conf_level = str(entry.get("certainty_tier") or "").strip()
        try:
            conf_val = float(final_conf_str.replace('%', '').strip())
        except ValueError:
            conf_val = 0.0

        if not conf_level:
            if conf_val >= 80:
                conf_level = "Very High"
            elif conf_val >= 65:
                conf_level = "High"
            elif conf_val >= 50:
                conf_level = "Moderate"
            elif conf_val >= 35:
                conf_level = "Low"
            else:
                conf_level = "Very Low"

        comparisons.append({
            "task_id": task_id,
            "ground_truth": ground_truth,
            "ai_result": ai_result,
            "score": entry.get("activity_score", "0%"),
            "final_conf": final_conf_str,
            "conf_level": conf_level,
            "match": "✅ MATCH" if match else "❌ MISMATCH"
        })

    # Calculate Accuracy
    accuracy = (correct_count / len(data)) * 100 if data else 0

    # Print & Save Results
    with open(comparison_report, "w", encoding="utf-8") as f:
        header = f"{'TASK ID':<20} | {'TRUTH':<10} | {'AI':<10} | {'SCORE':<8} | {'CONF':<8} | {'CERT_TIER':<12} | {'STATUS'}"
        print(header)
        f.write(header + "\n" + "-"*90 + "\n")
        
        for c in comparisons:
            line = f"{c['task_id']:<20} | {c['ground_truth']:<10} | {c['ai_result']:<10} | {c['score']:<8} | {c['final_conf']:<8} | {c['conf_level']:<12} | {c['match']}"
            print(line)
            f.write(line + "\n")

        summary = (
            f"\n{'='*30}\n"
            f"OVERALL ACCURACY: {accuracy:.2f}%\n"
            f"REVIEW CASES: {review_count}\n"
            f"{'='*30}"
        )
        print(summary)
        f.write(summary)

    print(f"\n📝 Detailed report saved to: {comparison_report}")

if __name__ == "__main__":
    generate_accuracy_report()
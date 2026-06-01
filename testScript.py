import requests
import json
import time
import os
import traceback

# --- CONFIGURATION ---
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
input_json_path = os.path.join(downloads_path, "test_parcels_300.json")
output_json_path = os.path.join(downloads_path, "live_batch_2026.json")

# TARGETING THE LIVE EXTRACTION ENDPOINT
API_URL = "https://test-terradrishti-413500342905.asia-south1.run.app/test/accuracy"

def run_live_batch_test():
    # 1. Load the parcels
    try:
        with open(input_json_path, "r") as f:
            parcels = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {input_json_path} not found.")
        return

    master_results = []
    print(f"🚀 Starting LIVE BATCH TEST for {len(parcels)} parcels...")
    print("⚠️  Note: This involves GEE calls and will take longer.")
    print("-" * 65)

    # 2. Iterate through parcels
    for index, parcel in enumerate(parcels):
        task_id = parcel.get("task_id")
        kml = parcel.get("kml_coordinates")
        
        print(f"[{index + 1}/{len(parcels)}] Processing: {task_id}...")

        payload = {
            "task_id": task_id,
            "kml_coordinates": kml,
            "end_date": "2026-05-25" # Consistent date for accuracy benchmarking
        }

        try:
            # High timeout (300s) because GEE extraction can be slow
            response = requests.post(API_URL, json=payload, timeout=300)

            if response.status_code == 200:
                data = response.json()
                
                # Print summary to console
                verdict = data.get("verdict", "N/A")
                score = data.get("activity_score", "N/A")
                print(f" ✅ SUCCESS | Verdict: {verdict} ({score})")
                
                # Append full response to master list
                master_results.append(data)
            else:
                error_entry = {
                    "task_id": task_id,
                    "status": "failed",
                    "http_code": response.status_code,
                    "error_detail": response.text
                }
                master_results.append(error_entry)
                print(f" ❌ FAILED | Status: {response.status_code}")

        except Exception as e:
            print(f" ⚠️ Connection Error: {e}")
            master_results.append({"task_id": task_id, "status": "error", "message": str(e)})

        # 3. Save progress incrementally (Safety feature)
        # This ensures that if the script crashes at parcel #50, you don't lose the first 49
        with open(output_json_path, "w") as f:
            json.dump(master_results, f, indent=4)

        # 4. Respectful Sleep for GEE Queue
        if index < len(parcels) - 1:
            time.sleep(3)

    print("-" * 65)
    print(f"🏁 Live Batch Test Complete.")
    print(f"📁 Master JSON saved to: {output_json_path}")

if __name__ == "__main__":
    run_live_batch_test()
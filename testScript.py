import requests
import json
import time
import os

# --- CONFIGURATION ---
# Path to the JSON file you generated in your Downloads
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
json_file_path = os.path.join(downloads_path, "test_parcels.json")
API_URL = "https://test-terradrishti-413500342905.asia-south1.run.app/test/accuracy"

def run_batch_accuracy_test():
    # 1. Load the parcels
    try:
        with open(json_file_path, "r") as f:
            parcels = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        return

    print(f"🚀 Starting batch test for {len(parcels)} parcels...")
    print("-" * 50)

    results_log = []

    # 2. Iterate through parcels one by one
    for index, parcel in enumerate(parcels):
        task_id = parcel.get("task_id")
        print(f"[{index + 1}/{len(parcels)}] Processing: {task_id}...")

        try:
            # The payload matches your GeometryRequest model
            payload = {
                "task_id": task_id,
                "kml_coordinates": parcel.get("kml_coordinates"),
                "end_date": "2026-03-15" # Consistent end date for accuracy check
            }

            # Call the API (Waiting for it to finish)
            response = requests.post(API_URL, json=payload, timeout=300) # 5 min timeout for GEE

            if response.status_code == 200:
                data = response.json()
                print(f" ✅ Success | Pickle: {data.get('local_pickle_path')}")
                print(f" 📄 Report: {data.get('report_url')}")
                results_log.append({"task_id": task_id, "status": "Success"})
            else:
                print(f" ❌ Failed | Status: {response.status_code} | Error: {response.text}")
                results_log.append({"task_id": task_id, "status": "Failed", "error": response.text})

        except Exception as e:
            print(f" ⚠️ Connection Error: {e}")
            results_log.append({"task_id": task_id, "status": "Error", "error": str(e)})

        # 3. Respectful Sleep (5 seconds)
        # This allows GEE to clear the high-volume queue for the next parcel
        if index < len(parcels) - 1:
            print(f"Sleeping for 5 seconds...")
            time.sleep(5)

    print("-" * 50)
    print(f"🏁 Batch test complete. Check your accuracy_tests folder for the pickles.")

if __name__ == "__main__":
    run_batch_accuracy_test()
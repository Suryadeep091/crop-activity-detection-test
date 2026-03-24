import requests
import json
import time
import os

# --- CONFIGURATION ---
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
input_json_path = os.path.join(downloads_path, "test_parcels.json")
output_json_path = os.path.join(downloads_path, "batch_results1.json") # NEW FILE
BASE_REPLAY_URL = "https://test-terradrishti-413500342905.asia-south1.run.app/test/replay"

def run_batch_replay_with_logging():
    # 1. Load the Task IDs
    try:
        with open(input_json_path, "r") as f:
            parcels = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {input_json_path} not found.")
        return

    master_results = [] # This will hold all your JSON responses
    print(f"🔄 Starting BATCH REPLAY for {len(parcels)} parcels...")
    print("-" * 60)

    # 2. Iterate through parcels
    for index, parcel in enumerate(parcels):
        task_id = parcel.get("task_id")
        if not task_id: continue

        replay_url = f"{BASE_REPLAY_URL}/{task_id}"
        print(f"[{index + 1}/{len(parcels)}] Replaying: {task_id}...")

        try:
            response = requests.post(replay_url, timeout=60) 

            if response.status_code == 200:
                data = response.json()
                print(f" ✅ DONE: {data.get('agri_activity')} ({data.get('activity_score')})")
                
                # Append the full response to our list
                master_results.append(data)
            else:
                error_log = {"task_id": task_id, "status": "failed", "error": response.text}
                master_results.append(error_log)
                print(f" ❌ FAILED: {task_id}")

        except Exception as e:
            master_results.append({"task_id": task_id, "status": "error", "message": str(e)})
            print(f" ⚠️ Error: {e}")

        # Small sleep to be safe
        time.sleep(1)

    # 3. SAVE TO SINGLE JSON FILE
    print("-" * 60)
    try:
        with open(output_json_path, "w") as f:
            json.dump(master_results, f, indent=4)
        print(f"🏁 DONE! All results saved to: {output_json_path}")
    except Exception as e:
        print(f"❌ Failed to save JSON: {e}")

if __name__ == "__main__":
    run_batch_replay_with_logging()
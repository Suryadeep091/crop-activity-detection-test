import requests
import json
import time
import os

# --- CONFIGURATION ---
# We still use the JSON file to get the list of Task IDs to replay
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
json_file_path = os.path.join(downloads_path, "test_parcels.json")

# TARGETING THE REPLAY ENDPOINT
BASE_REPLAY_URL = "https://test-terradrishti-413500342905.asia-south1.run.app/test/replay"

def run_batch_replay_test():
    # 1. Load the parcels to get the Task IDs
    try:
        with open(json_file_path, "r") as f:
            parcels = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {json_file_path} not found.")
        return

    print(f"🔄 Starting BATCH REPLAY for {len(parcels)} parcels...")
    print("🚀 Note: This bypasses GEE and uses saved pickle data.")
    print("-" * 60)

    # 2. Iterate through parcels
    for index, parcel in enumerate(parcels):
        task_id = parcel.get("task_id")
        if not task_id:
            continue

        # Construct the specific URL for this task
        replay_url = f"{BASE_REPLAY_URL}/{task_id}"
        
        print(f"[{index + 1}/{len(parcels)}] Replaying: {task_id}...")

        try:
            # Call the Replay API (No payload needed as it's a path parameter)
            # Timeout can be much lower now (e.g., 60s) since no GEE is involved
            response = requests.post(replay_url, timeout=60) 

            if response.status_code == 200:
                data = response.json()
                pdf_name = data.get('pdf_name')
                status = data.get('agri_activity')
                
                print(f" ✅ DONE: {pdf_name}")
                print(f" 🌾 STATUS: {status}")
                print(f" 🔗 REPORT: {data.get('report_url')}")
            else:
                print(f" ❌ FAILED | Status: {response.status_code}")
                print(f"    Detail: {response.text}")

        except Exception as e:
            print(f" ⚠️ Connection Error: {e}")

        # 3. Respectful Sleep (Optional but recommended)
        # Replay is fast, but 1-2 seconds helps the PDF engine breathe
        if index < len(parcels) - 1:
            time.sleep(2)

    print("-" * 60)
    print(f"🏁 Batch Replay Complete. All reports have been updated in GCS.")

if __name__ == "__main__":
    run_batch_replay_test()
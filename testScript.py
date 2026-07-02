import requests
import json
import time
import os
import traceback
from datetime import datetime  # Added for absolute timestamps

# --- CONFIGURATION ---
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
input_json_path = os.path.join(downloads_path, "_test_parcels_output.json")
output_json_path = os.path.join(downloads_path, "test.json")

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
    print("-" * 75)

    # 2. Iterate through parcels
    for index, parcel in enumerate(parcels):
        task_id = parcel.get("task_id")
        kml = parcel.get("kml_coordinates")
        
        # Capture the precise initialization metrics
        start_time = time.perf_counter()
        call_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{index + 1}/{len(parcels)}] 🕒 Request Sent At: {call_timestamp}")
        print(f"      Processing: {task_id}...")

        payload = {
            "task_id": task_id,
            "kml_coordinates": kml,
            "end_date": "2023-05-25" # Consistent date for accuracy benchmarking
        }

        try:
            # High timeout (300s) because GEE extraction can be slow
            response = requests.post(API_URL, json=payload, timeout=300)
            
            # Calculate elapsed time and capture completion timestamp
            end_time = time.perf_counter()
            duration = end_time - start_time
            report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if response.status_code == 200:
                data = response.json()
                
                # Print comprehensive summary to console
                verdict = data.get("verdict", "N/A")
                score = data.get("activity_score", "N/A")
                print(f"      📥 Report Delivered At: {report_timestamp}")
                print(f"      ⏱️ Duration: {duration:.2f} seconds")
                print(f"      ✅ SUCCESS | Verdict: {verdict} ({score})")
                print("-" * 50)
                
                # Append execution metadata directly inside your payload for tracking
                data["_test_metadata"] = {
                    "call_timestamp": call_timestamp,
                    "delivery_timestamp": report_timestamp,
                    "duration_seconds": round(duration, 2)
                }
                master_results.append(data)
            else:
                error_entry = {
                    "task_id": task_id,
                    "status": "failed",
                    "http_code": response.status_code,
                    "error_detail": response.text,
                    "metadata": {
                        "call_timestamp": call_timestamp,
                        "delivery_timestamp": report_timestamp,
                        "duration_seconds": round(duration, 2)
                    }
                }
                master_results.append(error_entry)
                print(f"      📥 Report Terminated At: {report_timestamp}")
                print(f"      ⏱️ Duration before crash: {duration:.2f} seconds")
                print(f"      ❌ FAILED | Status: {response.status_code}")
                print("-" * 50)

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"      📥 Exception Raised At: {report_timestamp}")
            print(f"      ⏱️ Time elapsed before crash: {duration:.2f} seconds")
            print(f"      ⚠️ Connection Error: {e}")
            print("-" * 50)
            
            master_results.append({
                "task_id": task_id, 
                "status": "error", 
                "message": str(e),
                "metadata": {
                    "call_timestamp": call_timestamp,
                    "delivery_timestamp": report_timestamp,
                    "duration_seconds": round(duration, 2)
                }
            })

        # 3. Save progress incrementally (Safety feature)
        with open(output_json_path, "w") as f:
            json.dump(master_results, f, indent=4)

        # 4. Respectful Sleep for GEE Queue
        if index < len(parcels) - 1:
            time.sleep(3)

    print("-" * 75)
    print(f"🏁 Live Batch Test Complete.")
    print(f"📁 Master JSON saved to: {output_json_path}")

if __name__ == "__main__":
    run_live_batch_test()
"""
Re-run /test/accuracy for all parcels to refresh GCS pickles (raw_vegetation_indices + charts).
Usage: python scripts/repickle_batch.py
"""
import json
import os
import time
import requests

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
input_json_path = os.path.join(downloads_path, "test_parcels_300.json")
output_json_path = os.path.join(downloads_path, "2024-2025.json")
API_URL = "https://test-terradrishti-413500342905.asia-south1.run.app/test/accuracy"


def main():
    with open(input_json_path, "r", encoding="utf-8") as f:
        parcels = json.load(f)

    results = []
    print(f"Re-pickling {len(parcels)} parcels via {API_URL}")

    for index, parcel in enumerate(parcels):
        task_id = parcel.get("task_id")
        kml = parcel.get("kml_coordinates")
        if not task_id or not kml:
            continue

        print(f"[{index + 1}/{len(parcels)}] {task_id}")
        payload = {
            "task_id": task_id,
            "kml_coordinates": kml,
            "end_date": "2025-04-25",
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=300)
            if resp.status_code == 200:
                results.append(resp.json())
                print(f"  OK {resp.json().get('final_confidence_score')} ({resp.json().get('certainty_tier', '')})")
            else:
                results.append({"task_id": task_id, "status": "failed", "error": resp.text})
                print(f"  FAIL {resp.status_code}")
        except Exception as exc:
            results.append({"task_id": task_id, "status": "error", "message": str(exc)})
            print(f"  ERROR {exc}")

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if index < len(parcels) - 1:
            time.sleep(3)

    print(f"Done. Wrote {output_json_path}")


if __name__ == "__main__":
    main()

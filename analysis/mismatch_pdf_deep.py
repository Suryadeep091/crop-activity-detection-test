import os
import json
import PyPDF2
import re

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
results_file = os.path.join(downloads_path, "batch_results_300_Final.json")
pdf_folder = os.path.join(downloads_path, "AdvaRisk - Test", "Cycle_Test_310_Draft_Latest")

def extract_pdf_data(pdf_path):
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = "\n".join([p.extract_text() for p in reader.pages])
    except Exception as e:
        return {"error": str(e)}

    data = {}
    
    # Extract land cover
    for cover in ["Trees", "Crops", "Water", "Built", "Grass", "Shrub And Scrub"]:
        match = re.search(fr"{cover} (\d+\.?\d*)%", text)
        if match:
            data[cover] = float(match.group(1))

    # Extract final confidence string
    conf_match = re.search(r"Final Confidence: (\d+\.?\d*)%", text)
    if conf_match:
         data["Final Conf"] = float(conf_match.group(1))
         
    # Extract prediction lines
    crop_activity_count = len(re.findall(r"Crop-Activity", text))
    no_crop_count = len(re.findall(r"No Crop-Activity", text))
    data["Active Days"] = crop_activity_count
    data["Inactive Days"] = no_crop_count
    
    return data

def main():
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
            mismatch_data = {
                "id": task_id,
                "cycles": entry.get("crop_cycles_count", 0),
                "p1_crop": float(entry.get("p1_avg_conf", "0%").strip('%')),
                "p2_crop": float(entry.get("p2_avg_conf", "0%").strip('%')),
                "score": entry.get("activity_score", "0%"),
            }
            
            pdf_path = os.path.join(pdf_folder, f"test_{task_id}.pdf")
            pdf_data = extract_pdf_data(pdf_path)
            mismatch_data.update(pdf_data)

            if ground_truth and not is_active:
                fns.append(mismatch_data)
            elif not ground_truth and is_active:
                fps.append(mismatch_data)

    print("=== FALSE POSITIVES (Pred: Crop, Truth: No-Crop) ===")
    for c in fps:
        trees = c.get("Trees", "N/A")
        crops = c.get("Crops", "N/A")
        water = c.get("Water", "N/A")
        built = c.get("Built", "N/A")
        grass = c.get("Grass", "N/A")
        active = c.get("Active Days", "N/A")
        inactive = c.get("Inactive Days", "N/A")
        print(f"[{c['id']}] Cyc:{c['cycles']} P1:{c['p1_crop']}% P2:{c['p2_crop']}% Score:{c['score']} | Trees:{trees}% Crops:{crops}% Water:{water}% Built:{built}% Grass:{grass}% | Active:{active} Inactive:{inactive}")

    print("\n=== FALSE NEGATIVES (Pred: No-Crop, Truth: Crop) ===")
    for c in fns:
        trees = c.get("Trees", "N/A")
        crops = c.get("Crops", "N/A")
        water = c.get("Water", "N/A")
        built = c.get("Built", "N/A")
        grass = c.get("Grass", "N/A")
        active = c.get("Active Days", "N/A")
        inactive = c.get("Inactive Days", "N/A")
        print(f"[{c['id']}] Cyc:{c['cycles']} P1:{c['p1_crop']}% P2:{c['p2_crop']}% Score:{c['score']} | Trees:{trees}% Crops:{crops}% Water:{water}% Built:{built}% Grass:{grass}% | Active:{active} Inactive:{inactive}")

if __name__ == "__main__":
    main()

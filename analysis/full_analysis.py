"""Full analysis of ALL 306 parcels — correct + mismatched — to find safe thresholds."""
import os, json, re, PyPDF2

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
results_file = os.path.join(downloads_path, "batch_results_300_Final.json")
pdf_folder = os.path.join(downloads_path, "AdvaRisk - Test", "Cycle_Test_310_Draft_Latest")

def extract_pdf_data(pdf_path):
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
    except:
        return {}
    data = {}
    for cover in ["Trees", "Crops", "Water", "Built", "Grass", "Shrub And Scrub", "Bare", "Flooded Vegetation"]:
        match = re.search(fr"{cover}\s*(\d+\.?\d*)%", text)
        if match: data[cover.lower().replace(" ","_")] = float(match.group(1))
    # Count predictions
    data["crop_days"] = text.count("Crop-Activity") - text.count("No Crop-Activity")
    data["nocrop_days"] = text.count("No Crop-Activity")
    return data

def main():
    with open(results_file, "r") as f:
        entries = json.load(f)

    rows = []
    for entry in entries:
        task_id = entry.get("task_id", "")
        if "(Crop)" in task_id:
            truth = "Crop"
        elif "(No-Crop)" in task_id:
            truth = "NoCrop"
        else:
            continue
        is_active = entry.get("is_active", False)
        pred = "Crop" if is_active else "NoCrop"
        correct = (truth == pred)

        p1 = float(entry.get("p1_avg_conf", "0%").strip('%'))
        p2 = float(entry.get("p2_avg_conf", "0%").strip('%'))
        cycles = entry.get("crop_cycles_count", 0)
        score = entry.get("activity_score", "0%")

        pdf_path = os.path.join(pdf_folder, f"test_{task_id}.pdf")
        pdf = extract_pdf_data(pdf_path) if os.path.exists(pdf_path) else {}

        rows.append({
            "id": task_id, "truth": truth, "pred": pred, "correct": correct,
            "p1": p1, "p2": p2, "cycles": cycles, "score": score,
            **{f"dw_{k}": v for k,v in pdf.items() if k not in ("crop_days","nocrop_days")},
            "crop_days": pdf.get("crop_days", 0),
            "nocrop_days": pdf.get("nocrop_days", 0),
        })

    # Aggregate stats by category
    cats = {
        "CorrectCrop": [r for r in rows if r["correct"] and r["truth"]=="Crop"],
        "CorrectNoCrop": [r for r in rows if r["correct"] and r["truth"]=="NoCrop"],
        "FP": [r for r in rows if not r["correct"] and r["truth"]=="NoCrop"],
        "FN": [r for r in rows if not r["correct"] and r["truth"]=="Crop"],
    }

    out = []
    for cat, items in cats.items():
        out.append(f"\n{'='*80}")
        out.append(f"{cat} ({len(items)} parcels)")
        out.append(f"{'='*80}")
        if not items: continue

        p1s = [r["p1"] for r in items]
        p2s = [r["p2"] for r in items]
        cyc = [r["cycles"] for r in items]
        trees = [r.get("dw_trees",0) for r in items]
        crops = [r.get("dw_crops",0) for r in items]
        grass = [r.get("dw_grass",0) for r in items]
        water = [r.get("dw_water",0) for r in items]
        built = [r.get("dw_built",0) for r in items]

        def s(v): return f"min={min(v):.1f} mean={sum(v)/len(v):.1f} max={max(v):.1f}"
        out.append(f"  P1 Crop   : {s(p1s)}")
        out.append(f"  P2 Crop   : {s(p2s)}")
        out.append(f"  Cycles    : {s(cyc)}")
        out.append(f"  DW Trees  : {s(trees)}")
        out.append(f"  DW Crops  : {s(crops)}")
        out.append(f"  DW Grass  : {s(grass)}")
        out.append(f"  DW Water  : {s(water)}")
        out.append(f"  DW Built  : {s(built)}")

        out.append(f"\n  --- Per Parcel ---")
        for r in items:
            out.append(f"  [{r['id']:<20}] P1={r['p1']:>5.1f}% P2={r['p2']:>5.1f}% Cyc={r['cycles']} Trees={r.get('dw_trees',0):>5.1f}% Crops={r.get('dw_crops',0):>5.1f}% Grass={r.get('dw_grass',0):>5.1f}% Water={r.get('dw_water',0):>5.1f}% Built={r.get('dw_built',0):>5.1f}%")

    report = "\n".join(out)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_306_analysis.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Done! Saved to {out_path}")

if __name__ == "__main__":
    main()

"""
Deep PDF Report Analyzer - Extracts ALL data from 306 TerraDrishti reports.
Parses: verdict, DW classes, daily annexure rows (NDVI/EVI/RVI/Crops/Prediction),
peak metrics, seasons, and activity summary.
"""
import os, re, json, sys
import fitz

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cycle_Test_310_Draft_Latest")

FALSE_POSITIVES = {
    "UP7","MP02","GJ06","GJ09","MH07","AP04","AP05","AP10",
    "HR04","KT09","KT18","JK10","MH20","MP14","MP19","AP13",
    "RJ13","TN05","TN12","WB01"
}
FALSE_NEGATIVES = {
    "PB10","KT02","KT06","KL05","KL06","KL07","OD08","HP10","KT14"
}

def get_parcel_id(filename):
    m = re.match(r'test_([A-Za-z]+\d+)', filename)
    return m.group(1) if m else filename

def extract_all(pdf_path):
    data = {}
    all_text = []
    annexure_rows = []

    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        all_text.append(text)

        # Parse annexure table rows: Index | Date | NDVI | EVI | RVI | Crops | Prediction
        for line in text.splitlines():
            m = re.match(r'\s*(\d+)\s+([\d-]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(Crop-Activity|No Crop-Activity)', line)
            if m:
                annexure_rows.append({
                    'idx': int(m.group(1)),
                    'date': m.group(2),
                    'ndvi': float(m.group(3)),
                    'evi': float(m.group(4)),
                    'rvi': float(m.group(5)),
                    'crops_prob': float(m.group(6)),
                    'prediction': m.group(7)
                })
    doc.close()

    full_text = "\n".join(all_text)

    # --- Page 1: Verdict ---
    m = re.search(r'Verdict:\s*(.*)', full_text)
    if m: data['verdict_line'] = m.group(1).strip()
    m = re.search(r'(\d+\.?\d*)%\s*Active', full_text)
    if m: data['active_pct'] = float(m.group(1))
    m = re.search(r'Final Confidence:\s*(\d+\.?\d*)%', full_text)
    if m: data['final_confidence'] = float(m.group(1))
    m = re.search(r'State\s*\n\s*(.+)', full_text)
    if m: data['state'] = m.group(1).strip()

    # --- Page 2: DW class breakdown ---
    dw_order = ['Water','Trees','Grass','Flooded Vegetation','Crops',
                'Shrub And Scrub','Built','Bare','Snow And Ice']
    for cls in dw_order:
        pattern = rf'{re.escape(cls)}\s*\n\s*(\d+\.?\d*)%'
        m = re.search(pattern, full_text)
        if m:
            key = cls.lower().replace(' ','_').replace('and_','')
            data[f'dw_{key}'] = float(m.group(1))

    # Active cropping instances
    m = re.search(r'INSTANCES OF ACTIVE CROPPING\s*\n\s*(\d+)', full_text)
    if m: data['active_instances'] = int(m.group(1))
    m = re.search(r'INSTANCES OF FALLOW.*?\n.*?\n\s*(\d+)', full_text)
    if m: data['fallow_instances'] = int(m.group(1))

    # Annual Crop Activity percentage
    m = re.search(r'(KHARIF|RABI|ZAID)\s*\n\s*(KHARIF|RABI|ZAID)?\s*\n?\s*(KHARIF|RABI|ZAID)?\s*\n?\s*(\d+\.?\d*)%', full_text)
    if m: data['annual_activity_pct'] = float(m.group(4))

    # Seasons detected
    seasons = []
    page2_text = all_text[1] if len(all_text) > 1 else ""
    if 'KHARIF' in page2_text: seasons.append('Kharif')
    if 'RABI' in page2_text: seasons.append('Rabi')
    if 'ZAID' in page2_text: seasons.append('Zaid')
    data['seasons'] = seasons

    # Monthly activity from page 7
    page7_text = all_text[6] if len(all_text) > 6 else ""
    active_months = page7_text.count('ACTIVE')
    inactive_months = page7_text.count('INACTIVE') + page7_text.count('FALLOW')
    data['active_months'] = active_months
    data['inactive_months'] = inactive_months

    # --- Page 5: Peak values ---
    m = re.search(r'NDVI\s*\n\s*([\d.]+)\s*\n\s*(\w+\s*\d{4})', full_text)
    if m:
        data['ndvi_peak'] = float(m.group(1))
        data['ndvi_peak_month'] = m.group(2).strip()
    m = re.search(r'RVI\s*\n\s*([\d.]+)\s*\n\s*(\w+\s*\d{4})', full_text)
    if m:
        data['rvi_peak'] = float(m.group(1))
        data['rvi_peak_month'] = m.group(2).strip()
    m = re.search(r'EVI\s*\n\s*([\d.]+)\s*\n\s*(\w+\s*\d{4})', full_text)
    if m:
        data['evi_peak'] = float(m.group(1))
        data['evi_peak_month'] = m.group(2).strip()

    # --- Annexure stats ---
    if annexure_rows:
        ndvi_vals = [r['ndvi'] for r in annexure_rows]
        rvi_vals = [r['rvi'] for r in annexure_rows]
        crop_probs = [r['crops_prob'] for r in annexure_rows]
        crop_count = sum(1 for r in annexure_rows if r['prediction'] == 'Crop-Activity')
        total = len(annexure_rows)

        data['total_rows'] = total
        data['crop_rows'] = crop_count
        data['nocrop_rows'] = total - crop_count
        data['activity_ratio'] = round(crop_count / total * 100, 2) if total > 0 else 0

        data['ndvi_min'] = round(min(ndvi_vals), 4)
        data['ndvi_max'] = round(max(ndvi_vals), 4)
        data['ndvi_mean'] = round(sum(ndvi_vals)/len(ndvi_vals), 4)
        data['ndvi_range'] = round(max(ndvi_vals) - min(ndvi_vals), 4)

        data['rvi_min'] = round(min(rvi_vals), 4)
        data['rvi_max'] = round(max(rvi_vals), 4)
        data['rvi_mean'] = round(sum(rvi_vals)/len(rvi_vals), 4)

        data['crops_prob_min'] = round(min(crop_probs), 4)
        data['crops_prob_max'] = round(max(crop_probs), 4)
        data['crops_prob_mean'] = round(sum(crop_probs)/len(crop_probs), 4)
    else:
        data['total_rows'] = 0

    return data

def main():
    files = sorted(f for f in os.listdir(REPORT_DIR) if f.endswith('.pdf'))
    print(f"Deep-parsing {len(files)} PDF reports...", flush=True)

    all_results = []
    for i, fname in enumerate(files):
        pid = get_parcel_id(fname)
        truth_nocrop = '(No-Crop)' in fname
        truth = 'Inactive' if truth_nocrop else 'Active'
        category = 'FP' if pid in FALSE_POSITIVES else ('FN' if pid in FALSE_NEGATIVES else 'CORRECT')

        try:
            data = extract_all(os.path.join(REPORT_DIR, fname))
        except Exception as e:
            data = {'error': str(e)}

        data['parcel_id'] = pid
        data['truth'] = truth
        data['category'] = category
        all_results.append(data)

        if (i+1) % 50 == 0:
            print(f"  ...parsed {i+1}/{len(files)}", flush=True)

    # Organize
    fp = [r for r in all_results if r['category'] == 'FP']
    fn = [r for r in all_results if r['category'] == 'FN']
    cc = [r for r in all_results if r['category'] == 'CORRECT' and r['truth'] == 'Active']
    cn = [r for r in all_results if r['category'] == 'CORRECT' and r['truth'] == 'Inactive']

    out = []
    out.append("=" * 90)
    out.append("DEEP PDF ANALYSIS — 306 PARCELS (ALL FIELDS)")
    out.append("=" * 90)

    # --- FALSE POSITIVES ---
    out.append(f"\n{'='*90}")
    out.append(f"FALSE POSITIVES ({len(fp)}) — Truth=Inactive, AI=Active")
    out.append(f"{'='*90}")
    for r in fp:
        out.append(f"\n  [{r['parcel_id']}] ({r.get('state','?')})")
        out.append(f"    Verdict        : {r.get('verdict_line','?')}")
        out.append(f"    Active%/Conf   : {r.get('active_pct','?')}% / {r.get('final_confidence','?')}%")
        out.append(f"    Seasons        : {r.get('seasons',[])}")
        out.append(f"    Active Months  : {r.get('active_months','?')} | Inactive Months: {r.get('inactive_months','?')}")
        out.append(f"    Active Insts   : {r.get('active_instances','?')} | Fallow Insts: {r.get('fallow_instances','?')}")
        out.append(f"    Total Rows     : {r.get('total_rows','?')} | Crop: {r.get('crop_rows','?')} | NoCrop: {r.get('nocrop_rows','?')}")
        out.append(f"    Activity Ratio : {r.get('activity_ratio','?')}%")
        out.append(f"    NDVI           : min={r.get('ndvi_min','?')} max={r.get('ndvi_max','?')} mean={r.get('ndvi_mean','?')} range={r.get('ndvi_range','?')}")
        out.append(f"    NDVI Peak      : {r.get('ndvi_peak','?')} ({r.get('ndvi_peak_month','?')})")
        out.append(f"    RVI            : min={r.get('rvi_min','?')} max={r.get('rvi_max','?')} mean={r.get('rvi_mean','?')}")
        out.append(f"    RVI Peak       : {r.get('rvi_peak','?')} ({r.get('rvi_peak_month','?')})")
        out.append(f"    EVI Peak       : {r.get('evi_peak','?')} ({r.get('evi_peak_month','?')})")
        out.append(f"    Crops Prob     : min={r.get('crops_prob_min','?')} max={r.get('crops_prob_max','?')} mean={r.get('crops_prob_mean','?')}")
        dw = {k.replace('dw_',''): v for k,v in r.items() if k.startswith('dw_')}
        out.append(f"    DW Classes     : {dw}")

    # --- FALSE NEGATIVES ---
    out.append(f"\n{'='*90}")
    out.append(f"FALSE NEGATIVES ({len(fn)}) — Truth=Active, AI=Inactive")
    out.append(f"{'='*90}")
    for r in fn:
        out.append(f"\n  [{r['parcel_id']}] ({r.get('state','?')})")
        out.append(f"    Verdict        : {r.get('verdict_line','?')}")
        out.append(f"    Active%/Conf   : {r.get('active_pct','?')}% / {r.get('final_confidence','?')}%")
        out.append(f"    Seasons        : {r.get('seasons',[])}")
        out.append(f"    Active Months  : {r.get('active_months','?')} | Inactive Months: {r.get('inactive_months','?')}")
        out.append(f"    Total Rows     : {r.get('total_rows','?')} | Crop: {r.get('crop_rows','?')} | NoCrop: {r.get('nocrop_rows','?')}")
        out.append(f"    Activity Ratio : {r.get('activity_ratio','?')}%")
        out.append(f"    NDVI           : min={r.get('ndvi_min','?')} max={r.get('ndvi_max','?')} mean={r.get('ndvi_mean','?')} range={r.get('ndvi_range','?')}")
        out.append(f"    NDVI Peak      : {r.get('ndvi_peak','?')} ({r.get('ndvi_peak_month','?')})")
        out.append(f"    RVI            : min={r.get('rvi_min','?')} max={r.get('rvi_max','?')} mean={r.get('rvi_mean','?')}")
        out.append(f"    Crops Prob     : min={r.get('crops_prob_min','?')} max={r.get('crops_prob_max','?')} mean={r.get('crops_prob_mean','?')}")
        dw = {k.replace('dw_',''): v for k,v in r.items() if k.startswith('dw_')}
        out.append(f"    DW Classes     : {dw}")

    # --- CORRECT CROP STATS ---
    out.append(f"\n{'='*90}")
    out.append(f"CORRECT CROP ({len(cc)}) — Aggregate Stats")
    out.append(f"{'='*90}")
    def stats(vals):
        return f"min={min(vals):.2f} mean={sum(vals)/len(vals):.2f} max={max(vals):.2f}" if vals else "N/A"
    out.append(f"  Active%     : {stats([r['active_pct'] for r in cc if 'active_pct' in r])}")
    out.append(f"  FinalConf   : {stats([r['final_confidence'] for r in cc if 'final_confidence' in r])}")
    out.append(f"  ActivityRat : {stats([r['activity_ratio'] for r in cc if 'activity_ratio' in r])}")
    out.append(f"  NDVI max    : {stats([r['ndvi_max'] for r in cc if 'ndvi_max' in r])}")
    out.append(f"  NDVI range  : {stats([r['ndvi_range'] for r in cc if 'ndvi_range' in r])}")
    out.append(f"  NDVI mean   : {stats([r['ndvi_mean'] for r in cc if 'ndvi_mean' in r])}")
    out.append(f"  RVI mean    : {stats([r['rvi_mean'] for r in cc if 'rvi_mean' in r])}")
    out.append(f"  CropsProb   : {stats([r['crops_prob_mean'] for r in cc if 'crops_prob_mean' in r])}")

    # Per-parcel crop list
    out.append(f"\n  --- Individual Correct Crop Parcels ---")
    for r in cc:
        dw_crops = r.get('dw_crops', '?')
        dw_trees = r.get('dw_trees', '?')
        out.append(f"  [{r['parcel_id']}] Active={r.get('active_pct','?')}% Ratio={r.get('activity_ratio','?')}% NDVI=[{r.get('ndvi_min','?')}-{r.get('ndvi_max','?')}] RVI_mean={r.get('rvi_mean','?')} CropProb={r.get('crops_prob_mean','?')} DW:crops={dw_crops}% trees={dw_trees}%")

    # --- CORRECT NO-CROP STATS ---
    out.append(f"\n{'='*90}")
    out.append(f"CORRECT NO-CROP ({len(cn)}) — Aggregate Stats")
    out.append(f"{'='*90}")
    out.append(f"  Active%     : {stats([r['active_pct'] for r in cn if 'active_pct' in r])}")
    out.append(f"  FinalConf   : {stats([r['final_confidence'] for r in cn if 'final_confidence' in r])}")
    out.append(f"  ActivityRat : {stats([r['activity_ratio'] for r in cn if 'activity_ratio' in r])}")
    out.append(f"  NDVI max    : {stats([r['ndvi_max'] for r in cn if 'ndvi_max' in r])}")
    out.append(f"  NDVI range  : {stats([r['ndvi_range'] for r in cn if 'ndvi_range' in r])}")
    out.append(f"  CropsProb   : {stats([r['crops_prob_mean'] for r in cn if 'crops_prob_mean' in r])}")

    out.append(f"\n  --- Individual Correct No-Crop Parcels ---")
    for r in cn:
        dw_crops = r.get('dw_crops', '?')
        dw_trees = r.get('dw_trees', '?')
        dw_water = r.get('dw_water', '?')
        dw_built = r.get('dw_built', '?')
        out.append(f"  [{r['parcel_id']}] Active={r.get('active_pct','?')}% Ratio={r.get('activity_ratio','?')}% NDVI=[{r.get('ndvi_min','?')}-{r.get('ndvi_max','?')}] range={r.get('ndvi_range','?')} CropProb={r.get('crops_prob_mean','?')} DW:crops={dw_crops}% trees={dw_trees}% water={dw_water}% built={dw_built}%")

    # Write
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(out))

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_analysis_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Report: {report_path}")
    print(f"JSON: {json_path}")

if __name__ == '__main__':
    main()

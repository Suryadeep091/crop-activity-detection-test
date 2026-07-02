import json
from pathlib import Path
import zipfile
from bs4 import BeautifulSoup


def convert_kmz_in_downloads(kmz_filename, json_filename):
    # Automatically locate the user's Downloads folder
    downloads_path = Path.home() / "Downloads"

    kmz_path = downloads_path / kmz_filename
    json_path = downloads_path / json_filename

    # Check if the input file actually exists in Downloads
    if not kmz_path.exists():
        print(f"Error: Could not find '{kmz_filename}' in {downloads_path}")
        return

    print(f"Reading: {kmz_path}")

    # 1. Extract the KML file from the KMZ archive
    with zipfile.ZipFile(kmz_path, "r") as kmz:
        kml_filenames = [f for f in kmz.namelist() if f.endswith(".kml")]
        if not kml_filenames:
            raise ValueError("No KML file found inside the KMZ archive.")
        kml_content = kmz.read(kml_filenames[0])

    # 2. Parse the KML
    soup = BeautifulSoup(kml_content, "xml")
    parcels_json = []
    placemarks = soup.find_all("Placemark")

    # 3. Process each parcel
    for index, placemark in enumerate(placemarks, start=1):
        name_tag = placemark.find("name")
        task_id = name_tag.text.strip() if name_tag else f"UP{index} (Crop)"

        coord_tag = placemark.find("coordinates")
        if not coord_tag:
            continue

        # Format coordinates into a clean space-separated string
        clean_coords = " ".join(coord_tag.text.strip().split())

        parcels_json.append(
            {"task_id": task_id, "kml_coordinates": clean_coords}
        )

    # 4. Save the formatted JSON back to the Downloads folder
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parcels_json, f, indent=4)

    print(f"Success! Saved formatted JSON to: {json_path}")


# --- Run the conversion ---
# Change 'parcels.kmz' to match the exact name of your file inside Downloads
convert_kmz_in_downloads("kk.kmz", "test_parcels_output.json")
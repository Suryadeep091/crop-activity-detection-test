import io
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
from rdp import rdp

def get_static_map_b64(coords, khasra_no, api_key, zoom=18, size="550x440"):
    """Fetches Google Static Map and returns it as a Base64 string."""
    if not coords or len(coords) < 3:
        return None

    # 1. Simplify Coordinates for URL limits
    simplified_coords = rdp(coords, epsilon=0.00001)
    if simplified_coords[0] != simplified_coords[-1]:
        simplified_coords.append(simplified_coords[0])

    path = "fillcolor:0x0000ff40|color:0x0000ff|weight:3|" + "|".join(f"{lat},{lon}" for lon, lat in simplified_coords)
    
    center_lat = sum(lat for lon, lat in simplified_coords) / len(simplified_coords)
    center_lon = sum(lon for lon, lat in simplified_coords) / len(simplified_coords)

    params = {
        "size": size,
        "maptype": "satellite",
        "path": path,
        "center": f"{center_lat},{center_lon}",
        "key": api_key
    }

    response = requests.get("https://maps.googleapis.com/maps/api/staticmap", params=params)
    if response.status_code != 200:
        return None

    # 2. Add Text Overlay in Memory
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    draw = ImageDraw.Draw(img)
    text = f"Khasra: {khasra_no}"
    
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    img_w, img_h = img.size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_pos = ((img_w - (bbox[2] - bbox[0])) // 2, (img_h - (bbox[3] - bbox[1])) // 2)
    draw.text(text_pos, text, fill="yellow", font=font, stroke_width=2, stroke_fill="black")

    # 3. Convert to Base64 String
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')
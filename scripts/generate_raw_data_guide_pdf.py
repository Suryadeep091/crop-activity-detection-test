"""
Generate technical PDF docs for AdvaRisk.

Usage (from project root):
    python scripts/generate_raw_data_guide_pdf.py              # both PDFs
    python scripts/generate_raw_data_guide_pdf.py --fetch-only # fetch guide only
    python scripts/generate_raw_data_guide_pdf.py --thesis-only # EO thesis addendum only

Outputs:
    docs/Raw_Satellite_Data_Fetching_Guide.pdf
    docs/AdvaRisk_EO_Downstream_Thesis_Addendum.pdf
"""
import argparse
import asyncio
import os
from datetime import datetime

from jinja2 import Environment, FileSystemLoader
from playwright.async_api import async_playwright


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "docs")

DOCS = [
    {
        "template": "raw_satellite_data_guide.html",
        "output": "Raw_Satellite_Data_Fetching_Guide.pdf",
    },
    {
        "template": "advarisk_eo_downstream_addendum.html",
        "output": "AdvaRisk_EO_Downstream_Thesis_Addendum.pdf",
    },
]


async def generate_pdf(template_name: str, output_name: str) -> str:
    template_dir = os.path.join(PROJECT_DIR, "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    html_content = template.render(
        generated_at=datetime.now().strftime("%d %B %Y, %I:%M %p"),
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html_content, wait_until="networkidle")
        await page.pdf(
            path=output_path,
            format="A4",
            print_background=True,
            margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
        )
        await browser.close()

    print(f"PDF written to: {output_path}")
    return output_path


async def main(which: str):
    if which == "fetch":
        docs = [DOCS[0]]
    elif which == "thesis":
        docs = [DOCS[1]]
    else:
        docs = DOCS

    for doc in docs:
        await generate_pdf(doc["template"], doc["output"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AdvaRisk technical PDFs")
    parser.add_argument("--fetch-only", action="store_true", help="Raw satellite fetch guide only")
    parser.add_argument("--thesis-only", action="store_true", help="EO downstream thesis addendum only")
    args = parser.parse_args()

    if args.fetch_only and args.thesis_only:
        parser.error("Use only one of --fetch-only or --thesis-only")

    mode = "fetch" if args.fetch_only else ("thesis" if args.thesis_only else "all")
    asyncio.run(main(mode))

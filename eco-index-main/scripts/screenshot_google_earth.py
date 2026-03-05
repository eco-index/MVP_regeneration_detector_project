#!/usr/bin/env python3
"""
Capture Google Earth screenshots for features defined in a KML file using Playwright.
This version has no dependency on geopandas and includes resume/start_at functionality.
"""

import argparse
import asyncio
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

from playwright.async_api import Page, async_playwright
from tqdm.asyncio import tqdm

# Altitude in meters for point screenshots. Lower is more zoomed in.
POINT_ALTITUDE_METERS = 800


def parse_kml_features(kml_path: str) -> List[Dict[str, Any]]:
    """
    Parses a KML file to extract feature information using standard libraries.
    """
    features = []
    try:
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        tree = ET.parse(kml_path)
        root = tree.getroot()

        for placemark in root.findall('.//kml:Placemark', namespaces):
            feature = {}
            name_tag = placemark.find('kml:name', namespaces)
            feature['name'] = name_tag.text.strip() if name_tag is not None and name_tag.text else 'unnamed'

            point_tag = placemark.find('.//kml:Point/kml:coordinates', namespaces)
            if point_tag is not None and point_tag.text:
                feature['type'] = 'Point'
                try:
                    lon, lat, *_ = point_tag.text.strip().split(',')
                    feature['coords'] = (float(lon), float(lat))
                except (ValueError, IndexError):
                    print(f"Warning: Skipping malformed Point coordinate in placemark '{feature['name']}'", file=sys.stderr)
                    continue
            else:
                poly_tag = placemark.find('.//kml:Polygon', namespaces)
                if poly_tag is not None:
                    feature['type'] = 'Polygon'
                    feature['coords'] = None
                else:
                    continue
            
            features.append(feature)

    except ET.ParseError as e:
        print(f"Error parsing XML in {kml_path}: {e}", file=sys.stderr)
        return []
    except FileNotFoundError:
        raise
        
    return features


async def click_at_coordinates(page: Page, x: int, y: int, sleep: float = 1.0) -> None:
    """Click at the given coordinates on the page."""
    await page.mouse.click(x, y)
    await page.wait_for_timeout(int(sleep * 1000))


async def setup_google_earth_ui(page: Page) -> None:
    """Navigates the Google Earth UI to get a clean, 2D satellite view."""
    print("Setting up Google Earth UI for clean screenshots...")
    await page.goto("https://earth.google.com/web/")
    await page.wait_for_timeout(10000) # Increased wait for stability
    await click_at_coordinates(page, 76, 9)
    await click_at_coordinates(page, 390, 2100)
    await click_at_coordinates(page, 600, 890)
    await click_at_coordinates(page, 600, 1094)
    await click_at_coordinates(page, 580, 1540)
    await click_at_coordinates(page, 608, 798)
    await click_at_coordinates(page, 530, 90)
    await click_at_coordinates(page, 3811, 30)
    await click_at_coordinates(page, 3508, 186)
    await click_at_coordinates(page, 2167, 184)
    await click_at_coordinates(page, 1994, 643)
    await click_at_coordinates(page, 2193, 2024)
    print("UI setup complete.")


async def screenshot_points(page: Page, features: List[Dict[str, Any]], output_dir: str, wait: int) -> None:
    """Navigate to each point via URL and take a screenshot."""
    print(f"Processing {len(features)} points...")
    await click_at_coordinates(page, 76, 9) # Close the file menu from setup

    for feature in tqdm(features, desc="Screenshotting Points"):
        name = feature["name"]
        lon, lat = feature["coords"]
        
        outfile = Path(output_dir) / f"{name}.png"
        
        # --- RESUME LOGIC ---
        # If the output file already exists, skip it.
        if outfile.exists():
            continue

        url = f"https://earth.google.com/web/@{lat},{lon},{POINT_ALTITUDE_METERS}a,0d,60y,0h,0t,0r"
        
        try:
            await page.goto(url)
            await page.wait_for_load_state('networkidle', timeout=wait * 1000)
            await page.screenshot(path=str(outfile))
        except Exception as e:
            # --- SKIP ON FAILURE LOGIC ---
            # If anything goes wrong (timeout, etc.), log it and move on.
            tqdm.write(f"Warning: Failed to capture point '{name}'. Error: {e}. Skipping.")
            continue


async def take_screenshots(kml_path: str, output_dir: str, wait: int, start_at: str, headless: bool) -> None:
    """
    Main function to launch browser and delegate screenshotting.
    """
    features = parse_kml_features(kml_path)
    if not features:
        print("KML file is empty or contains no valid features. Exiting.")
        return

    # --- START_AT LOGIC ---
    if start_at:
        try:
            start_index = [f['name'] for f in features].index(start_at)
            print(f"Starting at feature '{start_at}' (index {start_index}).")
            features = features[start_index:]
        except ValueError:
            print(f"Error: Start point '{start_at}' not found in KML file names.", file=sys.stderr)
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    geom_type = features[0]['type']

    async with async_playwright() as p:
        print(f"Launching browser (headless={headless})...")
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 3840, "height": 2160})
        page = await context.new_page()

        if geom_type == 'Point':
            await setup_google_earth_ui(page)
            await screenshot_points(page, features, output_dir, wait)
        else:
            print(f"Unsupported geometry type for this script: {geom_type}. Only 'Point' is supported for robust resume.")
        
        await browser.close()
        print("\nScreenshot process complete.")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture Google Earth screenshots for KML points with resume capability."
    )
    parser.add_argument("--kml", required=True, help="Path to the KML file")
    parser.add_argument("--output", required=True, help="Directory to store screenshots")
    parser.add_argument(
        "--wait", type=int, default=20,
        help="Seconds to wait for map to load. Increase for slow connections (default: 20)."
    )
    parser.add_argument(
        "--start_at", type=str, default=None,
        help="Name of the placemark to start from (e.g., 'tile_row_35_col_9')."
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run the browser without a visible UI. Recommended for performance."
    )
    args = parser.parse_args()
    await take_screenshots(args.kml, args.output, args.wait, args.start_at, args.headless)


if __name__ == "__main__":
    asyncio.run(main())

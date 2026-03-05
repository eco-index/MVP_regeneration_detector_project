#!/usr/bin/env python3
"""
Generates a KML file with a grid of points covering a specified region.

This script takes a KML file containing one or more polygons that define a
region, calculates its total bounding box, and generates a grid of coordinates
within that box. It then filters these points to keep only those that fall
within the actual region's geometry. The final output is a KML file where each
valid grid point is a <Placemark>, ready to be used by screenshotting or
other processing tools.

This version has no dependency on geopandas/shapely and uses standard libraries
for KML parsing and a pure Python implementation for the point-in-polygon test.

Dependencies:
- numpy: For efficiently generating coordinate ranges.
- tqdm: To display a progress bar for large regions.

Example Usage:
python scripts/generate_grid_kml.py \
    --region_kml path/to/my_region.kml \
    --output_kml data/kml/generated/my_region_grid.kml \
    --step_size_deg 0.005
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_kml_polygons(kml_path: str) -> list[list[tuple[float, float]]]:
    """
    Parses a KML file to extract all polygon coordinates using standard libraries.

    Args:
        kml_path (str): The path to the KML file.

    Returns:
        A list of polygons, where each polygon is a list of (lon, lat) tuples.
    """
    polygons = []
    try:
        # KML files use a namespace, which we need to handle
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        tree = ET.parse(kml_path)
        root = tree.getroot()

        # Find all <coordinates> tags within the document
        for coords_element in root.findall('.//kml:coordinates', namespaces):
            if coords_element.text:
                # Clean up whitespace and split into individual points
                coord_text = coords_element.text.strip()
                points_str = coord_text.split()
                
                polygon = []
                for point_str in points_str:
                    try:
                        # Coordinates are typically lon,lat,alt
                        lon, lat, *_ = point_str.split(',')
                        polygon.append((float(lon), float(lat)))
                    except (ValueError, IndexError):
                        # Skip malformed coordinate strings
                        continue
                
                if polygon:
                    polygons.append(polygon)
    except ET.ParseError as e:
        print(f"Error parsing XML in {kml_path}: {e}", file=sys.stderr)
        return []
    except FileNotFoundError:
        raise
        
    return polygons


def is_point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """
    Determines if a point is inside a given polygon using the Ray Casting algorithm.

    Args:
        point (tuple): A (longitude, latitude) tuple.
        polygon (list): A list of (lon, lat) tuples representing the polygon's vertices.

    Returns:
        True if the point is inside the polygon, False otherwise.
    """
    lon, lat = point
    num_vertices = len(polygon)
    is_inside = False
    
    p1_lon, p1_lat = polygon[0]
    for i in range(1, num_vertices + 1):
        p2_lon, p2_lat = polygon[i % num_vertices]
        if min(p1_lat, p2_lat) < lat <= max(p1_lat, p2_lat):
            if lon <= max(p1_lon, p2_lon):
                if p1_lat != p2_lat:
                    # Calculate the x-intersection of the line
                    x_intersection = (lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                    if p1_lon == p2_lon or lon <= x_intersection:
                        is_inside = not is_inside
        p1_lon, p1_lat = p2_lon, p2_lat
        
    return is_inside


def is_point_in_region(point: tuple[float, float], region_polygons: list[list[tuple[float, float]]]) -> bool:
    """Checks if a point is inside any of the polygons that define a region."""
    for polygon in region_polygons:
        if is_point_in_polygon(point, polygon):
            return True
    return False


def create_grid_kml(
    input_kml_path: str, output_kml_path: str, step_size: float
) -> None:
    """
    Reads a region KML, generates an internal grid, and writes it to a new KML.

    Args:
        input_kml_path (str): Path to the source KML file defining the region.
        output_kml_path (str): Path to save the generated grid KML file.
        step_size (float): The spacing between grid points in decimal degrees.
    """
    print(f"Reading region from {input_kml_path}...")
    region_polygons = parse_kml_polygons(input_kml_path)

    if not region_polygons:
        print(f"Error: No valid geometric features found in {input_kml_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Successfully parsed {len(region_polygons)} polygon(s) from the KML.")

    # Calculate the bounding box of the entire region
    all_points = [point for polygon in region_polygons for point in polygon]
    min_lon = min(p[0] for p in all_points)
    max_lon = max(p[0] for p in all_points)
    min_lat = min(p[1] for p in all_points)
    max_lat = max(p[1] for p in all_points)
    
    print(
        f"Region bounding box: "
        f"Lon ({min_lon:.4f} to {max_lon:.4f}), "
        f"Lat ({min_lat:.4f} to {max_lat:.4f})"
    )

    # Generate grid points covering the bounding box
    lons = np.arange(min_lon, max_lon, step_size)
    lats = np.arange(min_lat, max_lat, step_size)
    total_points = len(lons) * len(lats)
    print(f"Generating a grid of {total_points} points with a step of {step_size} degrees...")

    # Filter points to keep only those within the actual polygon
    valid_points = []
    grid_point_iterator = product(lons, lats)

    for lon, lat in tqdm(grid_point_iterator, total=total_points, desc="Filtering points"):
        point = (lon, lat)
        if is_point_in_region(point, region_polygons):
            valid_points.append(point)

    if not valid_points:
        print("Warning: No grid points fell within the specified region. The output file will be empty.", file=sys.stderr)
        print("Consider using a smaller --step_size_deg.", file=sys.stderr)
    else:
        print(f"Found {len(valid_points)} valid points inside the region.")

    # --- KML Generation ---
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Generated Grid</name>
    <description>Grid points generated by generate_grid_kml.py</description>
"""

    placemark_template = """    <Placemark>
      <name>tile_row_{row}_col_{col}</name>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>
"""

    kml_footer = """  </Document>
</kml>
"""

    # Create a mapping from grid coordinates to row/col indices for naming
    lon_map = {lon: i for i, lon in enumerate(lons)}
    lat_map = {lat: i for i, lat in enumerate(lats)}

    print("Generating KML content...")
    placemarks = []
    for point in tqdm(valid_points, desc="Creating KML placemarks"):
        lon, lat = point
        # Find the original grid indices for a structured name
        col_idx = lon_map.get(lon, -1)
        row_idx = lat_map.get(lat, -1)

        placemarks.append(
            placemark_template.format(row=row_idx, col=col_idx, lon=lon, lat=lat)
        )

    # Ensure output directory exists
    output_path = Path(output_kml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the final KML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(kml_header)
        f.write("\n".join(placemarks))
        f.write(kml_footer)

    print(f"\nSuccessfully generated KML file with {len(valid_points)} placemarks.")
    print(f"Output saved to: {output_path.resolve()}")


def main():
    """Parses command-line arguments and runs the grid generation."""
    parser = argparse.ArgumentParser(
        description="Generate a KML file with a grid of points covering a region defined in another KML."
    )
    parser.add_argument(
        "--region_kml",
        required=True,
        help="Path to the input KML file defining the region boundary.",
    )
    parser.add_argument(
        "--output_kml",
        required=True,
        help="Path to save the generated KML file with grid points.",
    )
    parser.add_argument(
        "--step_size_deg",
        type=float,
        required=True,
        help="The spacing between grid points in decimal degrees (e.g., 0.01 for ~1.1km spacing).",
    )
    args = parser.parse_args()

    try:
        create_grid_kml(args.region_kml, args.output_kml, args.step_size_deg)
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

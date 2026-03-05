# Google Earth Screenshot Capture

This guide explains how to use `scripts/screenshot_google_earth.py` to capture imagery from Google Earth for each polygon in a KML file. The script uses Playwright's asynchronous API, paving the way for parallel processing in the future.

## Requirements

- Python 3
- `geopandas` and `playwright` Python packages
- Run `playwright install` to download the required browsers

## Usage

Run the script with the path to a KML file and an output directory:

```bash
python scripts/screenshot_google_earth.py \
    --kml path/to/polygons.kml \
    --output imagery/
```

The script opens Google Earth in a headless browser and searches for the centroid of each polygon. It waits for the map to settle, then navigates to each location and saves a PNG screenshot to the specified output directory. You can control how long the map is allowed to settle before capturing an image using the `--wait` option.

Interaction with the Google Earth UI is handled via Playwright clicks. The KML
file is uploaded by intercepting the browser's file chooser event, avoiding the
need to query for the underlying input element.

The captured imagery is subject to Google Earth's terms of service. Ensure that automated access and data usage complies with those terms.

## Batch Processing

Run `scripts/screenshot_all_kml.py` to capture imagery for every KML file under
`data/kml/cleaned`. Screenshots are written to matching folders inside
`data/output/`.

```bash
python scripts/screenshot_all_kml.py
```

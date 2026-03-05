#!/usr/bin/env python3
"""Capture screenshots for all cleaned KML files."""

import subprocess
import sys
from pathlib import Path

CLEANED_DIR = Path("data/kml/cleaned")
OUTPUT_DIR = Path("data/output")


def run_screenshot(kml_path: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("screenshot_google_earth.py")),
        "--kml",
        str(kml_path),
        "--output",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    for kml_file in CLEANED_DIR.rglob("*.kml"):
        rel = kml_file.relative_to(CLEANED_DIR)
        out_dir = OUTPUT_DIR / rel
        print(f"Capturing {kml_file} -> {out_dir}")
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        run_screenshot(kml_file, out_dir)


if __name__ == "__main__":
    main()

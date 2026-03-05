#!/usr/bin/env python3
"""Clean all raw KML files unless a cleaned version already exists."""

import subprocess
import sys
from pathlib import Path

RAW_DIR = Path("data/kml/raw")
CLEANED_DIR = Path("data/kml/cleaned")


def run_clean_kml(raw_path: Path, cleaned_path: Path) -> None:
    cmd = [sys.executable, str(Path(__file__).with_name("clean_kml.py")),
           "--input", str(raw_path), "--output", str(cleaned_path)]
    subprocess.run(cmd, check=True)


def main() -> None:
    for raw_file in RAW_DIR.rglob("*.kml"):
        rel = raw_file.relative_to(RAW_DIR)
        cleaned_file = CLEANED_DIR / rel
        if cleaned_file.exists():
            print(f"Skipping {raw_file}: cleaned version already exists")
            continue
        print(f"Cleaning {raw_file} -> {cleaned_file}")
        cleaned_file.parent.mkdir(parents=True, exist_ok=True)
        run_clean_kml(raw_file, cleaned_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Utility script to clear the Magic Georeferencer tile cache.

This is useful if you have cached "access blocked" tiles from OSM
before the User-Agent fix was applied.

Run this from the QGIS Python console:
    import os
    script_path = os.path.join(os.path.expanduser('~'), 'georefio', 'clear_osm_cache.py')
    exec(open(script_path).read())

Or if you cloned the repo elsewhere, adjust the path accordingly.
"""

from pathlib import Path
from qgis.core import QgsApplication

# Get cache directory
cache_base = Path(QgsApplication.qgisSettingsDirPath())
cache_dir = cache_base / 'magic_georeferencer' / 'tiles'

if not cache_dir.exists():
    print(f"Cache directory does not exist: {cache_dir}")
    print("No cached tiles to clear.")
else:
    # Count files
    png_files = list(cache_dir.glob("*.png"))
    meta_files = list(cache_dir.glob("*.meta"))

    print(f"Found {len(png_files)} cached tile images")
    print(f"Found {len(meta_files)} cached metadata files")

    if png_files or meta_files:
        response = input(f"\nDelete {len(png_files) + len(meta_files)} cached files? (y/n): ")

        if response.lower() == 'y':
            # Delete all cache files
            for f in png_files:
                f.unlink()
            for f in meta_files:
                f.unlink()

            print(f"\n✓ Cleared {len(png_files) + len(meta_files)} cached files")
            print("Next tile fetch will use fresh tiles with proper User-Agent.")
        else:
            print("\nCache clearing cancelled.")
    else:
        print("\nNo cached files found.")

print(f"\nCache directory: {cache_dir}")

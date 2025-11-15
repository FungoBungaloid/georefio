"""
Comprehensive debugging script for tile fetching, zoom level, and coordinate systems.

This script will:
1. Show the current map extent and CRS
2. Calculate the appropriate zoom level based on map scale
3. Fetch tiles at multiple zoom levels for comparison
4. Save debug images showing what the model is seeing
5. Verify all coordinate transformations
6. Check if the model is actually working
"""

import sys
import math
from pathlib import Path
from qgis.core import (
    QgsApplication, QgsProject, QgsRectangle, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsMapSettings, QgsMapRendererCustomPainterJob
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QImage, QPainter
from qgis.utils import iface
import numpy as np
from PIL import Image

# Initialize QGIS
QgsApplication.setPrefixPath('/usr', True)
qgs = QgsApplication([], False)
qgs.initQgis()


def calculate_zoom_level_from_scale(scale, latitude):
    """
    Calculate appropriate tile zoom level from map scale.

    Args:
        scale: Map scale (e.g., 10000 for 1:10000)
        latitude: Latitude of the map center (affects tile size)

    Returns:
        Appropriate zoom level (0-19)
    """
    # At zoom level 0, one tile (256px) covers the entire world
    # Each zoom level doubles the resolution
    # At equator: zoom level z has resolution of (earth_circumference / (256 * 2^z)) meters per pixel

    earth_circumference = 40075016.686  # meters at equator

    # Adjust for latitude (tiles get smaller in geographic terms as you move from equator)
    lat_factor = math.cos(math.radians(latitude))

    # meters per pixel at this scale on screen
    # Assuming 96 DPI and 1:scale map
    screen_dpi = 96
    meters_per_inch = 0.0254
    screen_meters_per_pixel = (scale * meters_per_inch) / screen_dpi

    # Find zoom level where tile resolution matches screen resolution
    for zoom in range(20):
        tile_meters_per_pixel = (earth_circumference * lat_factor) / (256 * (2 ** zoom))

        # We want tile resolution to be close to or slightly better than screen resolution
        if tile_meters_per_pixel <= screen_meters_per_pixel * 1.5:
            return min(zoom, 19)  # Cap at max zoom

    return 19


def calculate_zoom_level_from_extent(extent_3857, image_size):
    """
    Calculate zoom level based on geographic extent and desired image size.

    Args:
        extent_3857: QgsRectangle in EPSG:3857
        image_size: Desired output image size in pixels

    Returns:
        Appropriate zoom level
    """
    # Width of extent in meters (EPSG:3857 is in meters)
    extent_width_meters = extent_3857.width()

    # At each zoom level, calculate how many meters one tile covers
    earth_circumference = 40075016.686  # meters

    for zoom in range(20):
        # At this zoom, the world is 2^zoom tiles wide
        # Each tile is 256 pixels
        world_width_tiles = 2 ** zoom
        meters_per_tile = earth_circumference / world_width_tiles

        # How many tiles would we need to cover our extent?
        tiles_needed = extent_width_meters / meters_per_tile

        # How many pixels would that be?
        pixels_needed = tiles_needed * 256

        # If we have enough pixels, use this zoom level
        # We want slightly more pixels than requested for better quality
        if pixels_needed >= image_size * 0.8:
            return min(zoom, 19)

    return 19


def latlon_to_tile_coords(lat, lon, zoom):
    """Convert lat/lon to tile coordinates at given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x_tile, y_tile


def tile_to_latlon(x_tile, y_tile, zoom):
    """Convert tile coordinates to lat/lon (NW corner)."""
    n = 2.0 ** zoom
    lon = x_tile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def epsg3857_to_latlon(x, y):
    """Convert EPSG:3857 (Web Mercator) to WGS84 lat/lon."""
    lon = x / 20037508.34 * 180
    lat = y / 20037508.34 * 180
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
    return lat, lon


def get_map_info():
    """Get current map extent and CRS information."""
    if not iface:
        print("ERROR: QGIS interface not available")
        return None

    canvas = iface.mapCanvas()
    extent = canvas.extent()
    crs = canvas.mapSettings().destinationCrs()
    scale = canvas.scale()

    print("\n" + "="*80)
    print("CURRENT MAP INFORMATION")
    print("="*80)
    print(f"\nMap CRS: {crs.authid()} - {crs.description()}")
    print(f"Map Scale: 1:{int(scale)}")
    print(f"\nExtent in map CRS ({crs.authid()}):")
    print(f"  X: {extent.xMinimum():.2f} to {extent.xMaximum():.2f}")
    print(f"  Y: {extent.yMinimum():.2f} to {extent.yMaximum():.2f}")
    print(f"  Width: {extent.width():.2f}")
    print(f"  Height: {extent.height():.2f}")

    # Convert to EPSG:3857 (Web Mercator - what tiles use)
    crs_3857 = QgsCoordinateReferenceSystem("EPSG:3857")
    transform = QgsCoordinateTransform(crs, crs_3857, QgsProject.instance())
    extent_3857 = transform.transformBoundingBox(extent)

    print(f"\nExtent in EPSG:3857 (Web Mercator - tile coordinates):")
    print(f"  X: {extent_3857.xMinimum():.2f} to {extent_3857.xMaximum():.2f}")
    print(f"  Y: {extent_3857.yMinimum():.2f} to {extent_3857.yMaximum():.2f}")
    print(f"  Width: {extent_3857.width():.2f} meters")
    print(f"  Height: {extent_3857.height():.2f} meters")

    # Convert center to lat/lon for zoom calculation
    center_3857 = extent_3857.center()
    center_lat, center_lon = epsg3857_to_latlon(center_3857.x(), center_3857.y())

    print(f"\nCenter point:")
    print(f"  Lat/Lon: {center_lat:.6f}, {center_lon:.6f}")
    print(f"  EPSG:3857: {center_3857.x():.2f}, {center_3857.y():.2f}")

    # Calculate appropriate zoom levels using different methods
    zoom_from_scale = calculate_zoom_level_from_scale(scale, center_lat)
    zoom_from_extent_512 = calculate_zoom_level_from_extent(extent_3857, 512)
    zoom_from_extent_832 = calculate_zoom_level_from_extent(extent_3857, 832)
    zoom_from_extent_1024 = calculate_zoom_level_from_extent(extent_3857, 1024)

    print(f"\n" + "="*80)
    print("CALCULATED ZOOM LEVELS")
    print("="*80)
    print(f"Zoom from scale (1:{int(scale)}): {zoom_from_scale}")
    print(f"Zoom for 512px image: {zoom_from_extent_512}")
    print(f"Zoom for 832px image: {zoom_from_extent_832}")
    print(f"Zoom for 1024px image: {zoom_from_extent_1024}")

    # Calculate tile coverage at different zoom levels
    print(f"\n" + "="*80)
    print("TILE COVERAGE AT DIFFERENT ZOOM LEVELS")
    print("="*80)

    # Convert extent corners to lat/lon
    nw_lat, nw_lon = epsg3857_to_latlon(extent_3857.xMinimum(), extent_3857.yMaximum())
    se_lat, se_lon = epsg3857_to_latlon(extent_3857.xMaximum(), extent_3857.yMinimum())

    for zoom in [zoom_from_extent_512, zoom_from_extent_832, zoom_from_extent_1024, 17, 18, 19]:
        x_min, y_min = latlon_to_tile_coords(nw_lat, nw_lon, zoom)
        x_max, y_max = latlon_to_tile_coords(se_lat, se_lon, zoom)

        tiles_x = abs(x_max - x_min) + 1
        tiles_y = abs(y_max - y_min) + 1
        total_tiles = tiles_x * tiles_y
        total_pixels_x = tiles_x * 256
        total_pixels_y = tiles_y * 256

        print(f"\nZoom {zoom}:")
        print(f"  Tiles: {tiles_x} × {tiles_y} = {total_tiles} tiles")
        print(f"  Pixels: {total_pixels_x} × {total_pixels_y}")
        print(f"  Tile range: X[{x_min}, {x_max}], Y[{y_min}, {y_max}]")

    return {
        'extent': extent,
        'extent_3857': extent_3857,
        'crs': crs,
        'crs_3857': crs_3857,
        'scale': scale,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'zoom_from_scale': zoom_from_scale,
        'zoom_for_832': zoom_from_extent_832,
        'nw_lat': nw_lat,
        'nw_lon': nw_lon,
        'se_lat': se_lat,
        'se_lon': se_lon,
    }


def fetch_tiles_at_zoom(extent_3857, zoom, output_path):
    """
    Fetch tiles at a specific zoom level and save as image.

    Args:
        extent_3857: QgsRectangle in EPSG:3857
        zoom: Tile zoom level
        output_path: Where to save the stitched image

    Returns:
        PIL Image of stitched tiles
    """
    import requests
    from io import BytesIO

    # Convert extent to lat/lon
    nw_lat, nw_lon = epsg3857_to_latlon(extent_3857.xMinimum(), extent_3857.yMaximum())
    se_lat, se_lon = epsg3857_to_latlon(extent_3857.xMaximum(), extent_3857.yMinimum())

    # Get tile range
    x_min, y_min = latlon_to_tile_coords(nw_lat, nw_lon, zoom)
    x_max, y_max = latlon_to_tile_coords(se_lat, se_lon, zoom)

    print(f"\nFetching tiles at zoom {zoom}:")
    print(f"  Tile range: X[{x_min}, {x_max}], Y[{y_min}, {y_max}]")

    tiles_x = abs(x_max - x_min) + 1
    tiles_y = abs(y_max - y_min) + 1

    print(f"  Tiles to fetch: {tiles_x} × {tiles_y} = {tiles_x * tiles_y}")

    # Create canvas for stitching
    canvas_width = tiles_x * 256
    canvas_height = tiles_y * 256
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    # Fetch and stitch tiles
    tile_url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

    for ix, x in enumerate(range(x_min, x_max + 1)):
        for iy, y in enumerate(range(y_min, y_max + 1)):
            url = tile_url_template.format(z=zoom, x=x, y=y)

            try:
                response = requests.get(url, headers={'User-Agent': 'QGIS Magic Georeferencer'})
                response.raise_for_status()
                tile_img = Image.open(BytesIO(response.content))

                # Paste tile onto canvas
                canvas.paste(tile_img, (ix * 256, iy * 256))

            except Exception as e:
                print(f"  Failed to fetch tile {x}/{y}: {e}")
                # Fill with gray for missing tiles
                from PIL import ImageDraw
                draw = ImageDraw.Draw(canvas)
                draw.rectangle([ix * 256, iy * 256, (ix + 1) * 256, (iy + 1) * 256], fill=(200, 200, 200))

    print(f"  Stitched image size: {canvas.width} × {canvas.height}")

    # Save
    canvas.save(output_path)
    print(f"  Saved to: {output_path}")

    return canvas


def render_qgis_canvas(size=832):
    """
    Render the current QGIS canvas to an image.

    Args:
        size: Target image size (will be scaled to fit)

    Returns:
        PIL Image of the rendered canvas
    """
    if not iface:
        print("ERROR: QGIS interface not available")
        return None

    canvas = iface.mapCanvas()

    # Get current map settings
    settings = canvas.mapSettings()

    # Calculate size maintaining aspect ratio
    extent = settings.extent()
    aspect = extent.width() / extent.height()

    if aspect > 1:
        width = size
        height = int(size / aspect)
    else:
        height = size
        width = int(size * aspect)

    print(f"\nRendering QGIS canvas:")
    print(f"  Target size: {width} × {height}")

    settings.setOutputSize(QSize(width, height))

    # Render
    img = QImage(QSize(width, height), QImage.Format_RGB32)
    img.fill(0xFFFFFFFF)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)

    job = QgsMapRendererCustomPainterJob(settings, painter)
    job.start()
    job.waitForFinished()

    painter.end()

    # Convert QImage to PIL Image
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    pil_img = Image.fromarray(arr[:, :, :3])  # Drop alpha channel

    print(f"  Rendered: {pil_img.width} × {pil_img.height}")

    return pil_img


def test_model_with_different_tiles():
    """
    Test the matching model with tiles fetched at different zoom levels.
    """
    print("\n" + "="*80)
    print("TESTING MODEL WITH DIFFERENT TILE ZOOM LEVELS")
    print("="*80)

    # Get map info
    map_info = get_map_info()
    if not map_info:
        return

    # Create output directory
    output_dir = Path("/tmp/georefio_debug_tiles")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Render current QGIS canvas
    print("\n" + "-"*80)
    print("Rendering QGIS canvas as reference")
    print("-"*80)
    qgis_render = render_qgis_canvas(832)
    if qgis_render:
        qgis_path = output_dir / "qgis_canvas_render.png"
        qgis_render.save(qgis_path)
        print(f"Saved QGIS render to: {qgis_path}")

    # Test different zoom levels
    zoom_levels = [
        map_info['zoom_from_scale'],
        map_info['zoom_for_832'],
        17,
        18,
    ]

    # Remove duplicates and sort
    zoom_levels = sorted(set(zoom_levels))

    for zoom in zoom_levels:
        print("\n" + "-"*80)
        print(f"Testing zoom level {zoom}")
        print("-"*80)

        tile_path = output_dir / f"tiles_zoom_{zoom}.png"
        tile_img = fetch_tiles_at_zoom(map_info['extent_3857'], zoom, tile_path)

        # Resize to 832x832 for comparison
        resized = tile_img.resize((832, 832), Image.Resampling.LANCZOS)
        resized_path = output_dir / f"tiles_zoom_{zoom}_832x832.png"
        resized.save(resized_path)
        print(f"  Resized to 832x832: {resized_path}")

    print("\n" + "="*80)
    print("TILE COMPARISON COMPLETE")
    print("="*80)
    print(f"\nCheck {output_dir} for:")
    print("  - qgis_canvas_render.png (what QGIS is showing)")
    print("  - tiles_zoom_*.png (raw tiles at different zooms)")
    print("  - tiles_zoom_*_832x832.png (tiles resized for model)")
    print("\nCompare these images to see which zoom level best matches the QGIS view.")
    print("The tiles should have similar detail level and features visible as the QGIS render.")


def check_coordinate_transforms():
    """
    Verify that coordinate transformations are working correctly.
    """
    print("\n" + "="*80)
    print("COORDINATE TRANSFORMATION VERIFICATION")
    print("="*80)

    map_info = get_map_info()
    if not map_info:
        return

    # Test round-trip conversions
    print("\nRound-trip transformation tests:")

    # Test 1: EPSG:3857 -> Lat/Lon -> EPSG:3857
    test_x_3857 = map_info['extent_3857'].center().x()
    test_y_3857 = map_info['extent_3857'].center().y()
    lat, lon = epsg3857_to_latlon(test_x_3857, test_y_3857)

    print(f"\nTest 1: EPSG:3857 center point")
    print(f"  Input EPSG:3857: ({test_x_3857:.2f}, {test_y_3857:.2f})")
    print(f"  Converted to Lat/Lon: ({lat:.6f}, {lon:.6f})")

    # Test 2: Lat/Lon -> Tile coords -> Lat/Lon
    zoom = map_info['zoom_for_832']
    x_tile, y_tile = latlon_to_tile_coords(lat, lon, zoom)
    lat_back, lon_back = tile_to_latlon(x_tile, y_tile, zoom)

    print(f"\nTest 2: Lat/Lon to tile coords at zoom {zoom}")
    print(f"  Input Lat/Lon: ({lat:.6f}, {lon:.6f})")
    print(f"  Tile coords: ({x_tile}, {y_tile})")
    print(f"  Back to Lat/Lon (NW corner): ({lat_back:.6f}, {lon_back:.6f})")
    print(f"  Error: Lat {abs(lat - lat_back):.6f}°, Lon {abs(lon - lon_back):.6f}°")

    # Test 3: Pixel to geographic coords
    print(f"\nTest 3: Pixel to geographic coordinate mapping")
    print(f"  (This should match what gcp_generator.py does)")

    # Simulate what happens in the matching process
    extent_3857 = map_info['extent_3857']
    image_width = 832
    image_height = 832

    # Test pixel (100, 100) - should be in NW portion
    test_pixel_x = 100
    test_pixel_y = 100

    # Convert pixel to geographic (EPSG:3857)
    geo_x = extent_3857.xMinimum() + (test_pixel_x / image_width) * extent_3857.width()
    geo_y = extent_3857.yMaximum() - (test_pixel_y / image_height) * extent_3857.height()

    print(f"  Pixel ({test_pixel_x}, {test_pixel_y}) in {image_width}x{image_height} image")
    print(f"  Geographic EPSG:3857: ({geo_x:.2f}, {geo_y:.2f})")

    # Convert to lat/lon for readability
    geo_lat, geo_lon = epsg3857_to_latlon(geo_x, geo_y)
    print(f"  Geographic Lat/Lon: ({geo_lat:.6f}, {geo_lon:.6f})")

    # Check if it's within our extent
    if (extent_3857.xMinimum() <= geo_x <= extent_3857.xMaximum() and
        extent_3857.yMinimum() <= geo_y <= extent_3857.yMaximum()):
        print(f"  ✓ Point is within extent")
    else:
        print(f"  ✗ ERROR: Point is OUTSIDE extent!")


def main():
    """Run all debug checks."""
    print("\n" + "="*80)
    print("MAGIC GEOREFERENCER - TILE/ZOOM/COORDINATE DEBUG")
    print("="*80)

    # Check QGIS interface
    if not iface:
        print("\nERROR: Must be run from QGIS Python console!")
        print("\nTo run this script:")
        print("  1. Open QGIS")
        print("  2. Load a basemap (e.g., OpenStreetMap)")
        print("  3. Navigate to your area of interest")
        print("  4. Open Python Console (Ctrl+Alt+P)")
        print("  5. Run: exec(open('/home/user/georefio/debug_tile_zoom_coordinates.py').read())")
        return

    # Run all checks
    get_map_info()
    check_coordinate_transforms()
    test_model_with_different_tiles()

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the output above to understand zoom levels and extents")
    print("  2. Check /tmp/georefio_debug_tiles/ for visual comparison of tiles")
    print("  3. Verify which zoom level best matches your QGIS view")
    print("  4. If tiles look correct but matches are poor, the issue is likely in:")
    print("     - Model preprocessing (image format, normalization)")
    print("     - Coordinate conversion from match pixels to GCPs")
    print("     - The model itself not being suitable for this type of imagery")


if __name__ == "__main__":
    main()

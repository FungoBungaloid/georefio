"""
Tile Fetcher module for Magic Georeferencer

Handles basemap tile capture from QGIS canvas and external tile sources.
"""

import json
import math
import numpy as np
import requests
from pathlib import Path
from typing import Tuple, List, Optional
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
import time

try:
    from qgis.core import (
        QgsRectangle,
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsProject,
        QgsMapSettings,
        QgsMapRendererCustomPainterJob,
        QgsApplication
    )
    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QImage, QPainter
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False


class TileFetcher:
    """Fetch and stitch basemap tiles"""

    def __init__(self):
        """Initialize TileFetcher"""
        # Load tile source configurations
        config_path = Path(__file__).parent.parent / 'config' / 'tile_sources.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            self.tile_sources = json.load(f)

        # Set up cache directory
        if QGIS_AVAILABLE:
            cache_base = Path(QgsApplication.qgisSettingsDirPath())
        else:
            cache_base = Path.home() / '.qgis3'

        self.cache_dir = cache_base / 'magic_georeferencer' / 'tiles'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create requests session with proper headers for OSM tile usage policy
        # See: https://operations.osmfoundation.org/policies/tiles/
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MagicGeoreferencer/1.0 (+https://github.com/FungoBungaloid/georefio; QGIS Plugin for AI-powered georeferencing)',
        })

        # Cache expiry (7 days minimum per OSM policy)
        self.cache_expiry_days = 7

    def capture_canvas(
        self,
        iface,
        size: int = 1024
    ) -> Tuple[np.ndarray, QgsRectangle]:
        """Capture current QGIS map canvas as numpy array.

        Args:
            iface: QGIS interface
            size: Target image size (will maintain aspect ratio)

        Returns:
            Tuple of (image_array, extent_rectangle):
            - image_array: RGB numpy array [H, W, 3]
            - extent_rectangle: QgsRectangle in map CRS
        """
        if not QGIS_AVAILABLE:
            raise RuntimeError("QGIS is not available")

        # Get current map canvas
        canvas = iface.mapCanvas()
        extent = canvas.extent()

        # Get map settings
        settings = canvas.mapSettings()

        # Calculate target size maintaining aspect ratio
        canvas_width = canvas.width()
        canvas_height = canvas.height()
        aspect_ratio = canvas_width / canvas_height

        if aspect_ratio > 1:
            # Landscape
            target_width = size
            target_height = int(size / aspect_ratio)
        else:
            # Portrait
            target_height = size
            target_width = int(size * aspect_ratio)

        # Create new map settings for rendering
        render_settings = QgsMapSettings(settings)
        render_settings.setOutputSize(QSize(target_width, target_height))
        render_settings.setExtent(extent)

        # Create QImage for rendering
        image = QImage(QSize(target_width, target_height), QImage.Format_RGB32)

        # Render map
        painter = QPainter(image)
        job = QgsMapRendererCustomPainterJob(render_settings, painter)
        job.start()
        job.waitForFinished()
        painter.end()

        # Convert QImage to numpy array
        image_array = self._qimage_to_numpy(image)

        return image_array, extent

    def fetch_tiles_from_center(
        self,
        center_lat: float,
        center_lon: float,
        extent_meters: float,
        extent_dimension: str,
        source_aspect_ratio: float,
        source_name: str,
        zoom_level: int,
        target_size: int = 832
    ) -> Tuple[np.ndarray, QgsRectangle]:
        """
        Fetch tiles based on center point and one extent dimension.

        This is the new preferred method that doesn't depend on UI dimensions.

        Args:
            center_lat: Center latitude (WGS84)
            center_lon: Center longitude (WGS84)
            extent_meters: Extent in meters (width or height depending on extent_dimension)
            extent_dimension: 'horizontal' or 'vertical'
            source_aspect_ratio: Aspect ratio of source image (width/height)
            source_name: Tile source key
            zoom_level: Zoom level to fetch
            target_size: Target output image size (will be square)

        Returns:
            Tuple of (image_array, extent_rectangle):
            - image_array: RGB numpy array [H, W, 3]
            - extent_rectangle: QgsRectangle in EPSG:3857
        """
        # Convert center point to EPSG:3857
        from qgis.core import QgsPointXY

        wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')
        web_mercator = QgsCoordinateReferenceSystem('EPSG:3857')
        transform = QgsCoordinateTransform(wgs84, web_mercator, QgsProject.instance())

        center_point = QgsPointXY(center_lon, center_lat)
        center_3857 = transform.transform(center_point)

        # Calculate extent based on the specified dimension and source aspect ratio
        if extent_dimension == 'horizontal':
            # User matched horizontal extent
            width_meters = extent_meters
            height_meters = width_meters / source_aspect_ratio
        else:
            # User matched vertical extent
            height_meters = extent_meters
            width_meters = height_meters * source_aspect_ratio

        # Create extent rectangle centered on the point
        half_width = width_meters / 2
        half_height = height_meters / 2

        extent = QgsRectangle(
            center_3857.x() - half_width,
            center_3857.y() - half_height,
            center_3857.x() + half_width,
            center_3857.y() + half_height
        )

        print(f"Fetching tiles for:")
        print(f"  Center: {center_lat:.6f}, {center_lon:.6f}")
        print(f"  Extent: {width_meters:.0f}m × {height_meters:.0f}m")
        print(f"  Zoom: {zoom_level}")
        print(f"  Aspect ratio: {source_aspect_ratio:.3f}")

        # Fetch tiles for this extent
        return self.fetch_tiles(extent, web_mercator, source_name, zoom_level, target_size)

    def fetch_tiles(
        self,
        extent: QgsRectangle,
        crs: QgsCoordinateReferenceSystem,
        source_name: str,
        zoom_level: int = 17,
        size: int = 1024
    ) -> Tuple[np.ndarray, QgsRectangle]:
        """Fetch and stitch tiles for given extent.

        Args:
            extent: Geographic extent to fetch
            crs: Coordinate system of extent
            source_name: Key from tile_sources.json
            zoom_level: Tile zoom level
            size: Target output size

        Returns:
            Tuple of (stitched_image, actual_extent)
        """
        if source_name not in self.tile_sources:
            raise ValueError(f"Unknown tile source: {source_name}")

        source = self.tile_sources[source_name]

        # Transform extent to Web Mercator (EPSG:3857) for tile calculations
        web_mercator_crs = QgsCoordinateReferenceSystem('EPSG:3857')
        if crs != web_mercator_crs:
            transform = QgsCoordinateTransform(crs, web_mercator_crs, QgsProject.instance())
            extent_3857 = transform.transformBoundingBox(extent)
        else:
            extent_3857 = extent

        # Get tile coordinates
        tile_coords = self.extent_to_tile_coords(extent_3857, zoom_level)

        if not tile_coords:
            raise ValueError("No tiles found for extent")

        # Fetch tiles
        tiles = []
        for x, y in tile_coords:
            tile_img = self._fetch_single_tile(source, x, y, zoom_level)
            if tile_img is not None:
                tiles.append((x, y, tile_img))

        if not tiles:
            raise RuntimeError("Failed to fetch any tiles")

        # Stitch tiles
        stitched_image, tile_extent = self._stitch_tiles(tiles, zoom_level)

        # Resize to target size
        stitched_resized = self._resize_image(stitched_image, size)

        return stitched_resized, tile_extent

    def calculate_optimal_zoom(
        self,
        extent_meters: float,
        target_pixels: int = 832
    ) -> int:
        """
        Calculate optimal zoom level for given extent and target image size.

        Args:
            extent_meters: Extent in meters (width or height)
            target_pixels: Target image size in pixels

        Returns:
            Optimal zoom level (0-19)
        """
        # At zoom level z, one tile (256px) covers (earth_circumference / 2^z) meters
        earth_circumference = 40075016.686  # meters

        # We want extent_meters to map to approximately target_pixels
        # meters_per_pixel = extent_meters / target_pixels
        meters_per_pixel = extent_meters / target_pixels

        # At zoom z: meters_per_pixel_at_zoom = earth_circumference / (256 * 2^z)
        # Solve for z: 2^z = earth_circumference / (256 * meters_per_pixel)
        tiles_needed = earth_circumference / (256 * meters_per_pixel)
        zoom = math.log2(tiles_needed)

        # Clamp to valid range and round
        zoom = max(0, min(19, round(zoom)))

        return int(zoom)

    def get_tile_url(self, source_name: str, x: int, y: int, z: int) -> str:
        """Generate tile URL for given TMS coordinates.

        Args:
            source_name: Tile source key
            x: Tile X coordinate
            y: Tile Y coordinate
            z: Zoom level

        Returns:
            Tile URL
        """
        if source_name not in self.tile_sources:
            raise ValueError(f"Unknown tile source: {source_name}")

        source = self.tile_sources[source_name]
        url_template = source['url']

        # Handle subdomains
        if 'subdomains' in source:
            subdomains = source['subdomains']
            subdomain = subdomains[(x + y) % len(subdomains)]
            url = url_template.replace('{s}', subdomain)
        else:
            url = url_template

        # Replace placeholders
        url = url.replace('{x}', str(x))
        url = url.replace('{y}', str(y))
        url = url.replace('{z}', str(z))

        return url

    def extent_to_tile_coords(
        self,
        extent: QgsRectangle,
        zoom: int
    ) -> List[Tuple[int, int]]:
        """Convert geographic extent to tile coordinates.

        Args:
            extent: QgsRectangle in EPSG:3857 (Web Mercator)
            zoom: Zoom level

        Returns:
            List of (x, y) tile coordinates
        """
        # Web Mercator bounds
        WORLD_MERCATOR_MIN = -20037508.342789244
        WORLD_MERCATOR_MAX = 20037508.342789244
        WORLD_MERCATOR_SIZE = WORLD_MERCATOR_MAX - WORLD_MERCATOR_MIN

        n = 2 ** zoom

        # Convert extent to tile coordinates
        x_min = int(((extent.xMinimum() - WORLD_MERCATOR_MIN) / WORLD_MERCATOR_SIZE) * n)
        x_max = int(((extent.xMaximum() - WORLD_MERCATOR_MIN) / WORLD_MERCATOR_SIZE) * n)
        y_min = int(((WORLD_MERCATOR_MAX - extent.yMaximum()) / WORLD_MERCATOR_SIZE) * n)
        y_max = int(((WORLD_MERCATOR_MAX - extent.yMinimum()) / WORLD_MERCATOR_SIZE) * n)

        # Clamp to valid range
        x_min = max(0, min(x_min, n - 1))
        x_max = max(0, min(x_max, n - 1))
        y_min = max(0, min(y_min, n - 1))
        y_max = max(0, min(y_max, n - 1))

        # Generate list of tile coordinates
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tiles.append((x, y))

        return tiles

    def _fetch_single_tile(
        self,
        source: dict,
        x: int,
        y: int,
        z: int
    ) -> Optional[np.ndarray]:
        """Fetch a single tile.

        Args:
            source: Tile source configuration
            x: Tile X coordinate
            y: Tile Y coordinate
            z: Zoom level

        Returns:
            Tile image as numpy array or None if failed
        """
        # Check cache first
        cache_file = self.cache_dir / f"{source['name']}_{z}_{x}_{y}.png"
        cache_meta_file = self.cache_dir / f"{source['name']}_{z}_{x}_{y}.meta"

        # Check if we have a valid cached tile
        if cache_file.exists():
            # Check cache age (OSM policy: minimum 7 days)
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(days=self.cache_expiry_days):
                # Cache is fresh, use it
                try:
                    img = Image.open(cache_file)
                    return np.array(img)
                except:
                    pass

        # Prepare headers for conditional request if we have cached metadata
        headers = {}
        if cache_meta_file.exists():
            try:
                with open(cache_meta_file, 'r') as f:
                    meta = json.load(f)
                    if 'etag' in meta:
                        headers['If-None-Match'] = meta['etag']
                    if 'last-modified' in meta:
                        headers['If-Modified-Since'] = meta['last-modified']
            except:
                pass

        # Fetch from URL
        url = self.get_tile_url(list(self.tile_sources.keys())[
            list(self.tile_sources.values()).index(source)
        ], x, y, z)

        try:
            # Use session with proper User-Agent
            response = self.session.get(url, headers=headers, timeout=10)

            # If 304 Not Modified, use cached version
            if response.status_code == 304:
                if cache_file.exists():
                    try:
                        # Update cache file timestamp
                        cache_file.touch()
                        img = Image.open(cache_file)
                        return np.array(img)
                    except:
                        pass

            response.raise_for_status()

            img = Image.open(BytesIO(response.content))
            img_array = np.array(img)

            # Cache the tile
            try:
                img.save(cache_file)

                # Save cache metadata (ETag, Last-Modified for future conditional requests)
                meta = {}
                if 'etag' in response.headers:
                    meta['etag'] = response.headers['etag']
                if 'last-modified' in response.headers:
                    meta['last-modified'] = response.headers['last-modified']
                if meta:
                    with open(cache_meta_file, 'w') as f:
                        json.dump(meta, f)
            except Exception as e:
                print(f"Failed to cache tile: {e}")

            return img_array

        except Exception as e:
            print(f"Failed to fetch tile {x},{y},{z}: {e}")
            # If fetch failed but we have a stale cache, use it anyway
            if cache_file.exists():
                try:
                    print(f"Using stale cache for tile {x},{y},{z}")
                    img = Image.open(cache_file)
                    return np.array(img)
                except:
                    pass
            return None

    def _stitch_tiles(
        self,
        tiles: List[Tuple[int, int, np.ndarray]],
        zoom: int
    ) -> Tuple[np.ndarray, QgsRectangle]:
        """Stitch tiles into a single image.

        Args:
            tiles: List of (x, y, image_array) tuples
            zoom: Zoom level

        Returns:
            Tuple of (stitched_image, extent)
        """
        if not tiles:
            raise ValueError("No tiles to stitch")

        # Get tile grid dimensions
        x_coords = [t[0] for t in tiles]
        y_coords = [t[1] for t in tiles]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        grid_width = x_max - x_min + 1
        grid_height = y_max - y_min + 1

        # Get tile size (assume 256x256)
        tile_size = 256

        # Create output image
        output_height = grid_height * tile_size
        output_width = grid_width * tile_size
        output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        # Place tiles
        for x, y, tile_img in tiles:
            row = y - y_min
            col = x - x_min

            y_start = row * tile_size
            x_start = col * tile_size

            # Handle different tile formats (grayscale, RGB, RGBA)
            if len(tile_img.shape) == 2:
                # Grayscale - convert to RGB
                h, w = tile_img.shape
                tile_rgb = np.stack([tile_img, tile_img, tile_img], axis=-1)
            elif len(tile_img.shape) == 3:
                h, w = tile_img.shape[:2]
                if tile_img.shape[2] == 4:
                    # RGBA - drop alpha channel
                    tile_rgb = tile_img[:, :, :3]
                elif tile_img.shape[2] == 3:
                    # RGB - use as is
                    tile_rgb = tile_img
                elif tile_img.shape[2] == 1:
                    # Single channel - convert to RGB
                    tile_rgb = np.repeat(tile_img, 3, axis=2)
                else:
                    # Unknown format - skip this tile
                    print(f"Warning: Unexpected tile format with {tile_img.shape[2]} channels, skipping")
                    continue
            else:
                print(f"Warning: Unexpected tile shape {tile_img.shape}, skipping")
                continue

            output_image[y_start:y_start+h, x_start:x_start+w] = tile_rgb

        # Calculate extent
        WORLD_MERCATOR_MIN = -20037508.342789244
        WORLD_MERCATOR_MAX = 20037508.342789244
        WORLD_MERCATOR_SIZE = WORLD_MERCATOR_MAX - WORLD_MERCATOR_MIN

        n = 2 ** zoom
        tile_world_size = WORLD_MERCATOR_SIZE / n

        extent_xmin = WORLD_MERCATOR_MIN + x_min * tile_world_size
        extent_xmax = WORLD_MERCATOR_MIN + (x_max + 1) * tile_world_size
        extent_ymax = WORLD_MERCATOR_MAX - y_min * tile_world_size
        extent_ymin = WORLD_MERCATOR_MAX - (y_max + 1) * tile_world_size

        extent = QgsRectangle(extent_xmin, extent_ymin, extent_xmax, extent_ymax)

        return output_image, extent

    def _qimage_to_numpy(self, qimage: 'QImage') -> np.ndarray:
        """Convert QImage to numpy array.

        Args:
            qimage: QImage

        Returns:
            Numpy array [H, W, 3] in RGB format
        """
        # Convert to RGB32 format
        qimage = qimage.convertToFormat(QImage.Format_RGB32)

        width = qimage.width()
        height = qimage.height()

        ptr = qimage.constBits()
        ptr.setsize(height * width * 4)

        arr = np.array(ptr).reshape(height, width, 4)

        # Convert BGRA to RGB
        rgb = arr[:, :, [2, 1, 0]]

        return rgb

    def _resize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image to target size maintaining aspect ratio.

        Args:
            image: Input image
            target_size: Target size for longest edge

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        scale = target_size / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        import cv2
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return resized

"""
GCP Generator module for Magic Georeferencer

Converts match results to QGIS Ground Control Points.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
from .matcher import MatchResult

try:
    from qgis.core import (
        QgsRectangle,
        QgsCoordinateReferenceSystem,
        QgsPointXY
    )
    QGIS_AVAILABLE = True

    # QGIS 3.x GCP class
    try:
        from qgis.core import QgsGcpPoint as QgsGeorefGCP
    except ImportError:
        # Fallback for different QGIS versions
        QgsGeorefGCP = None

except ImportError:
    QGIS_AVAILABLE = False
    QgsGeorefGCP = None


class GCPGenerator:
    """Generate Ground Control Points from match results"""

    def __init__(self):
        """Initialize GCPGenerator"""
        pass

    def matches_to_gcps(
        self,
        match_result: MatchResult,
        ref_extent: QgsRectangle,
        ref_crs: QgsCoordinateReferenceSystem,
        ref_image_size: Tuple[int, int],
        src_image_size: Tuple[int, int]
    ) -> List:
        """Convert match results to QGIS Ground Control Points.

        Args:
            match_result: MatchResult from Matcher
            ref_extent: Geographic extent of reference image
            ref_crs: CRS of reference extent
            ref_image_size: (width, height) of reference image in pixels
            src_image_size: (width, height) of source image in pixels

        Returns:
            List of GCP objects (format depends on QGIS version)

        Process:
        1. For each match pair:
           a. Source coords: pixel coords in ungeoreferenced image
           b. Ref coords: pixel coords in basemap image
           c. Transform ref pixel coords to geographic coords using extent
           d. Create GCP with source pixels -> geographic coords
        """
        if not QGIS_AVAILABLE:
            raise RuntimeError("QGIS is not available")

        gcps = []

        ref_width, ref_height = ref_image_size
        src_width, src_height = src_image_size

        # Iterate through matches
        for i in range(match_result.num_matches()):
            # Source pixel coordinates (in ungeoreferenced image)
            src_x = float(match_result.keypoints_src[i, 0])
            src_y = float(match_result.keypoints_src[i, 1])

            # Reference pixel coordinates (in basemap image)
            ref_x = float(match_result.keypoints_ref[i, 0])
            ref_y = float(match_result.keypoints_ref[i, 1])

            # Convert reference pixel coords to geographic coords
            geo_point = self.pixel_to_geo(
                ref_x, ref_y,
                ref_extent,
                ref_width, ref_height
            )

            # Create GCP
            # Different QGIS versions have different GCP formats
            if QgsGeorefGCP is not None:
                # QGIS 3.x
                gcp = QgsGeorefGCP(
                    QgsPointXY(src_x, src_y),  # Pixel coords in source
                    geo_point,                  # Geographic coords
                    ref_crs                     # CRS
                )
            else:
                # Fallback: create simple dict
                gcp = {
                    'pixelPoint': (src_x, src_y),
                    'mapPoint': (geo_point.x(), geo_point.y()),
                    'crs': ref_crs,
                    'enabled': True
                }

            gcps.append(gcp)

        # Validate GCP distribution
        is_valid, message = self.validate_gcp_distribution(
            gcps,
            src_width,
            src_height
        )

        if not is_valid:
            print(f"Warning: GCP distribution validation: {message}")

        return gcps

    def pixel_to_geo(
        self,
        pixel_x: float,
        pixel_y: float,
        extent: QgsRectangle,
        image_width: int,
        image_height: int
    ) -> QgsPointXY:
        """Convert pixel coordinates to geographic coordinates.

        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            extent: Geographic extent of the image
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            QgsPointXY with geographic coordinates
        """
        # Calculate geographic coordinates
        # Pixel (0, 0) is at top-left
        # Geographic coordinates increase from bottom-left

        # Geo transform
        geo_x = extent.xMinimum() + (pixel_x / image_width) * extent.width()
        geo_y = extent.yMaximum() - (pixel_y / image_height) * extent.height()

        return QgsPointXY(geo_x, geo_y)

    def validate_gcp_distribution(
        self,
        gcps: List,
        image_width: int,
        image_height: int
    ) -> Tuple[bool, str]:
        """Validate that GCPs are well-distributed.

        Args:
            gcps: List of GCPs
            image_width: Width of source image
            image_height: Height of source image

        Returns:
            Tuple of (is_valid, message)

        Checks:
        - Minimum number of GCPs (4+ for perspective, 6+ for polynomial)
        - Coverage across image quadrants
        - No excessive clustering
        """
        num_gcps = len(gcps)

        # Check minimum number
        if num_gcps < 4:
            return False, f"Insufficient GCPs: {num_gcps} (minimum 4 required)"

        if num_gcps < 6:
            return True, f"Warning: Only {num_gcps} GCPs (6+ recommended for polynomial transforms)"

        # Extract pixel coordinates
        if QgsGeorefGCP is not None and hasattr(gcps[0], 'sourcePoint'):
            # QGIS 3.x GCP objects
            pixel_coords = np.array([
                [gcp.sourcePoint().x(), gcp.sourcePoint().y()]
                for gcp in gcps
            ])
        elif isinstance(gcps[0], dict):
            # Dict format
            pixel_coords = np.array([
                gcp['pixelPoint']
                for gcp in gcps
            ])
        else:
            # Unknown format, skip validation
            return True, "GCP format unknown, skipping distribution validation"

        # Check coverage across quadrants
        # Divide image into 2x2 grid
        quadrants = np.zeros(4, dtype=int)

        for px, py in pixel_coords:
            # Determine quadrant (0-3)
            qx = 0 if px < image_width / 2 else 1
            qy = 0 if py < image_height / 2 else 1
            quadrant = qy * 2 + qx
            quadrants[quadrant] += 1

        # Check if all quadrants have at least one point
        empty_quadrants = np.sum(quadrants == 0)

        if empty_quadrants > 1:
            return True, f"Warning: {empty_quadrants} quadrants have no GCPs (may affect accuracy)"

        # Check for excessive clustering
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist

        if len(pixel_coords) >= 2:
            try:
                distances = pdist(pixel_coords)
                min_distance = np.min(distances)
                max_distance = np.max(distances)

                # Check if points are too clustered
                image_diagonal = np.sqrt(image_width**2 + image_height**2)
                clustering_threshold = image_diagonal * 0.05  # 5% of diagonal

                if min_distance < clustering_threshold:
                    return True, "Warning: Some GCPs are very close together (clustering detected)"

            except:
                # scipy not available, skip clustering check
                pass

        return True, "GCP distribution is good"

    def export_gcp_file(self, gcps: List, filepath: Path):
        """Export GCPs to QGIS .points file format.

        Args:
            gcps: List of GCPs
            filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write("mapX,mapY,pixelX,pixelY,enable,dX,dY,residual\n")

            # Write GCPs
            for i, gcp in enumerate(gcps):
                if QgsGeorefGCP is not None and hasattr(gcp, 'sourcePoint'):
                    # QGIS 3.x format
                    pixel_x = gcp.sourcePoint().x()
                    pixel_y = gcp.sourcePoint().y()
                    map_x = gcp.mapPoint().x()
                    map_y = gcp.mapPoint().y()
                    enabled = 1
                elif isinstance(gcp, dict):
                    # Dict format
                    pixel_x, pixel_y = gcp['pixelPoint']
                    map_x, map_y = gcp['mapPoint']
                    enabled = 1 if gcp.get('enabled', True) else 0
                else:
                    continue

                # Write line (residuals will be 0 initially)
                f.write(f"{map_x},{map_y},{pixel_x},{pixel_y},{enabled},0,0,0\n")

    def load_gcp_file(self, filepath: Path, crs: QgsCoordinateReferenceSystem) -> List:
        """Load GCPs from QGIS .points file format.

        Args:
            filepath: Input file path
            crs: CRS for the GCPs

        Returns:
            List of GCPs
        """
        gcps = []

        with open(filepath, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)

            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 5:
                    continue

                map_x = float(parts[0])
                map_y = float(parts[1])
                pixel_x = float(parts[2])
                pixel_y = float(parts[3])
                enabled = int(parts[4]) == 1

                if not enabled:
                    continue

                # Create GCP
                if QgsGeorefGCP is not None:
                    gcp = QgsGeorefGCP(
                        QgsPointXY(pixel_x, pixel_y),
                        QgsPointXY(map_x, map_y),
                        crs
                    )
                else:
                    gcp = {
                        'pixelPoint': (pixel_x, pixel_y),
                        'mapPoint': (map_x, map_y),
                        'crs': crs,
                        'enabled': enabled
                    }

                gcps.append(gcp)

        return gcps

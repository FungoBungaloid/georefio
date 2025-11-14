"""
Georeferencer module for Magic Georeferencer

Handles georeferencing using QGIS/GDAL tools.
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsRasterLayer,
        QgsProject
    )
    from qgis.PyQt.QtCore import QFileInfo
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False


class Georeferencer:
    """Handle georeferencing operations"""

    def __init__(self, iface=None):
        """Initialize Georeferencer.

        Args:
            iface: QGIS interface (optional)
        """
        self.iface = iface

    def georeference_image(
        self,
        source_image_path: Path,
        gcps: List,
        output_path: Path,
        target_crs: QgsCoordinateReferenceSystem,
        transform_type: str = 'polynomial_1',
        resampling: str = 'cubic',
        compression: str = 'LZW'
    ) -> Tuple[bool, str]:
        """Georeference image using GDAL.

        Args:
            source_image_path: Path to ungeoreferenced image
            gcps: List of ground control points
            output_path: Output georeferenced raster path
            target_crs: Target coordinate reference system
            transform_type: 'polynomial_1', 'polynomial_2', 'polynomial_3', 'thin_plate_spline', 'projective'
            resampling: 'nearest', 'bilinear', 'cubic', 'lanczos'
            compression: 'NONE', 'LZW', 'PACKBITS', 'DEFLATE'

        Returns:
            Tuple of (success, message)
        """
        if not source_image_path.exists():
            return False, f"Source image not found: {source_image_path}"

        try:
            # Create temporary GCP file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.points', delete=False) as f:
                gcp_file = Path(f.name)
                self._write_gcp_file(gcps, f, target_crs)

            # Map transform type to GDAL order
            transform_order_map = {
                'polynomial_1': 1,
                'polynomial_2': 2,
                'polynomial_3': 3,
                'thin_plate_spline': -1,  # TPS
                'projective': 0  # Linear (projective)
            }

            order = transform_order_map.get(transform_type, 1)

            # Map resampling method
            resampling_map = {
                'nearest': 'near',
                'bilinear': 'bilinear',
                'cubic': 'cubic',
                'lanczos': 'lanczos'
            }
            resampling_method = resampling_map.get(resampling, 'cubic')

            # Build GDAL command
            cmd = [
                'gdalwarp',
                '-r', resampling_method,
                '-co', f'COMPRESS={compression}',
                '-t_srs', target_crs.authid(),
            ]

            # Add transformation type
            if order == -1:
                cmd.extend(['-tps'])  # Thin plate spline
            else:
                cmd.extend(['-order', str(order)])

            # Add GCP points directly (alternative to GCP file)
            for gcp in gcps:
                pixel_x, pixel_y, map_x, map_y = self._extract_gcp_coords(gcp)
                cmd.extend(['-gcp', str(pixel_x), str(pixel_y), str(map_x), str(map_y)])

            # Add input and output
            cmd.extend([str(source_image_path), str(output_path)])

            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            # Cleanup temp file
            if gcp_file.exists():
                gcp_file.unlink()

            if result.returncode != 0:
                return False, f"GDAL error: {result.stderr}"

            # Load result into QGIS if interface available
            if self.iface is not None and output_path.exists():
                self.load_raster(output_path)

            return True, "Georeferencing completed successfully"

        except subprocess.TimeoutExpired:
            return False, "Georeferencing timed out (> 5 minutes)"
        except Exception as e:
            return False, f"Georeferencing failed: {str(e)}"

    def calculate_transform_error(
        self,
        gcps: List,
        transform_type: str
    ) -> Tuple[float, List[float]]:
        """Calculate reprojection error for given transform.

        Args:
            gcps: List of GCPs
            transform_type: Transform type

        Returns:
            Tuple of (mean_error, per_gcp_errors) in pixels
        """
        if len(gcps) < 4:
            return 0.0, []

        try:
            # Extract coordinates
            pixel_coords = []
            map_coords = []

            for gcp in gcps:
                pixel_x, pixel_y, map_x, map_y = self._extract_gcp_coords(gcp)
                pixel_coords.append([pixel_x, pixel_y])
                map_coords.append([map_x, map_y])

            pixel_coords = np.array(pixel_coords)
            map_coords = np.array(map_coords)

            # Estimate transformation
            if transform_type == 'polynomial_1' or transform_type == 'projective':
                # Affine/projective transform
                import cv2
                H, _ = cv2.findHomography(pixel_coords, map_coords, 0)

                if H is None:
                    return 0.0, []

                # Calculate reprojection errors
                pixel_coords_homo = np.hstack([pixel_coords, np.ones((len(pixel_coords), 1))])
                projected = (H @ pixel_coords_homo.T).T
                projected = projected[:, :2] / projected[:, 2:3]

                errors = np.linalg.norm(map_coords - projected, axis=1)

            else:
                # For higher order, use simple leave-one-out cross-validation
                errors = []
                for i in range(len(gcps)):
                    # Leave one out
                    train_idx = [j for j in range(len(gcps)) if j != i]
                    test_pixel = pixel_coords[i]
                    test_map = map_coords[i]

                    # Fit transform on remaining points
                    import cv2
                    H, _ = cv2.findHomography(pixel_coords[train_idx], map_coords[train_idx], 0)

                    if H is None:
                        errors.append(0.0)
                        continue

                    # Predict test point
                    test_pixel_homo = np.array([test_pixel[0], test_pixel[1], 1.0])
                    projected = H @ test_pixel_homo
                    projected = projected[:2] / projected[2]

                    error = np.linalg.norm(test_map - projected)
                    errors.append(error)

                errors = np.array(errors)

            mean_error = float(np.mean(errors))
            return mean_error, errors.tolist()

        except Exception as e:
            print(f"Error calculating transform error: {e}")
            return 0.0, []

    def suggest_transform_type(
        self,
        num_gcps: int,
        distribution_quality: float
    ) -> str:
        """Suggest appropriate transformation type.

        Args:
            num_gcps: Number of GCPs
            distribution_quality: Distribution quality score (0-1)

        Returns:
            Suggested transform type

        Rules:
        - 4-5 GCPs: polynomial_1 (affine)
        - 6-9 GCPs: polynomial_1 or polynomial_2
        - 10+ GCPs: polynomial_2 or thin_plate_spline
        - Poor distribution: polynomial_1 (more stable)
        - Good distribution + many GCPs: polynomial_3 or thin_plate_spline
        """
        if num_gcps < 4:
            return 'polynomial_1'  # Default

        if num_gcps <= 5:
            return 'polynomial_1'  # Affine

        if num_gcps <= 9:
            if distribution_quality < 0.5:
                return 'polynomial_1'  # More stable
            else:
                return 'polynomial_2'

        # 10+ GCPs
        if distribution_quality < 0.5:
            return 'polynomial_2'
        elif distribution_quality < 0.7:
            return 'polynomial_2'
        else:
            return 'thin_plate_spline'  # Most flexible

    def load_raster(self, raster_path: Path) -> bool:
        """Load georeferenced raster into QGIS.

        Args:
            raster_path: Path to raster file

        Returns:
            True if loaded successfully
        """
        if not QGIS_AVAILABLE or self.iface is None:
            return False

        try:
            # Create raster layer
            file_info = QFileInfo(str(raster_path))
            layer_name = file_info.baseName()

            raster_layer = QgsRasterLayer(str(raster_path), layer_name)

            if not raster_layer.isValid():
                return False

            # Add to project
            QgsProject.instance().addMapLayer(raster_layer)

            # Zoom to layer
            if self.iface.mapCanvas():
                self.iface.mapCanvas().setExtent(raster_layer.extent())
                self.iface.mapCanvas().refresh()

            return True

        except Exception as e:
            print(f"Error loading raster: {e}")
            return False

    def _write_gcp_file(self, gcps: List, file_handle, crs: QgsCoordinateReferenceSystem):
        """Write GCPs to file in QGIS .points format.

        Args:
            gcps: List of GCPs
            file_handle: File handle to write to
            crs: Coordinate reference system
        """
        # Write header
        file_handle.write("mapX,mapY,pixelX,pixelY,enable,dX,dY,residual\n")

        # Write GCPs
        for gcp in gcps:
            pixel_x, pixel_y, map_x, map_y = self._extract_gcp_coords(gcp)
            file_handle.write(f"{map_x},{map_y},{pixel_x},{pixel_y},1,0,0,0\n")

    def _extract_gcp_coords(self, gcp) -> Tuple[float, float, float, float]:
        """Extract coordinates from GCP object.

        Args:
            gcp: GCP object (various formats supported)

        Returns:
            Tuple of (pixel_x, pixel_y, map_x, map_y)
        """
        # Try QGIS GCP object
        if hasattr(gcp, 'sourcePoint') and hasattr(gcp, 'mapPoint'):
            pixel_x = gcp.sourcePoint().x()
            pixel_y = gcp.sourcePoint().y()
            map_x = gcp.mapPoint().x()
            map_y = gcp.mapPoint().y()
        # Try dict format
        elif isinstance(gcp, dict):
            pixel_x, pixel_y = gcp['pixelPoint']
            map_x, map_y = gcp['mapPoint']
        else:
            raise ValueError(f"Unknown GCP format: {type(gcp)}")

        return float(pixel_x), float(pixel_y), float(map_x), float(map_y)

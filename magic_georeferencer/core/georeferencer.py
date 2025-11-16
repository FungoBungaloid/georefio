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
        compression: str = 'LZW',
        progress_callback: Optional[callable] = None
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

            # Step 1: Use gdal_translate to add GCPs to the source image
            # Create a temporary VRT file with GCPs
            temp_vrt = tempfile.NamedTemporaryFile(mode='w', suffix='.vrt', delete=False)
            temp_vrt_path = Path(temp_vrt.name)
            temp_vrt.close()

            # Build gdal_translate command with GCPs
            translate_cmd = ['gdal_translate']

            # Add GCP points
            for gcp in gcps:
                pixel_x, pixel_y, map_x, map_y = self._extract_gcp_coords(gcp)
                translate_cmd.extend(['-gcp', str(pixel_x), str(pixel_y), str(map_x), str(map_y)])

            # Add target CRS
            translate_cmd.extend(['-a_srs', target_crs.authid()])

            # Output as VRT (virtual format, fast)
            translate_cmd.extend(['-of', 'VRT'])

            # Add input and output
            translate_cmd.extend([str(source_image_path), str(temp_vrt_path)])

            if progress_callback:
                progress_callback("Step 1/3: Adding GCPs to image...", 10, 100)

            print("\n" + "="*80)
            print("STEP 1: Adding GCPs with gdal_translate")
            print("="*80)
            print(f"Command: {' '.join(translate_cmd[:10])}...")
            print(f"GCPs: {len(gcps)}")
            print(f"Output VRT: {temp_vrt_path}")
            print("="*80 + "\n")

            # Execute gdal_translate
            result1 = subprocess.run(
                translate_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if progress_callback:
                progress_callback("Step 1/3: GCPs added successfully", 40, 100)

            print(f"gdal_translate return code: {result1.returncode}")
            if result1.stdout:
                print(f"gdal_translate stdout: {result1.stdout}")
            if result1.stderr:
                print(f"gdal_translate stderr: {result1.stderr}")

            if result1.returncode != 0:
                if temp_vrt_path.exists():
                    temp_vrt_path.unlink()
                error_msg = f"gdal_translate error (return code {result1.returncode}):\n{result1.stderr}"
                if "not recognized" in result1.stderr or "not found" in result1.stderr:
                    error_msg += "\n\nNote: gdal_translate may not be in your system PATH. " \
                                "Please ensure GDAL is installed and accessible from QGIS."
                return False, error_msg

            # Step 2: Use gdalwarp to warp the VRT with GCPs to final output
            # Build gdalwarp command
            warp_cmd = [
                'gdalwarp',
                '-r', resampling_method,
                '-co', f'COMPRESS={compression}',
                '-t_srs', target_crs.authid(),
            ]

            # Add transformation type
            if order == -1:
                warp_cmd.extend(['-tps'])  # Thin plate spline
            else:
                warp_cmd.extend(['-order', str(order)])

            # Add input VRT and output
            warp_cmd.extend([str(temp_vrt_path), str(output_path)])

            if progress_callback:
                progress_callback("Step 2/3: Warping image (this may take a while)...", 50, 100)

            print("\n" + "="*80)
            print("STEP 2: Warping with gdalwarp")
            print("="*80)
            print(f"Command: {' '.join(warp_cmd[:10])}...")
            print(f"Transform type: {transform_type} (order={order})")
            print(f"Resampling: {resampling_method}")
            print(f"Source VRT: {temp_vrt_path}")
            print(f"Output: {output_path}")
            print(f"CRS: {target_crs.authid()}")
            print("="*80 + "\n")

            # Execute gdalwarp
            result2 = subprocess.run(
                warp_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if progress_callback:
                progress_callback("Step 2/3: Warping complete", 80, 100)

            # Cleanup temp VRT
            if temp_vrt_path.exists():
                temp_vrt_path.unlink()

            # Debug: Print result
            print(f"gdalwarp return code: {result2.returncode}")
            if result2.stdout:
                print(f"gdalwarp stdout: {result2.stdout}")
            if result2.stderr:
                print(f"gdalwarp stderr: {result2.stderr}")

            if result2.returncode != 0:
                error_msg = f"gdalwarp error (return code {result2.returncode}):\n{result2.stderr}"
                if "not recognized" in result2.stderr or "not found" in result2.stderr:
                    error_msg += "\n\nNote: gdalwarp may not be in your system PATH. " \
                                "Please ensure GDAL is installed and accessible from QGIS."
                return False, error_msg

            # Check if output file was created
            if not output_path.exists():
                return False, f"Output file was not created: {output_path}"

            print(f"✓ Georeferencing complete, output file created: {output_path}")

            # Load result into QGIS if interface available
            if self.iface is not None:
                if progress_callback:
                    progress_callback("Step 3/3: Loading into QGIS...", 90, 100)

                print(f"Loading georeferenced raster into QGIS...")
                load_success = self.load_raster(output_path)
                if load_success:
                    print(f"✓ Raster loaded successfully into QGIS")
                else:
                    print(f"⚠ Failed to load raster into QGIS (but file was created)")

                if progress_callback:
                    progress_callback("Complete!", 100, 100)

            return True, "Georeferencing completed successfully"

        except subprocess.TimeoutExpired:
            return False, "Georeferencing timed out (> 5 minutes)"
        except FileNotFoundError as e:
            return False, f"gdalwarp not found. Please ensure GDAL is installed and in your PATH.\n\nError: {str(e)}"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Georeferencing exception:\n{error_details}")
            return False, f"Georeferencing failed: {str(e)}\n\nSee QGIS Python console for details."

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

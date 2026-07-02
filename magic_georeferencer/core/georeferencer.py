"""
Georeferencer module for Magic Georeferencer

Handles georeferencing using QGIS/GDAL tools.

The primary path uses the GDAL Python bindings (osgeo.gdal), which are always
bundled with QGIS, so no external executables are required. If the bindings
are unavailable (e.g. running outside QGIS with a minimal GDAL install), it
falls back to invoking gdal_translate/gdalwarp on the PATH.
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from .logging_utils import get_logger

logger = get_logger('georeferencer')

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

try:
    from osgeo import gdal
    GDAL_PYTHON_AVAILABLE = True
except ImportError:
    GDAL_PYTHON_AVAILABLE = False

# GDAL warp polynomial order per transform type (-1 = thin plate spline)
TRANSFORM_ORDER_MAP = {
    'polynomial_1': 1,
    'polynomial_2': 2,
    'polynomial_3': 3,
    'thin_plate_spline': -1,
    'projective': 0,
}

RESAMPLING_MAP = {
    'nearest': 'near',
    'bilinear': 'bilinear',
    'cubic': 'cubic',
    'lanczos': 'lanczos',
}


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
            progress_callback: Optional callback(status, current, total)

        Returns:
            Tuple of (success, message)
        """
        source_image_path = Path(source_image_path)
        output_path = Path(output_path)

        if not source_image_path.exists():
            return False, f"Source image not found: {source_image_path}"

        if len(gcps) < 3:
            return False, f"At least 3 GCPs are required (got {len(gcps)})"

        order = TRANSFORM_ORDER_MAP.get(transform_type, 1)
        resampling_method = RESAMPLING_MAP.get(resampling, 'cubic')

        try:
            if GDAL_PYTHON_AVAILABLE:
                success, message = self._georeference_with_gdal_python(
                    source_image_path, gcps, output_path, target_crs,
                    order, resampling_method, compression, progress_callback
                )
            else:
                logger.warning(
                    "GDAL Python bindings unavailable; falling back to "
                    "gdal_translate/gdalwarp executables on PATH"
                )
                success, message = self._georeference_with_subprocess(
                    source_image_path, gcps, output_path, target_crs,
                    order, resampling_method, compression, progress_callback
                )

            if not success:
                return False, message

            if not output_path.exists():
                return False, f"Output file was not created: {output_path}"

            logger.info(f"Georeferencing complete: {output_path}")

            # Load result into QGIS if interface available
            if self.iface is not None:
                if progress_callback:
                    progress_callback("Loading result into QGIS...", 90, 100)

                if self.load_raster(output_path):
                    logger.info("Georeferenced raster loaded into QGIS")
                else:
                    logger.warning("Failed to load raster into QGIS (but file was created)")

            if progress_callback:
                progress_callback("Complete!", 100, 100)

            return True, "Georeferencing completed successfully"

        except Exception as e:
            import traceback
            logger.error(f"Georeferencing exception:\n{traceback.format_exc()}")
            return False, f"Georeferencing failed: {str(e)}\n\nSee QGIS Python console for details."

    def _gdal_gcps(self, gcps: List) -> List['gdal.GCP']:
        """Convert plugin GCPs to gdal.GCP objects."""
        gdal_gcps = []
        for gcp in gcps:
            pixel_x, pixel_y, map_x, map_y = self._extract_gcp_coords(gcp)
            # gdal.GCP(x, y, z, pixel, line)
            gdal_gcps.append(gdal.GCP(map_x, map_y, 0.0, pixel_x, pixel_y))
        return gdal_gcps

    def _georeference_with_gdal_python(
        self,
        source_image_path: Path,
        gcps: List,
        output_path: Path,
        target_crs,
        order: int,
        resampling_method: str,
        compression: str,
        progress_callback: Optional[callable]
    ) -> Tuple[bool, str]:
        """Georeference using the GDAL Python API (no external executables)."""
        gdal.UseExceptions()

        srs_wkt = target_crs.toWkt() if hasattr(target_crs, 'toWkt') else str(target_crs)
        authid = target_crs.authid() if hasattr(target_crs, 'authid') else str(target_crs)

        if progress_callback:
            progress_callback("Attaching ground control points...", 10, 100)

        logger.info(
            f"GDAL Python georeferencing: {len(gcps)} GCPs, order={order}, "
            f"resampling={resampling_method}, CRS={authid}"
        )

        src_ds = gdal.Open(str(source_image_path))
        if src_ds is None:
            return False, f"GDAL could not open source image: {source_image_path}"

        vrt_ds = None
        try:
            # Step 1: attach GCPs via an in-memory VRT (equivalent of gdal_translate -gcp)
            vrt_ds = gdal.Translate(
                '',
                src_ds,
                format='VRT',
                GCPs=self._gdal_gcps(gcps),
                outputSRS=srs_wkt
            )
            if vrt_ds is None:
                return False, "gdal.Translate failed to attach GCPs"

            if progress_callback:
                progress_callback("Warping image (this may take a while)...", 40, 100)

            # Step 2: warp to the final output (equivalent of gdalwarp)
            warp_options = {
                'dstSRS': srs_wkt,
                'resampleAlg': resampling_method,
                'creationOptions': [f'COMPRESS={compression}'],
                'format': 'GTiff',
            }
            if order == -1:
                warp_options['tps'] = True
            else:
                warp_options['polynomialOrder'] = order

            def _gdal_progress(fraction, message, data):
                if progress_callback:
                    # Map warp progress onto the 40-85% range
                    progress_callback("Warping image...", 40 + int(fraction * 45), 100)
                return 1  # continue

            out_ds = gdal.Warp(
                str(output_path),
                vrt_ds,
                callback=_gdal_progress,
                **warp_options
            )
            if out_ds is None:
                return False, "gdal.Warp failed to produce output"

            out_ds = None  # flush and close

            return True, "OK"

        except Exception as e:
            return False, f"GDAL error: {e}"
        finally:
            vrt_ds = None
            src_ds = None

    def _georeference_with_subprocess(
        self,
        source_image_path: Path,
        gcps: List,
        output_path: Path,
        target_crs,
        order: int,
        resampling_method: str,
        compression: str,
        progress_callback: Optional[callable]
    ) -> Tuple[bool, str]:
        """Georeference by invoking gdal_translate/gdalwarp on the PATH."""
        authid = target_crs.authid() if hasattr(target_crs, 'authid') else str(target_crs)

        temp_vrt = tempfile.NamedTemporaryFile(mode='w', suffix='.vrt', delete=False)
        temp_vrt_path = Path(temp_vrt.name)
        temp_vrt.close()

        try:
            # Step 1: gdal_translate to attach GCPs into a VRT
            translate_cmd = ['gdal_translate']
            for gcp in gcps:
                pixel_x, pixel_y, map_x, map_y = self._extract_gcp_coords(gcp)
                translate_cmd.extend(['-gcp', str(pixel_x), str(pixel_y), str(map_x), str(map_y)])
            translate_cmd.extend(['-a_srs', authid])
            translate_cmd.extend(['-of', 'VRT'])
            translate_cmd.extend([str(source_image_path), str(temp_vrt_path)])

            if progress_callback:
                progress_callback("Attaching ground control points...", 10, 100)

            logger.info(f"Running gdal_translate with {len(gcps)} GCPs")

            result1 = subprocess.run(
                translate_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result1.returncode != 0:
                error_msg = f"gdal_translate error (return code {result1.returncode}):\n{result1.stderr}"
                if "not recognized" in result1.stderr or "not found" in result1.stderr:
                    error_msg += ("\n\nNote: gdal_translate may not be in your system PATH. "
                                  "Please ensure GDAL is installed and accessible from QGIS.")
                return False, error_msg

            # Step 2: gdalwarp the VRT to the final output
            warp_cmd = [
                'gdalwarp',
                '-r', resampling_method,
                '-co', f'COMPRESS={compression}',
                '-t_srs', authid,
            ]
            if order == -1:
                warp_cmd.append('-tps')
            else:
                warp_cmd.extend(['-order', str(order)])
            warp_cmd.extend([str(temp_vrt_path), str(output_path)])

            if progress_callback:
                progress_callback("Warping image (this may take a while)...", 50, 100)

            logger.info(f"Running gdalwarp (order={order}, resampling={resampling_method})")

            result2 = subprocess.run(
                warp_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )

            if result2.returncode != 0:
                error_msg = f"gdalwarp error (return code {result2.returncode}):\n{result2.stderr}"
                if "not recognized" in result2.stderr or "not found" in result2.stderr:
                    error_msg += ("\n\nNote: gdalwarp may not be in your system PATH. "
                                  "Please ensure GDAL is installed and accessible from QGIS.")
                return False, error_msg

            return True, "OK"

        except subprocess.TimeoutExpired:
            return False, "Georeferencing timed out (> 5 minutes)"
        except FileNotFoundError as e:
            return False, (
                "GDAL executables not found. Please ensure GDAL is installed "
                f"and in your PATH.\n\nError: {str(e)}"
            )
        finally:
            if temp_vrt_path.exists():
                temp_vrt_path.unlink()

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
            Tuple of (mean_error, per_gcp_errors) in map units
        """
        if len(gcps) < 4:
            return 0.0, []

        try:
            import cv2

            pixel_coords = []
            map_coords = []

            for gcp in gcps:
                pixel_x, pixel_y, map_x, map_y = self._extract_gcp_coords(gcp)
                pixel_coords.append([pixel_x, pixel_y])
                map_coords.append([map_x, map_y])

            pixel_coords = np.array(pixel_coords)
            map_coords = np.array(map_coords)

            if transform_type in ('polynomial_1', 'projective'):
                # Fit a single homography and measure residuals
                H, _ = cv2.findHomography(pixel_coords, map_coords, 0)

                if H is None:
                    return 0.0, []

                pixel_coords_homo = np.hstack([pixel_coords, np.ones((len(pixel_coords), 1))])
                projected = (H @ pixel_coords_homo.T).T
                projected = projected[:, :2] / projected[:, 2:3]

                errors = np.linalg.norm(map_coords - projected, axis=1)

            else:
                # For higher order, use leave-one-out cross-validation
                errors = []
                for i in range(len(gcps)):
                    train_idx = [j for j in range(len(gcps)) if j != i]
                    test_pixel = pixel_coords[i]
                    test_map = map_coords[i]

                    H, _ = cv2.findHomography(pixel_coords[train_idx], map_coords[train_idx], 0)

                    if H is None:
                        errors.append(0.0)
                        continue

                    test_pixel_homo = np.array([test_pixel[0], test_pixel[1], 1.0])
                    projected = H @ test_pixel_homo
                    projected = projected[:2] / projected[2]

                    errors.append(np.linalg.norm(test_map - projected))

                errors = np.array(errors)

            return float(np.mean(errors)), errors.tolist()

        except Exception as e:
            logger.warning(f"Error calculating transform error: {e}")
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
        - Poor distribution: lower order (more stable)
        """
        if num_gcps <= 5:
            return 'polynomial_1'

        if num_gcps <= 9:
            return 'polynomial_1' if distribution_quality < 0.5 else 'polynomial_2'

        # 10+ GCPs
        if distribution_quality < 0.7:
            return 'polynomial_2'
        return 'thin_plate_spline'

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
            file_info = QFileInfo(str(raster_path))
            layer_name = file_info.baseName()

            raster_layer = QgsRasterLayer(str(raster_path), layer_name)

            if not raster_layer.isValid():
                return False

            QgsProject.instance().addMapLayer(raster_layer)

            # Zoom to layer
            if self.iface.mapCanvas():
                self.iface.mapCanvas().setExtent(raster_layer.extent())
                self.iface.mapCanvas().refresh()

            return True

        except Exception as e:
            logger.warning(f"Error loading raster: {e}")
            return False

    def _extract_gcp_coords(self, gcp) -> Tuple[float, float, float, float]:
        """Extract coordinates from GCP object.

        Args:
            gcp: GCP object (QGIS GCP or dict format)

        Returns:
            Tuple of (pixel_x, pixel_y, map_x, map_y)
        """
        if hasattr(gcp, 'sourcePoint') and hasattr(gcp, 'mapPoint'):
            pixel_x = gcp.sourcePoint().x()
            pixel_y = gcp.sourcePoint().y()
            map_x = gcp.mapPoint().x()
            map_y = gcp.mapPoint().y()
        elif isinstance(gcp, dict):
            pixel_x, pixel_y = gcp['pixelPoint']
            map_x, map_y = gcp['mapPoint']
        else:
            raise ValueError(f"Unknown GCP format: {type(gcp)}")

        return float(pixel_x), float(pixel_y), float(map_x), float(map_y)

"""
Matcher module for Magic Georeferencer

Handles progressive multi-scale image matching using MatchAnything model.
"""

import math
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from .logging_utils import get_logger

logger = get_logger('matcher')


@dataclass
class MatchResult:
    """Container for match results"""

    # Keypoints in source image coordinates
    keypoints_src: np.ndarray  # [N, 2] (x, y)

    # Keypoints in reference image coordinates
    keypoints_ref: np.ndarray  # [N, 2] (x, y)

    # Per-match confidence scores
    confidence: np.ndarray  # [N]

    # Geometric consistency (post-RANSAC)
    geometric_score: float  # 0-1

    # Spatial distribution quality
    distribution_quality: float  # 0-1

    # Estimated transformation matrix (if computed)
    transform_matrix: Optional[np.ndarray] = None  # [3, 3] homography

    # Processing metadata
    processing_time: float = 0.0  # seconds
    device: str = 'cpu'  # 'cuda' or 'cpu'
    resolution: int = 512  # Image size used for matching

    def num_matches(self) -> int:
        """Get number of matches"""
        return len(self.keypoints_src)

    def mean_confidence(self) -> float:
        """Get mean confidence score"""
        if len(self.confidence) == 0:
            return 0.0
        return float(np.mean(self.confidence))

    def quality_score(self) -> float:
        """Combined quality score used to compare candidate matches.

        Weights match count (capped), confidence, geometric consistency
        and spatial distribution.
        """
        return (
            min(self.num_matches() / 100.0, 1.0) * 0.25 +
            self.mean_confidence() * 0.35 +
            self.geometric_score * 0.25 +
            self.distribution_quality * 0.15
        )


class Matcher:
    """Handle image matching with progressive refinement"""

    # Stop the scale search early once a candidate scores at least this well
    EARLY_EXIT_QUALITY = 0.75

    def __init__(self, model_manager):
        """Initialize Matcher.

        Args:
            model_manager: ModelManager instance with loaded model
        """
        self.model_manager = model_manager
        self.config = model_manager.get_inference_config()

        if model_manager.model is None:
            raise RuntimeError("Model not loaded. Call model_manager.load_model() first.")

        self.model = model_manager.model

        # Debug directory (can be set by UI, None by default)
        self.debug_dir = None

    def match_progressive(
        self,
        image_src: np.ndarray,
        image_ref: np.ndarray,
        scales: Optional[List[int]] = None,
        min_gcps: int = 6
    ) -> MatchResult:
        """Multi-scale progressive matching.

        Args:
            image_src: Source (ungeoreferenced) image [H, W, 3]
            image_ref: Reference (basemap) image [H, W, 3]
            scales: List of image sizes to try, e.g., [512, 768, 1024]
                   If None, uses single scale from config
            min_gcps: Minimum number of GCPs required

        Returns:
            MatchResult containing keypoints, confidence scores, etc.
        """
        if scales is None:
            return self.match_single_scale(image_src, image_ref, self.config['size'])

        best_result = None

        for size in scales:
            result = self.match_single_scale(image_src, image_ref, size)
            result = self.filter_matches(result, confidence_threshold=0.7, min_gcps=min_gcps)

            if result.num_matches() >= min_gcps:
                best_result = result
            else:
                # Not enough matches at this scale; keep the best so far
                break

        if best_result is None:
            return self._empty_result()

        return best_result

    def match_multi_zoom(
        self,
        image_src: np.ndarray,
        center_lat: float,
        center_lon: float,
        extent_meters: float,
        extent_dimension: str,
        tile_fetcher,
        basemap_source: str,
        base_zoom: int,
        zoom_range: int = 2
    ) -> Tuple[MatchResult, np.ndarray, 'QgsRectangle', int]:
        """
        Try matching at multiple extent scales and return the best result.

        The reference basemap is rendered from a geographic extent, so the
        meaningful search dimension is the extent estimate itself (the user's
        or AI's scale guess may be off), not the tile zoom level. Each step
        corresponds to roughly half a "zoom level" (a sqrt(2) scale factor).

        Args:
            image_src: Source (ungeoreferenced) image [H, W, 3]
            center_lat: Center latitude
            center_lon: Center longitude
            extent_meters: Estimated extent in meters
            extent_dimension: 'horizontal' or 'vertical'
            tile_fetcher: TileFetcher instance
            basemap_source: Basemap source name
            base_zoom: Base zoom level used for tile rendering detail
            zoom_range: Number of scale steps to try on each side of the estimate

        Returns:
            Tuple of (best_match_result, best_ref_image, best_extent, best_zoom)
        """
        h, w = image_src.shape[:2]
        source_aspect_ratio = w / h

        # Scale factors around the extent estimate, e.g. range=2 ->
        # [0.5, 0.71, 1.0, 1.41, 2.0]. The estimate itself is tried first
        # so the early exit can skip the alternatives entirely.
        offsets = sorted(range(-zoom_range, zoom_range + 1), key=abs)
        scale_factors = [math.sqrt(2) ** offset for offset in offsets]

        logger.info(
            f"Matching against '{basemap_source}' around "
            f"({center_lat:.5f}, {center_lon:.5f}), extent ~{extent_meters:.0f}m, "
            f"trying scale factors {[f'{s:.2f}' for s in scale_factors]}"
        )

        best_result = None
        best_ref_image = None
        best_extent = None
        best_zoom = base_zoom
        best_quality_score = -1.0

        for factor in scale_factors:
            candidate_extent = extent_meters * factor
            candidate_zoom = tile_fetcher.calculate_optimal_zoom(candidate_extent, 832)

            try:
                ref_image, ref_extent = tile_fetcher.fetch_tiles_from_center(
                    center_lat=center_lat,
                    center_lon=center_lon,
                    extent_meters=candidate_extent,
                    extent_dimension=extent_dimension,
                    source_aspect_ratio=source_aspect_ratio,
                    source_name=basemap_source,
                    zoom_level=candidate_zoom,
                    target_size=832
                )

                match_result = self.match_single_scale(image_src, ref_image, 832)
                quality_score = match_result.quality_score()

                logger.info(
                    f"  scale {factor:.2f} ({candidate_extent:.0f}m): "
                    f"{match_result.num_matches()} matches, "
                    f"conf {match_result.mean_confidence():.3f}, "
                    f"geom {match_result.geometric_score:.3f}, "
                    f"quality {quality_score:.3f}"
                )

                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                    best_result = match_result
                    best_ref_image = ref_image
                    best_extent = ref_extent
                    best_zoom = candidate_zoom

                if quality_score >= self.EARLY_EXIT_QUALITY:
                    logger.info(f"  Early exit: quality {quality_score:.3f} is sufficient")
                    break

            except Exception as e:
                logger.warning(f"  scale {factor:.2f}: matching failed: {e}")
                continue

        if best_result is None:
            raise RuntimeError("Failed to match at any extent scale")

        logger.info(
            f"Best result: quality {best_quality_score:.3f}, "
            f"{best_result.num_matches()} matches, "
            f"mean confidence {best_result.mean_confidence():.3f}"
        )

        return best_result, best_ref_image, best_extent, best_zoom

    def match_single_scale(
        self,
        image_src: np.ndarray,
        image_ref: np.ndarray,
        size: int
    ) -> MatchResult:
        """Single-scale matching.

        Args:
            image_src: Source image [H, W, 3]
            image_ref: Reference image [H, W, 3]
            size: Target size for matching

        Returns:
            MatchResult with matches
        """
        start_time = time.time()

        logger.debug(
            f"match_single_scale: src {image_src.shape} "
            f"[{image_src.min()}, {image_src.max()}], "
            f"ref {image_ref.shape} [{image_ref.min()}, {image_ref.max()}]"
        )

        if self.debug_dir is not None:
            self._save_debug_images(image_src, image_ref)

        # Resize images to target size
        img_src_resized, scale_src = self.model.resize_image(image_src, size)
        img_ref_resized, scale_ref = self.model.resize_image(image_ref, size)

        # Run MatchAnything inference
        keypoints_src_scaled, keypoints_ref_scaled, confidence = self.model.match_images(
            img_src_resized,
            img_ref_resized,
            max_keypoints=self.config['num_keypoints']
        )

        # Scale keypoints back to original image coordinates
        keypoints_src = self.model.scale_keypoints(keypoints_src_scaled, 1.0 / scale_src)
        keypoints_ref = self.model.scale_keypoints(keypoints_ref_scaled, 1.0 / scale_ref)

        # Estimate homography and get geometric consistency
        H, inlier_mask = self.model.estimate_homography(
            keypoints_src,
            keypoints_ref,
            confidence
        )

        if H is not None and len(inlier_mask) > 0:
            geometric_score = float(np.sum(inlier_mask)) / len(inlier_mask)
        else:
            geometric_score = 0.0

        distribution_quality = self.estimate_spatial_distribution_quality(keypoints_src)

        processing_time = time.time() - start_time

        return MatchResult(
            keypoints_src=keypoints_src,
            keypoints_ref=keypoints_ref,
            confidence=confidence,
            geometric_score=geometric_score,
            distribution_quality=distribution_quality,
            transform_matrix=H,
            processing_time=processing_time,
            device=self.model.device,
            resolution=size
        )

    def _save_debug_images(self, image_src: np.ndarray, image_ref: np.ndarray):
        """Save the images being matched to the debug directory for inspection."""
        try:
            from PIL import Image as PILImage
            debug_dir = Path(self.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)

            PILImage.fromarray(image_src.astype('uint8')).save(debug_dir / "source_image.png")
            PILImage.fromarray(image_ref.astype('uint8')).save(debug_dir / "reference_tiles.png")
            logger.info(f"Saved debug images to {debug_dir}")

            if image_ref.max() < 100:
                logger.warning(
                    f"Reference image is very dark (max value {image_ref.max()}); "
                    "tiles may not have loaded correctly"
                )
        except Exception as e:
            logger.warning(f"Failed to save debug images: {e}")

    def filter_matches(
        self,
        matches: MatchResult,
        confidence_threshold: float = 0.7,
        geometric_threshold: float = 0.8,
        min_gcps: int = 6
    ) -> MatchResult:
        """Filter matches based on quality criteria.

        Args:
            matches: Input MatchResult
            confidence_threshold: Minimum confidence score
            geometric_threshold: Minimum geometric consistency (informational)
            min_gcps: Minimum number of GCPs required (informational)

        Returns:
            Filtered MatchResult
        """
        if matches.num_matches() == 0:
            return matches

        logger.debug(
            f"filter_matches: {len(matches.confidence)} candidates, confidence "
            f"min/mean/max = {matches.confidence.min():.3f}/"
            f"{matches.confidence.mean():.3f}/{matches.confidence.max():.3f}, "
            f"threshold {confidence_threshold}"
        )

        # Keep confident matches
        confidence_mask = matches.confidence >= confidence_threshold

        keypoints_src = matches.keypoints_src[confidence_mask]
        keypoints_ref = matches.keypoints_ref[confidence_mask]
        confidence = matches.confidence[confidence_mask]

        # Remove geometric outliers from the confident subset via RANSAC
        H = None
        geometric_score = 0.0
        if len(keypoints_src) >= 4:
            H, inlier_mask = self.model.estimate_homography(
                keypoints_src, keypoints_ref, confidence
            )
            if H is not None and inlier_mask.any():
                geometric_score = float(np.sum(inlier_mask)) / len(inlier_mask)
                keypoints_src = keypoints_src[inlier_mask]
                keypoints_ref = keypoints_ref[inlier_mask]
                confidence = confidence[inlier_mask]

        distribution_quality = self.estimate_spatial_distribution_quality(keypoints_src)

        logger.debug(
            f"filter_matches: kept {len(keypoints_src)} matches "
            f"(geometric score {geometric_score:.3f})"
        )

        return MatchResult(
            keypoints_src=keypoints_src,
            keypoints_ref=keypoints_ref,
            confidence=confidence,
            geometric_score=geometric_score,
            distribution_quality=distribution_quality,
            transform_matrix=H,
            processing_time=matches.processing_time,
            device=matches.device,
            resolution=matches.resolution
        )

    def estimate_spatial_distribution_quality(self, keypoints: np.ndarray) -> float:
        """Score how well-distributed keypoints are (0-1).

        Args:
            keypoints: [N, 2] keypoint coordinates

        Returns:
            Quality score (1.0 = perfectly distributed, 0.0 = all clustered)
        """
        if len(keypoints) < 4:
            return 0.0

        min_x, min_y = keypoints.min(axis=0)
        max_x, max_y = keypoints.max(axis=0)

        # All points clustered in a tiny area
        if (max_x - min_x) < 10 or (max_y - min_y) < 10:
            return 0.0

        # Count points in a 3x3 grid over the keypoint bounding box
        grid_size = 3
        x_idx = np.clip(
            ((keypoints[:, 0] - min_x) / (max_x - min_x) * grid_size).astype(int),
            0, grid_size - 1
        )
        y_idx = np.clip(
            ((keypoints[:, 1] - min_y) / (max_y - min_y) * grid_size).astype(int),
            0, grid_size - 1
        )
        grid = np.zeros((grid_size, grid_size))
        np.add.at(grid, (y_idx, x_idx), 1)

        # Coverage: fraction of grid cells containing at least one point
        coverage = np.sum(grid > 0) / (grid_size * grid_size)

        # Uniformity: penalize high variance in per-cell counts
        expected_per_cell = len(keypoints) / (grid_size * grid_size)
        variance = np.var(grid)
        uniformity = 1.0 / (1.0 + variance / (expected_per_cell + 1e-6))

        return float(0.6 * coverage + 0.4 * uniformity)

    def _empty_result(self) -> MatchResult:
        """Create empty MatchResult"""
        return MatchResult(
            keypoints_src=np.zeros((0, 2)),
            keypoints_ref=np.zeros((0, 2)),
            confidence=np.zeros(0),
            geometric_score=0.0,
            distribution_quality=0.0,
            transform_matrix=None,
            processing_time=0.0,
            device=self.config['device'],
            resolution=self.config['size']
        )

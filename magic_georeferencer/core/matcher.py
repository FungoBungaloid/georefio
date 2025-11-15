"""
Matcher module for Magic Georeferencer

Handles progressive multi-scale image matching using MatchAnything model.
"""

import time
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


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


class Matcher:
    """Handle image matching with progressive refinement"""

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

        Process:
        1. Start at lowest resolution
        2. Get initial matches
        3. Filter by confidence threshold
        4. If > min_gcps found, proceed to next scale
        5. Use previous matches to refine search space
        6. Repeat until highest resolution or failure
        """
        if scales is None:
            # Single-scale matching
            return self.match_single_scale(image_src, image_ref, self.config['size'])

        best_result = None

        for i, size in enumerate(scales):
            result = self.match_single_scale(image_src, image_ref, size)

            # Filter matches
            result = self.filter_matches(
                result,
                confidence_threshold=0.7,
                min_gcps=min_gcps
            )

            if result.num_matches() >= min_gcps:
                best_result = result

                # If not the last scale, continue refining
                if i < len(scales) - 1:
                    continue
                else:
                    # Final scale reached
                    break
            else:
                # Not enough matches, stop progressive refinement
                break

        if best_result is None:
            # Return empty result
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
        Try matching at multiple zoom levels and return the best result.

        Args:
            image_src: Source (ungeoreferenced) image [H, W, 3]
            center_lat: Center latitude
            center_lon: Center longitude
            extent_meters: Extent in meters
            extent_dimension: 'horizontal' or 'vertical'
            tile_fetcher: TileFetcher instance
            basemap_source: Basemap source name
            base_zoom: Base zoom level (will try base_zoom-zoom_range to base_zoom+zoom_range)
            zoom_range: Number of zoom levels to try on each side

        Returns:
            Tuple of (best_match_result, best_ref_image, best_extent, best_zoom)
        """
        # Calculate source aspect ratio
        h, w = image_src.shape[:2]
        source_aspect_ratio = w / h

        # Try different zoom levels
        zoom_levels = range(
            max(0, base_zoom - zoom_range),
            min(20, base_zoom + zoom_range + 1)
        )

        print(f"\nTrying matching at zoom levels: {list(zoom_levels)}")

        best_result = None
        best_ref_image = None
        best_extent = None
        best_zoom = base_zoom
        best_quality_score = -1

        for zoom in zoom_levels:
            print(f"\n{'='*60}")
            print(f"Trying zoom level {zoom}")
            print(f"{'='*60}")

            try:
                # Fetch tiles at this zoom level
                ref_image, ref_extent = tile_fetcher.fetch_tiles_from_center(
                    center_lat=center_lat,
                    center_lon=center_lon,
                    extent_meters=extent_meters,
                    extent_dimension=extent_dimension,
                    source_aspect_ratio=source_aspect_ratio,
                    source_name=basemap_source,
                    zoom_level=zoom,
                    target_size=832
                )

                print(f"Fetched reference image: {ref_image.shape}")

                # Run matching
                match_result = self.match_single_scale(image_src, ref_image, 832)

                # Calculate quality score
                # Combine multiple factors:
                # - Number of matches (more is better)
                # - Mean confidence (higher is better)
                # - Geometric score (higher is better)
                # - Distribution quality (higher is better)

                num_matches = match_result.num_matches()
                mean_conf = match_result.mean_confidence()
                geom_score = match_result.geometric_score
                dist_quality = match_result.distribution_quality

                # Weighted quality score
                quality_score = (
                    (num_matches / 100.0) * 0.25 +  # Normalize num matches (assume 100 is very good)
                    mean_conf * 0.35 +               # Confidence is important
                    geom_score * 0.25 +              # Geometric consistency is important
                    dist_quality * 0.15              # Distribution quality
                )

                print(f"Zoom {zoom} results:")
                print(f"  - Matches: {num_matches}")
                print(f"  - Mean confidence: {mean_conf:.3f}")
                print(f"  - Geometric score: {geom_score:.3f}")
                print(f"  - Distribution quality: {dist_quality:.3f}")
                print(f"  - Overall quality score: {quality_score:.3f}")

                # Keep the best result
                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                    best_result = match_result
                    best_ref_image = ref_image
                    best_extent = ref_extent
                    best_zoom = zoom
                    print(f"  ★ New best zoom level!")

            except Exception as e:
                print(f"Failed to match at zoom {zoom}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if best_result is None:
            raise RuntimeError("Failed to match at any zoom level")

        print(f"\n{'='*60}")
        print(f"BEST RESULT: Zoom level {best_zoom}")
        print(f"Quality score: {best_quality_score:.3f}")
        print(f"Matches: {best_result.num_matches()}")
        print(f"Mean confidence: {best_result.mean_confidence():.3f}")
        print(f"{'='*60}\n")

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

        # Debug: Check input images
        print(f"Debug match_single_scale:")
        print(f"  image_src shape: {image_src.shape}, dtype: {image_src.dtype}, range: [{image_src.min()}, {image_src.max()}]")
        print(f"  image_ref shape: {image_ref.shape}, dtype: {image_ref.dtype}, range: [{image_ref.min()}, {image_ref.max()}]")

        # Resize images to target size
        img_src_resized, scale_src = self.model.resize_image(image_src, size)
        img_ref_resized, scale_ref = self.model.resize_image(image_ref, size)

        # Debug: Check resized images
        print(f"  After resize:")
        print(f"  img_src_resized shape: {img_src_resized.shape}, dtype: {img_src_resized.dtype}, range: [{img_src_resized.min()}, {img_src_resized.max()}]")
        print(f"  img_ref_resized shape: {img_ref_resized.shape}, dtype: {img_ref_resized.dtype}, range: [{img_ref_resized.min()}, {img_ref_resized.max()}]")

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

        # Calculate geometric score
        if H is not None:
            geometric_score = np.sum(inlier_mask) / len(inlier_mask) if len(inlier_mask) > 0 else 0.0
        else:
            geometric_score = 0.0
            inlier_mask = np.zeros(len(keypoints_src), dtype=bool)

        # Calculate spatial distribution quality
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
            geometric_threshold: Minimum geometric consistency (not used for filtering, just info)
            min_gcps: Minimum number of GCPs required

        Returns:
            Filtered MatchResult
        """
        if matches.num_matches() == 0:
            return matches

        # Debug: Print confidence statistics
        print(f"Debug: Confidence values before filtering:")
        print(f"  - Count: {len(matches.confidence)}")
        print(f"  - Min: {matches.confidence.min():.4f}")
        print(f"  - Max: {matches.confidence.max():.4f}")
        print(f"  - Mean: {matches.confidence.mean():.4f}")
        print(f"  - Median: {np.median(matches.confidence):.4f}")
        print(f"  - Threshold: {confidence_threshold}")
        print(f"  - Sample values: {matches.confidence[:10]}")

        # Filter by confidence threshold
        confidence_mask = matches.confidence >= confidence_threshold
        print(f"Debug: Matches passing confidence threshold: {confidence_mask.sum()} / {len(confidence_mask)}")

        # If we have a homography, also filter by geometric consistency
        if matches.transform_matrix is not None:
            _, inlier_mask = self.model.estimate_homography(
                matches.keypoints_src,
                matches.keypoints_ref,
                matches.confidence
            )
            combined_mask = confidence_mask & inlier_mask
        else:
            combined_mask = confidence_mask

        # Apply filtering
        keypoints_src_filtered = matches.keypoints_src[combined_mask]
        keypoints_ref_filtered = matches.keypoints_ref[combined_mask]
        confidence_filtered = matches.confidence[combined_mask]

        # Recalculate metrics
        if len(keypoints_src_filtered) >= 4:
            H, inlier_mask = self.model.estimate_homography(
                keypoints_src_filtered,
                keypoints_ref_filtered,
                confidence_filtered
            )
            geometric_score = np.sum(inlier_mask) / len(inlier_mask) if len(inlier_mask) > 0 else 0.0
        else:
            H = None
            geometric_score = 0.0

        distribution_quality = self.estimate_spatial_distribution_quality(keypoints_src_filtered)

        return MatchResult(
            keypoints_src=keypoints_src_filtered,
            keypoints_ref=keypoints_ref_filtered,
            confidence=confidence_filtered,
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

        # Calculate bounding box
        min_x, min_y = keypoints.min(axis=0)
        max_x, max_y = keypoints.max(axis=0)

        # Check if all points are clustered
        if (max_x - min_x) < 10 or (max_y - min_y) < 10:
            return 0.0

        # Divide image into 3x3 grid
        grid_size = 3
        x_bins = np.linspace(min_x, max_x, grid_size + 1)
        y_bins = np.linspace(min_y, max_y, grid_size + 1)

        # Count points in each grid cell
        grid = np.zeros((grid_size, grid_size))
        for kp in keypoints:
            x_idx = np.searchsorted(x_bins[1:], kp[0])
            y_idx = np.searchsorted(y_bins[1:], kp[1])
            x_idx = min(x_idx, grid_size - 1)
            y_idx = min(y_idx, grid_size - 1)
            grid[y_idx, x_idx] += 1

        # Calculate coverage (how many grid cells have at least one point)
        coverage = np.sum(grid > 0) / (grid_size * grid_size)

        # Calculate uniformity (variance in point distribution)
        if len(keypoints) > 0:
            expected_per_cell = len(keypoints) / (grid_size * grid_size)
            variance = np.var(grid)
            uniformity = 1.0 / (1.0 + variance / (expected_per_cell + 1e-6))
        else:
            uniformity = 0.0

        # Combined score
        quality = 0.6 * coverage + 0.4 * uniformity

        return float(quality)

    def _empty_result(self) -> MatchResult:
        """Create empty MatchResult"""
        return MatchResult(
            keypoints_src=np.array([]).reshape(0, 2),
            keypoints_ref=np.array([]).reshape(0, 2),
            confidence=np.array([]),
            geometric_score=0.0,
            distribution_quality=0.0,
            transform_matrix=None,
            processing_time=0.0,
            device=self.config['device'],
            resolution=self.config['size']
        )

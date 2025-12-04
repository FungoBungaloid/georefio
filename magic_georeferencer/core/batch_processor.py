"""
Batch Processor module for Magic Georeferencer

Orchestrates batch georeferencing using vision AI for geographic location estimation.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Dict, Any
import traceback

import numpy as np
from PIL import Image

from .vision_api import (
    VisionAPIClient,
    APIProvider,
    BoundingBoxEstimate,
    VISION_MODELS,
    estimate_batch_cost,
)
from .matcher import Matcher, MatchResult
from .tile_fetcher import TileFetcher
from .gcp_generator import GCPGenerator
from .georeferencer import Georeferencer

try:
    from qgis.core import QgsCoordinateReferenceSystem
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False


class BatchItemStatus(Enum):
    """Status of a batch processing item."""
    PENDING = "pending"
    PROCESSING = "processing"
    ESTIMATING_LOCATION = "estimating_location"
    FETCHING_TILES = "fetching_tiles"
    MATCHING = "matching"
    GEOREFERENCING = "georeferencing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchItemResult:
    """Result for a single batch item."""
    image_path: Path
    status: BatchItemStatus
    output_path: Optional[Path] = None
    bounding_box: Optional[BoundingBoxEstimate] = None
    num_gcps: int = 0
    error_message: Optional[str] = None
    processing_time: float = 0.0

    # Detailed metrics
    match_confidence: float = 0.0
    geometric_score: float = 0.0
    distribution_quality: float = 0.0


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    # API settings
    api_provider: APIProvider = APIProvider.OPENAI
    api_key: str = ""
    model_key: str = "gpt-4o-mini"

    # Output settings
    output_directory: Optional[Path] = None  # None = same as input
    output_suffix: str = "_georef"

    # Matching settings
    basemap_source: str = "osm_standard"
    quality_preset: str = "balanced"
    confidence_threshold: float = 0.70
    min_gcps: int = 6

    # Georeferencing settings
    transform_type: str = "auto"
    resampling: str = "cubic"
    compression: str = "LZW"
    target_crs: str = "EPSG:3857"

    # Processing settings
    max_image_dimension: int = 4096  # Max dimension for processing
    zoom_range: int = 2  # Zoom levels to try around optimal


@dataclass
class BatchProgress:
    """Progress information for batch processing."""
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    current_item: int = 0
    current_item_name: str = ""
    current_step: str = ""
    estimated_cost: Tuple[float, float] = (0.0, 0.0)
    elapsed_time: float = 0.0


# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif', '.bmp', '.webp'}


def find_images_in_path(path: Path, recursive: bool = False) -> List[Path]:
    """
    Find all supported images in a path.

    Args:
        path: Path to a file or directory
        recursive: If True, search subdirectories

    Returns:
        List of image file paths
    """
    images = []

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(path)
    elif path.is_dir():
        pattern = '**/*' if recursive else '*'
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            images.extend(path.glob(f'{pattern}{ext}'))
            images.extend(path.glob(f'{pattern}{ext.upper()}'))

    # Remove duplicates and sort
    images = sorted(set(images))
    return images


def generate_output_path(
    input_path: Path,
    output_dir: Optional[Path],
    suffix: str
) -> Path:
    """
    Generate output path for a georeferenced image.

    Args:
        input_path: Path to input image
        output_dir: Output directory (None = same as input)
        suffix: Filename suffix (e.g., "_georef")

    Returns:
        Output file path
    """
    if output_dir is None:
        output_dir = input_path.parent

    # Generate output filename
    stem = input_path.stem
    output_name = f"{stem}{suffix}.tif"

    return output_dir / output_name


class BatchProcessor:
    """
    Orchestrates batch georeferencing using vision AI for location estimation.

    Workflow:
    1. User provides list of images or folder
    2. For each image:
       a. Send to vision API to estimate bounding box
       b. Fetch reference tiles for that area
       c. Run feature matching
       d. Generate GCPs and georeference
    3. Report results
    """

    def __init__(
        self,
        model_manager,
        iface=None,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize batch processor.

        Args:
            model_manager: ModelManager instance with loaded MatchAnything model
            iface: QGIS interface (optional)
            progress_callback: Callback for progress updates
            log_callback: Callback for log messages
        """
        self.model_manager = model_manager
        self.iface = iface
        self.progress_callback = progress_callback
        self.log_callback = log_callback

        # Initialize components
        self.tile_fetcher = TileFetcher()
        self.gcp_generator = GCPGenerator()
        self.georeferencer = Georeferencer(iface)
        self.matcher = None  # Created when processing starts

        # State
        self._cancelled = False
        self._results: List[BatchItemResult] = []

    def log(self, message: str):
        """Log a message."""
        if self.log_callback:
            self.log_callback(message)
        print(message)

    def update_progress(self, progress: BatchProgress):
        """Update progress."""
        if self.progress_callback:
            self.progress_callback(progress)

    def cancel(self):
        """Cancel batch processing."""
        self._cancelled = True
        self.log("Cancellation requested...")

    def is_cancelled(self) -> bool:
        """Check if processing was cancelled."""
        return self._cancelled

    def process_batch(
        self,
        image_paths: List[Path],
        config: BatchConfig
    ) -> List[BatchItemResult]:
        """
        Process a batch of images.

        Args:
            image_paths: List of image paths to process
            config: Batch configuration

        Returns:
            List of BatchItemResult for each image
        """
        self._cancelled = False
        self._results = []
        start_time = time.time()

        # Initialize progress
        progress = BatchProgress(
            total_items=len(image_paths),
            estimated_cost=estimate_batch_cost(len(image_paths), config.model_key)
        )
        self.update_progress(progress)

        self.log(f"Starting batch processing of {len(image_paths)} images")
        self.log(f"Estimated API cost: ${progress.estimated_cost[0]:.2f} - ${progress.estimated_cost[1]:.2f}")

        # Initialize vision API client
        try:
            vision_client = VisionAPIClient(
                provider=config.api_provider,
                api_key=config.api_key,
                model_key=config.model_key
            )
            self.log(f"Vision API: {VISION_MODELS[config.model_key].display_name}")
        except Exception as e:
            self.log(f"Failed to initialize vision API: {e}")
            # Return all items as failed
            for path in image_paths:
                self._results.append(BatchItemResult(
                    image_path=path,
                    status=BatchItemStatus.FAILED,
                    error_message=f"Vision API initialization failed: {e}"
                ))
            return self._results

        # Initialize matcher (if model is loaded)
        try:
            if self.model_manager.model is not None:
                self.matcher = Matcher(self.model_manager)
            else:
                self.log("Loading MatchAnything model...")
                if self.model_manager.load_model():
                    self.matcher = Matcher(self.model_manager)
                else:
                    raise RuntimeError("Failed to load model")
        except Exception as e:
            self.log(f"Failed to initialize matcher: {e}")
            for path in image_paths:
                self._results.append(BatchItemResult(
                    image_path=path,
                    status=BatchItemStatus.FAILED,
                    error_message=f"Matcher initialization failed: {e}"
                ))
            return self._results

        # Get target CRS
        target_crs = QgsCoordinateReferenceSystem(config.target_crs)

        # Get confidence threshold from quality preset
        confidence_thresholds = {
            'strict': 0.85,
            'balanced': 0.70,
            'permissive': 0.55,
            'very_permissive': 0.20
        }
        confidence_threshold = confidence_thresholds.get(
            config.quality_preset,
            config.confidence_threshold
        )

        # Process each image
        for i, image_path in enumerate(image_paths):
            if self._cancelled:
                self.log("Processing cancelled by user")
                # Mark remaining items as skipped
                for remaining_path in image_paths[i:]:
                    self._results.append(BatchItemResult(
                        image_path=remaining_path,
                        status=BatchItemStatus.SKIPPED,
                        error_message="Cancelled by user"
                    ))
                break

            # Update progress
            progress.current_item = i + 1
            progress.current_item_name = image_path.name
            progress.current_step = "Starting..."
            progress.elapsed_time = time.time() - start_time
            self.update_progress(progress)

            self.log(f"\n{'='*60}")
            self.log(f"Processing [{i+1}/{len(image_paths)}]: {image_path.name}")
            self.log(f"{'='*60}")

            # Process single image
            result = self._process_single_image(
                image_path=image_path,
                config=config,
                vision_client=vision_client,
                target_crs=target_crs,
                confidence_threshold=confidence_threshold,
                progress=progress
            )

            self._results.append(result)

            # Update progress
            if result.status == BatchItemStatus.COMPLETED:
                progress.completed_items += 1
            else:
                progress.failed_items += 1

            self.update_progress(progress)

        # Final summary
        elapsed = time.time() - start_time
        self.log(f"\n{'='*60}")
        self.log(f"BATCH PROCESSING COMPLETE")
        self.log(f"{'='*60}")
        self.log(f"Total: {len(image_paths)}")
        self.log(f"Completed: {progress.completed_items}")
        self.log(f"Failed: {progress.failed_items}")
        self.log(f"Time: {elapsed:.1f} seconds")
        self.log(f"{'='*60}\n")

        return self._results

    def _process_single_image(
        self,
        image_path: Path,
        config: BatchConfig,
        vision_client: VisionAPIClient,
        target_crs: 'QgsCoordinateReferenceSystem',
        confidence_threshold: float,
        progress: BatchProgress
    ) -> BatchItemResult:
        """
        Process a single image.

        Args:
            image_path: Path to image
            config: Batch configuration
            vision_client: Vision API client
            target_crs: Target CRS for output
            confidence_threshold: Confidence threshold for matching
            progress: Progress object for updates

        Returns:
            BatchItemResult
        """
        item_start_time = time.time()

        result = BatchItemResult(
            image_path=image_path,
            status=BatchItemStatus.PROCESSING
        )

        try:
            # Step 1: Estimate geographic location using vision API
            progress.current_step = "Estimating location with AI..."
            self.update_progress(progress)
            result.status = BatchItemStatus.ESTIMATING_LOCATION

            self.log(f"  Step 1: Estimating location...")
            bbox = vision_client.estimate_bounding_box(
                image_path,
                progress_callback=lambda msg: self.log(f"    {msg}")
            )

            # Validate bounding box
            is_valid, validation_msg = bbox.is_valid()
            if not is_valid:
                raise ValueError(f"Invalid bounding box: {validation_msg}")

            result.bounding_box = bbox
            self.log(f"    Location: ({bbox.center_lat:.4f}, {bbox.center_lon:.4f})")
            self.log(f"    Extent: ~{bbox.to_extent_meters()/1000:.1f} km")
            self.log(f"    Reasoning: {bbox.reasoning[:100]}...")

            # Step 2: Load and prepare source image
            progress.current_step = "Loading source image..."
            self.update_progress(progress)

            self.log(f"  Step 2: Loading source image...")
            source_image, original_size, downsample_factor = self._load_and_prepare_image(
                image_path,
                config.max_image_dimension
            )
            self.log(f"    Original size: {original_size[0]}x{original_size[1]}")
            if downsample_factor > 1.0:
                self.log(f"    Downsampled by {downsample_factor:.2f}x for processing")

            # Calculate aspect ratio
            src_height, src_width = source_image.shape[:2]
            source_aspect_ratio = src_width / src_height

            # Step 3: Fetch reference tiles
            progress.current_step = "Fetching reference tiles..."
            self.update_progress(progress)
            result.status = BatchItemStatus.FETCHING_TILES

            self.log(f"  Step 3: Fetching reference tiles...")

            # Calculate extent in meters and optimal zoom
            extent_meters = bbox.to_extent_meters()
            base_zoom = self.tile_fetcher.calculate_optimal_zoom(extent_meters, 832)

            self.log(f"    Extent: {extent_meters:.0f}m, Base zoom: {base_zoom}")

            # Step 4: Run multi-zoom matching
            progress.current_step = "Matching images..."
            self.update_progress(progress)
            result.status = BatchItemStatus.MATCHING

            self.log(f"  Step 4: Running image matching...")

            match_result, ref_image, ref_extent, best_zoom = self.matcher.match_multi_zoom(
                image_src=source_image,
                center_lat=bbox.center_lat,
                center_lon=bbox.center_lon,
                extent_meters=extent_meters,
                extent_dimension='horizontal',  # Use horizontal as default
                tile_fetcher=self.tile_fetcher,
                basemap_source=config.basemap_source,
                base_zoom=base_zoom,
                zoom_range=config.zoom_range
            )

            # Filter matches by confidence
            match_result = self.matcher.filter_matches(
                match_result,
                confidence_threshold=confidence_threshold,
                min_gcps=config.min_gcps
            )

            num_matches = match_result.num_matches()
            self.log(f"    Found {num_matches} matches (confidence >= {confidence_threshold})")

            if num_matches < config.min_gcps:
                raise ValueError(
                    f"Insufficient matches: {num_matches} (minimum {config.min_gcps} required)"
                )

            # Store match metrics
            result.match_confidence = match_result.mean_confidence()
            result.geometric_score = match_result.geometric_score
            result.distribution_quality = match_result.distribution_quality

            # Step 5: Generate GCPs
            progress.current_step = "Generating control points..."
            self.update_progress(progress)

            self.log(f"  Step 5: Generating GCPs...")

            # Reference image CRS is EPSG:3857 (Web Mercator)
            ref_crs = QgsCoordinateReferenceSystem('EPSG:3857')

            gcps = self.gcp_generator.matches_to_gcps(
                match_result=match_result,
                ref_extent=ref_extent,
                ref_crs=ref_crs,
                ref_image_size=(ref_image.shape[1], ref_image.shape[0]),
                src_image_size=original_size,
                downsample_factor=downsample_factor
            )

            result.num_gcps = len(gcps)
            self.log(f"    Generated {len(gcps)} GCPs")

            # Step 6: Georeference
            progress.current_step = "Georeferencing..."
            self.update_progress(progress)
            result.status = BatchItemStatus.GEOREFERENCING

            self.log(f"  Step 6: Georeferencing...")

            # Generate output path
            output_path = generate_output_path(
                image_path,
                config.output_directory,
                config.output_suffix
            )

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine transform type
            if config.transform_type == 'auto':
                transform_type = self.georeferencer.suggest_transform_type(
                    len(gcps),
                    match_result.distribution_quality
                )
            else:
                transform_type = config.transform_type

            self.log(f"    Transform: {transform_type}")
            self.log(f"    Output: {output_path}")

            # Run georeferencing
            success, message = self.georeferencer.georeference_image(
                source_image_path=image_path,
                gcps=gcps,
                output_path=output_path,
                target_crs=target_crs,
                transform_type=transform_type,
                resampling=config.resampling,
                compression=config.compression,
                progress_callback=None  # Don't use nested progress
            )

            if not success:
                raise RuntimeError(f"Georeferencing failed: {message}")

            result.output_path = output_path
            result.status = BatchItemStatus.COMPLETED
            result.processing_time = time.time() - item_start_time

            self.log(f"  ✓ Completed in {result.processing_time:.1f}s")

        except Exception as e:
            result.status = BatchItemStatus.FAILED
            result.error_message = str(e)
            result.processing_time = time.time() - item_start_time

            self.log(f"  ✗ Failed: {e}")
            if self.log_callback:
                # Log full traceback for debugging
                self.log(f"    {traceback.format_exc()}")

        return result

    def _load_and_prepare_image(
        self,
        image_path: Path,
        max_dimension: int
    ) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        Load and prepare image for processing.

        Args:
            image_path: Path to image
            max_dimension: Maximum dimension for processing

        Returns:
            Tuple of (image_array, original_size, downsample_factor)
        """
        # Load image
        img = Image.open(image_path)

        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        elif img.mode == 'L':
            img = img.convert('RGB')

        original_size = (img.width, img.height)

        # Check if downsampling is needed
        max_dim = max(img.width, img.height)

        if max_dim > max_dimension:
            # Calculate downsample factor
            downsample_factor = max_dim / max_dimension
            new_width = int(img.width / downsample_factor)
            new_height = int(img.height / downsample_factor)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            downsample_factor = 1.0

        # Convert to numpy array
        image_array = np.array(img)

        return image_array, original_size, downsample_factor

    def get_results(self) -> List[BatchItemResult]:
        """Get processing results."""
        return self._results

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        total = len(self._results)
        completed = sum(1 for r in self._results if r.status == BatchItemStatus.COMPLETED)
        failed = sum(1 for r in self._results if r.status == BatchItemStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == BatchItemStatus.SKIPPED)

        total_time = sum(r.processing_time for r in self._results)
        avg_time = total_time / completed if completed > 0 else 0

        avg_gcps = sum(r.num_gcps for r in self._results if r.status == BatchItemStatus.COMPLETED)
        avg_gcps = avg_gcps / completed if completed > 0 else 0

        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': completed / total if total > 0 else 0,
            'total_time': total_time,
            'average_time': avg_time,
            'average_gcps': avg_gcps,
        }

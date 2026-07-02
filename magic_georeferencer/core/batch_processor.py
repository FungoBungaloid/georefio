"""
Batch Processor module for Magic Georeferencer

Orchestrates batch georeferencing. Each image's approximate location is
resolved through a chain of location hint providers (capture-path manifest,
embedded georeferencing, EXIF GPS, then optionally a vision AI model), and
the hint feeds the same match-and-refine pipeline as the interactive tool.
"""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Dict, Any
import traceback

import numpy as np
from PIL import Image

from .logging_utils import get_logger
from .vision_api import (
    VisionAPIClient,
    APIProvider,
    BoundingBoxEstimate,
    VISION_MODELS,
    estimate_batch_cost,
)
from .location_providers import build_hint_chain, CompositeHintProvider
from .matcher import Matcher
from .tile_fetcher import TileFetcher
from .gcp_generator import GCPGenerator
from .georeferencer import Georeferencer

try:
    from qgis.core import QgsCoordinateReferenceSystem
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False

logger = get_logger('batch')


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

    # Which location hint provider located this image (e.g. 'manifest',
    # 'exif_gps', 'vision_api')
    location_source: Optional[str] = None


# Quality presets in order from strictest to most permissive
QUALITY_PRESETS_ORDER = ['strict', 'balanced', 'permissive', 'very_permissive']

CONFIDENCE_THRESHOLDS = {
    'strict': 0.85,
    'balanced': 0.70,
    'permissive': 0.55,
    'very_permissive': 0.20
}


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    # Vision API settings (optional when metadata hints cover the batch)
    api_provider: APIProvider = APIProvider.OPENAI
    api_key: str = ""
    model_key: str = "gpt-5-nano-2025-08-07"
    use_vision_api: bool = True  # Allow vision AI as a location source

    # Location hint settings (see core/location_providers.py)
    use_metadata_hints: bool = True  # EXIF GPS, world files, GeoTIFF tags
    manifest_path: Optional[Path] = None  # Capture path / manifest (CSV, GeoJSON, GPX)
    default_hint_extent_m: float = 1000.0  # Footprint assumed for point hints
    world_file_crs: str = "EPSG:4326"  # CRS assumed for world files without .prj

    # Output settings
    output_directory: Optional[Path] = None  # None = same as input
    output_suffix: str = "_georef"

    # Automatic mode settings (default: fully automatic)
    auto_basemap: bool = True  # Use hint-recommended basemap per image
    auto_quality: bool = True  # Use progressive quality backoff

    # Manual matching settings (used when auto modes are disabled)
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
    zoom_range: int = 2  # Extent scale steps to try around the estimate
    auto_load_result: bool = False  # Don't auto-load into QGIS by default


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
        Sorted list of image file paths (deduplicated)
    """
    images = []

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(path)
    elif path.is_dir():
        pattern = '**/*' if recursive else '*'
        for candidate in path.glob(pattern):
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                images.append(candidate)

    return sorted(set(images))


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

    return output_dir / f"{input_path.stem}{suffix}.tif"


class BatchProcessor:
    """
    Orchestrates batch georeferencing.

    Workflow per image:
    1. Resolve an approximate location via the hint provider chain
       (manifest -> embedded georeferencing -> EXIF GPS -> vision AI)
    2. Fetch reference basemap tiles for that area
    3. Run feature matching (searching around the extent estimate)
    4. Filter matches, relaxing the confidence threshold if needed
    5. Generate GCPs and georeference
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
        self.georeferencer = Georeferencer(iface=None)  # loading handled per-config
        self.matcher = None  # Created when processing starts

        # State
        self._cancelled = False
        self._results: List[BatchItemResult] = []

    def log(self, message: str):
        """Log a message to the UI callback and the plugin logger."""
        if self.log_callback:
            self.log_callback(message)
        logger.info(message)

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

    def _fail_all(self, image_paths: List[Path], message: str) -> List[BatchItemResult]:
        """Mark every image as failed with the same message."""
        self.log(message)
        for path in image_paths:
            self._results.append(BatchItemResult(
                image_path=path,
                status=BatchItemStatus.FAILED,
                error_message=message
            ))
        return self._results

    def _build_hint_provider(self, config: BatchConfig) -> CompositeHintProvider:
        """Assemble the location hint chain from the batch configuration."""
        vision_client = None
        if config.use_vision_api and config.api_key:
            vision_client = VisionAPIClient(
                provider=config.api_provider,
                api_key=config.api_key,
                model_key=config.model_key
            )
            self.log(f"Vision API fallback: {VISION_MODELS[config.model_key].display_name}")
        else:
            self.log("Vision API disabled; using metadata/manifest hints only")

        return build_hint_chain(
            use_metadata_hints=config.use_metadata_hints,
            manifest_path=config.manifest_path,
            default_extent_m=config.default_hint_extent_m,
            world_file_crs=config.world_file_crs,
            vision_client=vision_client,
            vision_progress_callback=lambda msg: self.log(f"    {msg}"),
        )

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

        # Estimated cost is a worst case: metadata hints reduce it since only
        # images without hints reach the vision API
        estimated_cost = (0.0, 0.0)
        if config.use_vision_api and config.model_key in VISION_MODELS:
            estimated_cost = estimate_batch_cost(len(image_paths), config.model_key)

        progress = BatchProgress(
            total_items=len(image_paths),
            estimated_cost=estimated_cost
        )
        self.update_progress(progress)

        self.log(f"Starting batch processing of {len(image_paths)} images")
        if config.use_vision_api:
            self.log(
                f"Estimated API cost (worst case): "
                f"${estimated_cost[0]:.2f} - ${estimated_cost[1]:.2f}"
            )

        # Build the location hint provider chain
        try:
            hint_provider = self._build_hint_provider(config)
        except Exception as e:
            return self._fail_all(image_paths, f"Location hint setup failed: {e}")

        # Initialize matcher (loading the model if needed)
        try:
            if self.model_manager.model is None:
                self.log("Loading MatchAnything model...")
                success, message = self.model_manager.load_model()
                if not success:
                    raise RuntimeError(message)
            self.matcher = Matcher(self.model_manager)
        except Exception as e:
            return self._fail_all(image_paths, f"Matcher initialization failed: {e}")

        # Get target CRS
        target_crs = QgsCoordinateReferenceSystem(config.target_crs)

        # Log mode settings
        if config.manifest_path:
            self.log(f"Capture-path manifest: {config.manifest_path}")
        self.log(f"Metadata hints (EXIF/world files): {'ENABLED' if config.use_metadata_hints else 'DISABLED'}")
        if config.auto_basemap:
            self.log("Automatic basemap selection: ENABLED (per-image recommendation)")
        else:
            self.log(f"Automatic basemap selection: DISABLED (using {config.basemap_source})")
        if config.auto_quality:
            self.log(f"Progressive quality backoff: ENABLED ({' -> '.join(QUALITY_PRESETS_ORDER)})")
        else:
            self.log(f"Progressive quality backoff: DISABLED (using {config.quality_preset})")

        # Process each image
        for i, image_path in enumerate(image_paths):
            if self._cancelled:
                self.log("Processing cancelled by user")
                for remaining_path in image_paths[i:]:
                    self._results.append(BatchItemResult(
                        image_path=remaining_path,
                        status=BatchItemStatus.SKIPPED,
                        error_message="Cancelled by user"
                    ))
                break

            progress.current_item = i + 1
            progress.current_item_name = image_path.name
            progress.current_step = "Starting..."
            progress.elapsed_time = time.time() - start_time
            self.update_progress(progress)

            self.log(f"\n{'='*60}")
            self.log(f"Processing [{i+1}/{len(image_paths)}]: {image_path.name}")
            self.log(f"{'='*60}")

            result = self._process_single_image(
                image_path=image_path,
                config=config,
                hint_provider=hint_provider,
                target_crs=target_crs,
                progress=progress
            )

            self._results.append(result)

            if result.status == BatchItemStatus.COMPLETED:
                progress.completed_items += 1
            else:
                progress.failed_items += 1

            self.update_progress(progress)

        # Final summary
        elapsed = time.time() - start_time
        self.log(f"\n{'='*60}")
        self.log("BATCH PROCESSING COMPLETE")
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
        hint_provider: CompositeHintProvider,
        target_crs: 'QgsCoordinateReferenceSystem',
        progress: BatchProgress
    ) -> BatchItemResult:
        """
        Process a single image.

        Args:
            image_path: Path to image
            config: Batch configuration
            hint_provider: Location hint provider chain
            target_crs: Target CRS for output
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
            # Step 1: Resolve approximate location
            progress.current_step = "Resolving location..."
            self.update_progress(progress)
            result.status = BatchItemStatus.ESTIMATING_LOCATION

            self.log("  Step 1: Resolving location...")
            bbox = hint_provider.get_hint(image_path)

            if bbox is None:
                if config.use_vision_api:
                    raise ValueError("No location hint found for this image")
                raise ValueError(
                    "No location metadata found (no manifest entry, world file, "
                    "or EXIF GPS). Enable the vision API fallback or add this "
                    "image to a capture-path manifest."
                )

            is_valid, validation_msg = bbox.is_valid()
            if not is_valid:
                raise ValueError(f"Invalid bounding box: {validation_msg}")

            result.bounding_box = bbox
            result.location_source = bbox.source
            self.log(f"    Source: {bbox.source}")
            self.log(f"    Location: ({bbox.center_lat:.4f}, {bbox.center_lon:.4f})")
            self.log(f"    Extent: ~{bbox.to_extent_meters()/1000:.1f} km")
            self.log(f"    Reasoning: {bbox.reasoning[:120]}")

            # Determine which basemap to use
            if config.auto_basemap:
                basemap_source = bbox.recommended_basemap
            else:
                basemap_source = config.basemap_source
            self.log(f"    Basemap: {basemap_source}")

            # Step 2: Load and prepare source image
            progress.current_step = "Loading source image..."
            self.update_progress(progress)

            self.log("  Step 2: Loading source image...")
            source_image, original_size, downsample_factor = self._load_and_prepare_image(
                image_path,
                config.max_image_dimension
            )
            self.log(f"    Original size: {original_size[0]}x{original_size[1]}")
            if downsample_factor > 1.0:
                self.log(f"    Downsampled by {downsample_factor:.2f}x for processing")

            extent_meters = bbox.to_extent_meters()
            base_zoom = self.tile_fetcher.calculate_optimal_zoom(extent_meters, 832)

            # Step 3: Fetch tiles and run matching once. The matcher already
            # searches around the extent estimate and keeps the best result.
            progress.current_step = "Fetching tiles and matching..."
            self.update_progress(progress)
            result.status = BatchItemStatus.MATCHING

            self.log("  Step 3: Fetching tiles and matching...")
            self.log(f"    Extent: {extent_meters:.0f}m, base zoom: {base_zoom}")

            raw_match, ref_image, ref_extent, best_zoom = self.matcher.match_multi_zoom(
                image_src=source_image,
                center_lat=bbox.center_lat,
                center_lon=bbox.center_lon,
                extent_meters=extent_meters,
                extent_dimension='horizontal',
                tile_fetcher=self.tile_fetcher,
                basemap_source=basemap_source,
                base_zoom=base_zoom,
                zoom_range=config.zoom_range
            )

            # Step 4: Filter the raw matches, relaxing the confidence
            # threshold progressively. Matching itself doesn't depend on the
            # threshold, so only the (cheap) filtering step is repeated.
            progress.current_step = "Filtering matches..."
            self.update_progress(progress)

            if config.auto_quality:
                presets_to_try = QUALITY_PRESETS_ORDER
            else:
                presets_to_try = [config.quality_preset]

            match_result = None
            successful_preset = None

            for preset in presets_to_try:
                threshold = CONFIDENCE_THRESHOLDS[preset]
                filtered = self.matcher.filter_matches(
                    raw_match,
                    confidence_threshold=threshold,
                    min_gcps=config.min_gcps
                )
                num_matches = filtered.num_matches()
                self.log(f"    {preset} (threshold {threshold}): {num_matches} matches")

                if num_matches >= config.min_gcps:
                    match_result = filtered
                    successful_preset = preset
                    break

            if match_result is None:
                raise ValueError(
                    f"Could not find sufficient matches at any quality level "
                    f"(minimum {config.min_gcps} required; raw candidates: "
                    f"{raw_match.num_matches()})"
                )

            self.log(f"    Using {successful_preset} quality ({match_result.num_matches()} GCPs)")

            result.match_confidence = match_result.mean_confidence()
            result.geometric_score = match_result.geometric_score
            result.distribution_quality = match_result.distribution_quality

            # Step 5: Generate GCPs
            progress.current_step = "Generating control points..."
            self.update_progress(progress)

            self.log("  Step 5: Generating GCPs...")

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

            self.log("  Step 6: Georeferencing...")

            output_path = generate_output_path(
                image_path,
                config.output_directory,
                config.output_suffix
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if config.transform_type == 'auto':
                transform_type = self.georeferencer.suggest_transform_type(
                    len(gcps),
                    match_result.distribution_quality
                )
            else:
                transform_type = config.transform_type

            self.log(f"    Transform: {transform_type}")
            self.log(f"    Output: {output_path}")

            # Only load results into QGIS when explicitly requested
            self.georeferencer.iface = self.iface if config.auto_load_result else None

            success, message = self.georeferencer.georeference_image(
                source_image_path=image_path,
                gcps=gcps,
                output_path=output_path,
                target_crs=target_crs,
                transform_type=transform_type,
                resampling=config.resampling,
                compression=config.compression,
                progress_callback=None
            )

            if not success:
                raise RuntimeError(f"Georeferencing failed: {message}")

            result.output_path = output_path
            result.status = BatchItemStatus.COMPLETED
            result.processing_time = time.time() - item_start_time

            self.log(
                f"  Completed in {result.processing_time:.1f}s "
                f"(location: {bbox.source}, basemap: {basemap_source}, "
                f"quality: {successful_preset})"
            )

        except Exception as e:
            result.status = BatchItemStatus.FAILED
            result.error_message = str(e)
            result.processing_time = time.time() - item_start_time

            self.log(f"  Failed: {e}")
            logger.debug(traceback.format_exc())

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
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            original_size = (img.width, img.height)

            max_dim = max(img.width, img.height)
            if max_dim > max_dimension:
                downsample_factor = max_dim / max_dimension
                new_width = int(img.width / downsample_factor)
                new_height = int(img.height / downsample_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                downsample_factor = 1.0

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

        # How many images were located by each provider
        location_sources: Dict[str, int] = {}
        for r in self._results:
            if r.location_source:
                location_sources[r.location_source] = location_sources.get(r.location_source, 0) + 1

        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': completed / total if total > 0 else 0,
            'total_time': total_time,
            'average_time': avg_time,
            'average_gcps': avg_gcps,
            'location_sources': location_sources,
        }

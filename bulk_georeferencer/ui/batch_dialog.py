"""
Batch Processing Dialog for Bulk Georeferencer

Enhanced dialog for batch georeferencing with support for:
- Vision AI location estimation
- Pre-geocoded images (with loose positioning)
- Flight path/centroid shapefile input
- Customizable LLM prompts
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QSplitter,
    QWidget,
    QSizePolicy,
    QStyle,
    QApplication,
    QTabWidget,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QFrame
)
from qgis.PyQt.QtCore import Qt, QSettings, QThread, pyqtSignal
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY
)

# Import from magic_georeferencer
from magic_georeferencer.core.vision_api import (
    APIProvider,
    VISION_MODELS,
    MODELS_BY_PROVIDER,
    estimate_batch_cost,
    get_provider_display_name,
    VisionAPIClient,
    DEFAULT_SYSTEM_PROMPT
)
from magic_georeferencer.core.batch_processor import (
    BatchProcessor,
    BatchConfig,
    BatchProgress,
    BatchItemResult,
    BatchItemStatus,
    find_images_in_path,
    SUPPORTED_IMAGE_EXTENSIONS
)


class InputMode(Enum):
    """Input mode for batch processing."""
    FOLDER = "folder"
    FILES = "files"
    PRE_GEOCODED = "pre_geocoded"
    FLIGHT_PATH = "flight_path"


# Settings keys
SETTINGS_PREFIX = "BulkGeoreferencer"
SETTINGS_PROVIDER = f"{SETTINGS_PREFIX}/provider"
SETTINGS_MODEL = f"{SETTINGS_PREFIX}/model"
SETTINGS_REMEMBER_KEY = f"{SETTINGS_PREFIX}/remember_api_key"
SETTINGS_API_KEY_OPENAI = f"{SETTINGS_PREFIX}/api_key_openai"
SETTINGS_API_KEY_ANTHROPIC = f"{SETTINGS_PREFIX}/api_key_anthropic"
SETTINGS_OUTPUT_SUFFIX = f"{SETTINGS_PREFIX}/output_suffix"
SETTINGS_BASEMAP = f"{SETTINGS_PREFIX}/basemap"
SETTINGS_QUALITY = f"{SETTINGS_PREFIX}/quality"
SETTINGS_CUSTOM_PROMPT = f"{SETTINGS_PREFIX}/custom_prompt"
SETTINGS_USE_CUSTOM_PROMPT = f"{SETTINGS_PREFIX}/use_custom_prompt"
SETTINGS_BUFFER_METERS = f"{SETTINGS_PREFIX}/buffer_meters"


@dataclass
class PreGeocodedImage:
    """An image with rough location metadata."""
    path: Path
    center_lat: float
    center_lon: float
    extent_meters: float = 1000  # Default 1km extent


class BatchProcessingThread(QThread):
    """Thread for running batch processing."""

    progress_updated = pyqtSignal(object)  # BatchProgress
    log_message = pyqtSignal(str)
    finished_processing = pyqtSignal(list)  # List[BatchItemResult]

    def __init__(
        self,
        model_manager,
        image_paths: List[Path],
        config: BatchConfig,
        iface=None,
        pre_geocoded_images: Optional[List[PreGeocodedImage]] = None,
        custom_prompt: Optional[str] = None
    ):
        super().__init__()
        self.model_manager = model_manager
        self.image_paths = image_paths
        self.config = config
        self.iface = iface
        self.pre_geocoded_images = pre_geocoded_images
        self.custom_prompt = custom_prompt
        self.processor: Optional[BatchProcessor] = None

    def run(self):
        """Run batch processing in thread."""
        self.processor = BatchProcessor(
            model_manager=self.model_manager,
            iface=self.iface,
            progress_callback=lambda p: self.progress_updated.emit(p),
            log_callback=lambda m: self.log_message.emit(m)
        )

        # If we have pre-geocoded images, use a modified workflow
        if self.pre_geocoded_images:
            results = self._process_pre_geocoded()
        else:
            # Standard batch processing
            results = self.processor.process_batch(self.image_paths, self.config)

        self.finished_processing.emit(results)

    def _process_pre_geocoded(self) -> List[BatchItemResult]:
        """Process pre-geocoded images without AI location estimation."""
        from magic_georeferencer.core import Matcher, TileFetcher, GCPGenerator, Georeferencer
        from magic_georeferencer.core.batch_processor import generate_output_path
        import time
        import numpy as np
        from PIL import Image

        results = []
        tile_fetcher = TileFetcher()
        gcp_generator = GCPGenerator()
        georeferencer = Georeferencer(self.iface)

        # Initialize matcher
        if self.model_manager.model is None:
            self.log_message.emit("Loading MatchAnything model...")
            self.model_manager.load_model()

        matcher = Matcher(self.model_manager)

        target_crs = QgsCoordinateReferenceSystem(self.config.target_crs)

        for i, pre_geo in enumerate(self.pre_geocoded_images):
            if self.processor and self.processor.is_cancelled():
                break

            start_time = time.time()
            self.log_message.emit(f"\n{'='*60}")
            self.log_message.emit(f"Processing [{i+1}/{len(self.pre_geocoded_images)}]: {pre_geo.path.name}")
            self.log_message.emit(f"Pre-defined location: ({pre_geo.center_lat:.4f}, {pre_geo.center_lon:.4f})")
            self.log_message.emit(f"Extent: {pre_geo.extent_meters:.0f}m")
            self.log_message.emit(f"{'='*60}")

            progress = BatchProgress(
                total_items=len(self.pre_geocoded_images),
                current_item=i + 1,
                current_item_name=pre_geo.path.name,
                current_step="Loading image..."
            )
            self.progress_updated.emit(progress)

            result = BatchItemResult(
                image_path=pre_geo.path,
                status=BatchItemStatus.PROCESSING
            )

            try:
                # Load source image
                src_image_pil = Image.open(pre_geo.path).convert('RGB')
                original_size = src_image_pil.size

                # Downsample if needed
                MAX_DIMENSION = 4096
                downsample_factor = 1.0
                if max(original_size) > MAX_DIMENSION:
                    downsample_factor = max(original_size) / MAX_DIMENSION
                    new_width = int(original_size[0] / downsample_factor)
                    new_height = int(original_size[1] / downsample_factor)
                    src_image_pil = src_image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

                src_image = np.array(src_image_pil)

                # Calculate zoom level
                base_zoom = tile_fetcher.calculate_optimal_zoom(pre_geo.extent_meters, target_pixels=832)

                # Run multi-zoom matching
                progress.current_step = "Matching..."
                self.progress_updated.emit(progress)

                basemap_source = self.config.basemap_source

                match_result, ref_image, ref_extent, best_zoom = matcher.match_multi_zoom(
                    image_src=src_image,
                    center_lat=pre_geo.center_lat,
                    center_lon=pre_geo.center_lon,
                    extent_meters=pre_geo.extent_meters,
                    extent_dimension='horizontal',
                    tile_fetcher=tile_fetcher,
                    basemap_source=basemap_source,
                    base_zoom=base_zoom,
                    zoom_range=self.config.zoom_range
                )

                # Filter matches
                filtered_result = matcher.filter_matches(
                    match_result,
                    confidence_threshold=0.20,  # Permissive for pre-geocoded
                    min_gcps=self.config.min_gcps
                )

                if filtered_result.num_matches() < self.config.min_gcps:
                    raise ValueError(f"Only {filtered_result.num_matches()} matches found")

                # Generate GCPs
                progress.current_step = "Generating GCPs..."
                self.progress_updated.emit(progress)

                web_mercator_crs = QgsCoordinateReferenceSystem('EPSG:3857')
                gcps = gcp_generator.matches_to_gcps(
                    filtered_result,
                    ref_extent,
                    web_mercator_crs,
                    (ref_image.shape[1], ref_image.shape[0]),
                    original_size,
                    downsample_factor=downsample_factor
                )

                # Georeference
                progress.current_step = "Georeferencing..."
                self.progress_updated.emit(progress)

                output_path = generate_output_path(
                    pre_geo.path,
                    self.config.output_directory,
                    self.config.output_suffix
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                transform_type = georeferencer.suggest_transform_type(
                    len(gcps),
                    filtered_result.distribution_quality
                )

                success, message = georeferencer.georeference_image(
                    pre_geo.path,
                    gcps,
                    output_path,
                    target_crs,
                    transform_type=transform_type,
                    resampling=self.config.resampling,
                    compression=self.config.compression,
                    cloud_optimized=self.config.cloud_optimized
                )

                if success:
                    result.status = BatchItemStatus.COMPLETED
                    result.output_path = output_path
                    result.num_gcps = len(gcps)
                    result.match_confidence = filtered_result.mean_confidence()
                    self.log_message.emit(f"  ✓ Completed with {len(gcps)} GCPs")
                else:
                    raise RuntimeError(f"Georeferencing failed: {message}")

            except Exception as e:
                result.status = BatchItemStatus.FAILED
                result.error_message = str(e)
                self.log_message.emit(f"  ✗ Failed: {e}")

            result.processing_time = time.time() - start_time
            results.append(result)

        return results

    def cancel(self):
        """Cancel processing."""
        if self.processor:
            self.processor.cancel()


class BulkBatchDialog(QDialog):
    """Enhanced dialog for bulk georeferencing."""

    def __init__(self, iface, model_manager, parent=None):
        """Initialize batch dialog.

        Args:
            iface: QGIS interface
            model_manager: ModelManager instance
            parent: Parent widget
        """
        super().__init__(parent)

        self.iface = iface
        self.model_manager = model_manager
        self.settings = QSettings()

        # State
        self.image_paths: List[Path] = []
        self.pre_geocoded_images: List[PreGeocodedImage] = []
        self.processing_thread: Optional[BatchProcessingThread] = None
        self.is_processing = False
        self.input_mode = InputMode.FOLDER

        # Load tile sources from magic_georeferencer
        tile_config_path = Path(__file__).parent.parent.parent / 'magic_georeferencer' / 'config' / 'tile_sources.json'
        if tile_config_path.exists():
            with open(tile_config_path, 'r', encoding='utf-8') as f:
                self.tile_sources = json.load(f)
        else:
            # Fallback
            self.tile_sources = {
                "osm_standard": {"name": "OpenStreetMap Standard"},
                "esri_world_imagery": {"name": "ESRI World Imagery"}
            }

        # Setup UI
        self.setWindowTitle("Bulk Georeference")
        self.setMinimumWidth(800)
        self.setMinimumHeight(700)
        self._setup_ui()

        # Load saved settings
        self._load_settings()

        # Update initial state
        self._update_model_combo()
        self._update_cost_estimate()
        self._update_output_warning()

    def _setup_ui(self):
        """Setup user interface."""
        layout = QVBoxLayout()

        # Tab widget for different input modes
        self.tab_widget = QTabWidget()

        # Standard input tab
        standard_tab = QWidget()
        standard_layout = QVBoxLayout(standard_tab)
        standard_layout.addWidget(self._create_input_group())
        standard_layout.addWidget(self._create_api_group())
        standard_layout.addStretch()
        self.tab_widget.addTab(standard_tab, "Standard Input")

        # Pre-geocoded input tab
        pre_geo_tab = QWidget()
        pre_geo_layout = QVBoxLayout(pre_geo_tab)
        pre_geo_layout.addWidget(self._create_pre_geocoded_group())
        pre_geo_layout.addStretch()
        self.tab_widget.addTab(pre_geo_tab, "Pre-Geocoded Images")

        # Flight path input tab
        flight_tab = QWidget()
        flight_layout = QVBoxLayout(flight_tab)
        flight_layout.addWidget(self._create_flight_path_group())
        flight_layout.addStretch()
        self.tab_widget.addTab(flight_tab, "Flight Path / Centroids")

        layout.addWidget(self.tab_widget)

        # Matching section (common to all modes)
        layout.addWidget(self._create_matching_group())

        # Output section
        layout.addWidget(self._create_output_group())

        # Advanced section (collapsible)
        layout.addWidget(self._create_advanced_group())

        # Processing section
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._create_processing_group())
        layout.addWidget(splitter)

        # Action buttons
        layout.addLayout(self._create_action_buttons())

        self.setLayout(layout)

    def _create_input_group(self) -> QGroupBox:
        """Create standard input selection group."""
        group = QGroupBox("Input Images")
        layout = QVBoxLayout()

        # Info label
        info = QLabel(
            "Select a folder or individual image files. The AI will analyze each image "
            "to estimate its geographic location before georeferencing."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; margin-bottom: 8px;")
        layout.addWidget(info)

        # File/folder selection row
        select_layout = QHBoxLayout()

        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        self.input_path_edit.setPlaceholderText("Select folder or files...")
        select_layout.addWidget(self.input_path_edit)

        folder_btn = QPushButton("Select Folder...")
        folder_btn.clicked.connect(self._browse_folder)
        select_layout.addWidget(folder_btn)

        files_btn = QPushButton("Select Files...")
        files_btn.clicked.connect(self._browse_files)
        select_layout.addWidget(files_btn)

        layout.addLayout(select_layout)

        # Image count display
        self.image_count_label = QLabel("No images selected")
        layout.addWidget(self.image_count_label)

        group.setLayout(layout)
        return group

    def _create_pre_geocoded_group(self) -> QGroupBox:
        """Create pre-geocoded input group."""
        group = QGroupBox("Pre-Geocoded Images")
        layout = QVBoxLayout()

        # Info label
        info = QLabel(
            "Use this mode for images that already have rough location metadata (e.g., from EXIF GPS, "
            "world files, or manual positioning). The plugin will use these locations to fetch tiles "
            "instead of using AI location estimation.\n\n"
            "Supported sources:\n"
            "• EXIF GPS data in images\n"
            "• World files (.jgw, .pgw, .tfw) alongside images\n"
            "• Folder of images with a CSV file containing coordinates"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; margin-bottom: 8px;")
        layout.addWidget(info)

        # Folder selection
        folder_layout = QHBoxLayout()
        self.pre_geo_path_edit = QLineEdit()
        self.pre_geo_path_edit.setReadOnly(True)
        self.pre_geo_path_edit.setPlaceholderText("Select folder with pre-geocoded images...")
        folder_layout.addWidget(self.pre_geo_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_pre_geocoded_folder)
        folder_layout.addWidget(browse_btn)
        layout.addLayout(folder_layout)

        # Buffer/extent settings
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Search radius (meters):"))
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(100, 10000)
        self.buffer_spin.setValue(1000)
        self.buffer_spin.setSingleStep(100)
        self.buffer_spin.setToolTip(
            "Distance around each image center to search for matching features.\n"
            "Increase for very roughly positioned images."
        )
        buffer_layout.addWidget(self.buffer_spin)
        buffer_layout.addStretch()
        layout.addLayout(buffer_layout)

        # Status
        self.pre_geo_status_label = QLabel("No images loaded")
        layout.addWidget(self.pre_geo_status_label)

        group.setLayout(layout)
        return group

    def _create_flight_path_group(self) -> QGroupBox:
        """Create flight path input group."""
        group = QGroupBox("Flight Path / Centroids")
        layout = QVBoxLayout()

        # Info label
        info = QLabel(
            "Use this mode to georeference images using a flight path or centroid shapefile. "
            "Each feature in the shapefile should correspond to an image file.\n\n"
            "The shapefile should contain:\n"
            "• Point or line geometries representing image locations/paths\n"
            "• An attribute field with the image filename"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; margin-bottom: 8px;")
        layout.addWidget(info)

        # Image folder
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("Image folder:"))
        self.flight_img_path_edit = QLineEdit()
        self.flight_img_path_edit.setReadOnly(True)
        img_layout.addWidget(self.flight_img_path_edit)

        img_browse_btn = QPushButton("Browse...")
        img_browse_btn.clicked.connect(self._browse_flight_images)
        img_layout.addWidget(img_browse_btn)
        layout.addLayout(img_layout)

        # Shapefile selection
        shp_layout = QHBoxLayout()
        shp_layout.addWidget(QLabel("Shapefile:"))
        self.flight_shp_path_edit = QLineEdit()
        self.flight_shp_path_edit.setReadOnly(True)
        shp_layout.addWidget(self.flight_shp_path_edit)

        shp_browse_btn = QPushButton("Browse...")
        shp_browse_btn.clicked.connect(self._browse_flight_shapefile)
        shp_layout.addWidget(shp_browse_btn)
        layout.addLayout(shp_layout)

        # Filename field
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Filename field:"))
        self.filename_field_combo = QComboBox()
        self.filename_field_combo.setMinimumWidth(150)
        field_layout.addWidget(self.filename_field_combo)
        field_layout.addStretch()
        layout.addLayout(field_layout)

        # Buffer settings
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Search radius (meters):"))
        self.flight_buffer_spin = QSpinBox()
        self.flight_buffer_spin.setRange(100, 10000)
        self.flight_buffer_spin.setValue(500)
        self.flight_buffer_spin.setSingleStep(100)
        buffer_layout.addWidget(self.flight_buffer_spin)
        buffer_layout.addStretch()
        layout.addLayout(buffer_layout)

        # Status
        self.flight_status_label = QLabel("No shapefile loaded")
        layout.addWidget(self.flight_status_label)

        group.setLayout(layout)
        return group

    def _create_api_group(self) -> QGroupBox:
        """Create API configuration group."""
        group = QGroupBox("Vision API (for location estimation)")
        layout = QVBoxLayout()

        # Provider selection
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Provider:"))

        self.provider_combo = QComboBox()
        self.provider_combo.addItem("OpenAI", APIProvider.OPENAI)
        self.provider_combo.addItem("Anthropic", APIProvider.ANTHROPIC)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        provider_layout.addWidget(self.provider_combo)

        provider_layout.addStretch()
        layout.addLayout(provider_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._update_cost_estimate)
        model_layout.addWidget(self.model_combo)

        model_layout.addStretch()
        layout.addLayout(model_layout)

        # API key
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("API Key:"))

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Enter your API key...")
        key_layout.addWidget(self.api_key_edit)

        self.show_key_btn = QPushButton()
        self.show_key_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.show_key_btn.setToolTip("Show/Hide API key")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.clicked.connect(self._toggle_key_visibility)
        self.show_key_btn.setMaximumWidth(30)
        key_layout.addWidget(self.show_key_btn)

        layout.addLayout(key_layout)

        # Remember API key checkbox
        self.remember_key_checkbox = QCheckBox("Remember API key (stored locally)")
        self.remember_key_checkbox.setChecked(False)
        layout.addWidget(self.remember_key_checkbox)

        # Cost estimate
        self.cost_label = QLabel("Estimated cost: $0.00 - $0.00")
        self.cost_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.cost_label)

        group.setLayout(layout)
        return group

    def _create_matching_group(self) -> QGroupBox:
        """Create matching configuration group."""
        group = QGroupBox("Matching Settings")
        layout = QVBoxLayout()

        # Manual control checkbox
        self.manual_control_checkbox = QCheckBox("Enable manual control (override automatic settings)")
        self.manual_control_checkbox.setChecked(False)
        self.manual_control_checkbox.stateChanged.connect(self._on_manual_control_changed)
        layout.addWidget(self.manual_control_checkbox)

        # Basemap source
        basemap_layout = QHBoxLayout()
        self.basemap_label = QLabel("Basemap:")
        basemap_layout.addWidget(self.basemap_label)

        self.basemap_combo = QComboBox()
        for source_key, source_config in self.tile_sources.items():
            self.basemap_combo.addItem(source_config['name'], source_key)
        basemap_layout.addWidget(self.basemap_combo)
        basemap_layout.addStretch()
        layout.addLayout(basemap_layout)

        self.auto_basemap_label = QLabel("(AI will select best basemap per image)")
        self.auto_basemap_label.setStyleSheet("color: #080; font-style: italic;")
        layout.addWidget(self.auto_basemap_label)

        # Quality preset
        quality_layout = QHBoxLayout()
        self.quality_label = QLabel("Quality:")
        quality_layout.addWidget(self.quality_label)

        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Strict (0.85)", "strict")
        self.quality_combo.addItem("Balanced (0.70)", "balanced")
        self.quality_combo.addItem("Permissive (0.55)", "permissive")
        self.quality_combo.addItem("Very Permissive (0.20)", "very_permissive")
        self.quality_combo.setCurrentIndex(1)
        quality_layout.addWidget(self.quality_combo)
        quality_layout.addStretch()
        layout.addLayout(quality_layout)

        self.auto_quality_label = QLabel("(Will try strict first, then relax until match found)")
        self.auto_quality_label.setStyleSheet("color: #080; font-style: italic;")
        layout.addWidget(self.auto_quality_label)

        # Set initial state
        self._on_manual_control_changed()

        group.setLayout(layout)
        return group

    def _create_output_group(self) -> QGroupBox:
        """Create output configuration group."""
        group = QGroupBox("Output Settings")
        layout = QVBoxLayout()

        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output folder:"))

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Same as input (default)")
        self.output_dir_edit.textChanged.connect(self._update_output_warning)
        dir_layout.addWidget(self.output_dir_edit)

        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(browse_dir_btn)

        same_btn = QPushButton("Same as Input")
        same_btn.clicked.connect(lambda: self.output_dir_edit.clear())
        dir_layout.addWidget(same_btn)

        layout.addLayout(dir_layout)

        # Filename suffix
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Filename suffix:"))

        self.suffix_edit = QLineEdit()
        self.suffix_edit.setText("_georef")
        self.suffix_edit.textChanged.connect(self._update_output_warning)
        suffix_layout.addWidget(self.suffix_edit)

        layout.addLayout(suffix_layout)

        # Warning label
        self.output_warning_label = QLabel("")
        self.output_warning_label.setStyleSheet("color: red; font-weight: bold;")
        self.output_warning_label.setWordWrap(True)
        layout.addWidget(self.output_warning_label)

        group.setLayout(layout)
        return group

    def _create_advanced_group(self) -> QGroupBox:
        """Create advanced settings group (collapsible)."""
        group = QGroupBox("Advanced Settings")
        group.setCheckable(True)
        group.setChecked(False)  # Collapsed by default
        layout = QVBoxLayout()

        # LLM prompt customization
        prompt_label = QLabel("Location Extraction Prompt:")
        prompt_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(prompt_label)

        prompt_info = QLabel(
            "Customize the prompt sent to the Vision AI for extracting geographic location from images. "
            "This is used only for standard input mode (not pre-geocoded or flight path modes)."
        )
        prompt_info.setWordWrap(True)
        prompt_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(prompt_info)

        self.use_custom_prompt_checkbox = QCheckBox("Use custom prompt")
        self.use_custom_prompt_checkbox.setChecked(False)
        self.use_custom_prompt_checkbox.stateChanged.connect(self._on_custom_prompt_changed)
        layout.addWidget(self.use_custom_prompt_checkbox)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Enter custom prompt here...")
        self.prompt_edit.setMaximumHeight(150)
        self.prompt_edit.setEnabled(False)
        layout.addWidget(self.prompt_edit)

        # Reset to default button
        reset_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset to Default Prompt")
        reset_btn.clicked.connect(self._reset_prompt_to_default)
        reset_layout.addWidget(reset_btn)
        reset_layout.addStretch()
        layout.addLayout(reset_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # GCP export option
        self.export_gcps_checkbox = QCheckBox("Export control points (.points files) alongside georeferenced images")
        self.export_gcps_checkbox.setChecked(False)
        layout.addWidget(self.export_gcps_checkbox)

        group.setLayout(layout)
        return group

    def _create_processing_group(self) -> QWidget:
        """Create processing status group."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Current item label
        self.current_item_label = QLabel("Ready to process")
        layout.addWidget(self.current_item_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Progress detail
        self.progress_detail_label = QLabel("")
        layout.addWidget(self.progress_detail_label)

        # Log output
        layout.addWidget(QLabel("Processing Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.log_text)

        return widget

    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons."""
        layout = QHBoxLayout()

        # Help button
        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self._show_help)
        layout.addWidget(help_btn)

        layout.addStretch()

        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self._on_close)
        layout.addWidget(self.close_btn)

        # Cancel button (hidden initially)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setVisible(False)
        layout.addWidget(self.cancel_btn)

        # Start button
        self.start_btn = QPushButton("Start Batch Processing")
        self.start_btn.clicked.connect(self._start_processing)
        self.start_btn.setDefault(True)
        layout.addWidget(self.start_btn)

        return layout

    def _on_manual_control_changed(self):
        """Handle manual control checkbox state change."""
        manual_mode = self.manual_control_checkbox.isChecked()
        self.basemap_combo.setEnabled(manual_mode)
        self.basemap_label.setEnabled(manual_mode)
        self.quality_combo.setEnabled(manual_mode)
        self.quality_label.setEnabled(manual_mode)
        self.auto_basemap_label.setVisible(not manual_mode)
        self.auto_quality_label.setVisible(not manual_mode)

    def _on_custom_prompt_changed(self):
        """Handle custom prompt checkbox change."""
        use_custom = self.use_custom_prompt_checkbox.isChecked()
        self.prompt_edit.setEnabled(use_custom)

        if use_custom and not self.prompt_edit.toPlainText():
            self.prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT)

    def _reset_prompt_to_default(self):
        """Reset prompt to default."""
        self.prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT)

    def _on_provider_changed(self):
        """Handle provider selection change."""
        self._update_model_combo()

        if self.remember_key_checkbox.isChecked():
            provider = self.provider_combo.currentData()
            if provider == APIProvider.OPENAI:
                key = self.settings.value(SETTINGS_API_KEY_OPENAI, "")
            else:
                key = self.settings.value(SETTINGS_API_KEY_ANTHROPIC, "")
            self.api_key_edit.setText(key)

    def _update_model_combo(self):
        """Update model combo based on selected provider."""
        self.model_combo.clear()

        provider = self.provider_combo.currentData()
        model_keys = MODELS_BY_PROVIDER.get(provider, [])

        for key in model_keys:
            model = VISION_MODELS[key]
            self.model_combo.addItem(model.display_name, key)

    def _update_cost_estimate(self):
        """Update cost estimate display."""
        if len(self.image_paths) == 0:
            self.cost_label.setText("Estimated cost: $0.00")
            return

        model_key = self.model_combo.currentData()
        if not model_key:
            return

        try:
            low, high = estimate_batch_cost(len(self.image_paths), model_key)
            self.cost_label.setText(f"Estimated cost: ${low:.2f} - ${high:.2f}")
        except Exception:
            self.cost_label.setText("Estimated cost: Unknown")

    def _update_output_warning(self):
        """Update output warning based on settings."""
        suffix = self.suffix_edit.text()
        output_dir = self.output_dir_edit.text().strip()

        if not suffix and not output_dir:
            self.output_warning_label.setText("WARNING: Output will overwrite input files!")
        elif not suffix:
            self.output_warning_label.setText("WARNING: Files with same name will be overwritten!")
        else:
            self.output_warning_label.setText("")

    def _toggle_key_visibility(self):
        """Toggle API key visibility."""
        if self.show_key_btn.isChecked():
            self.api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_edit.setEchoMode(QLineEdit.Password)

    def _browse_folder(self):
        """Browse for input folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with Images", str(Path.home())
        )

        if folder:
            folder_path = Path(folder)
            self.input_path_edit.setText(str(folder_path))
            self.image_paths = find_images_in_path(folder_path)
            self.input_mode = InputMode.FOLDER
            self._update_image_count()

    def _browse_files(self):
        """Browse for input files."""
        extensions = " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", str(Path.home()),
            f"Images ({extensions});;All Files (*)"
        )

        if files:
            self.image_paths = [Path(f) for f in files]
            if len(files) == 1:
                self.input_path_edit.setText(str(self.image_paths[0]))
            else:
                self.input_path_edit.setText(f"{len(files)} files selected")
            self.input_mode = InputMode.FILES
            self._update_image_count()

    def _browse_pre_geocoded_folder(self):
        """Browse for pre-geocoded folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with Pre-Geocoded Images", str(Path.home())
        )

        if folder:
            folder_path = Path(folder)
            self.pre_geo_path_edit.setText(str(folder_path))
            self._load_pre_geocoded_images(folder_path)

    def _load_pre_geocoded_images(self, folder: Path):
        """Load pre-geocoded images from folder."""
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        self.pre_geocoded_images = []
        images = find_images_in_path(folder)
        buffer_meters = self.buffer_spin.value()

        loaded_count = 0
        for img_path in images:
            try:
                # Try to extract GPS from EXIF
                img = Image.open(img_path)
                exif_data = img._getexif()

                if exif_data:
                    gps_info = {}
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == 'GPSInfo':
                            for gps_tag_id, gps_value in value.items():
                                gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                                gps_info[gps_tag] = gps_value

                    if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                        lat = self._convert_gps_to_decimal(
                            gps_info['GPSLatitude'],
                            gps_info.get('GPSLatitudeRef', 'N')
                        )
                        lon = self._convert_gps_to_decimal(
                            gps_info['GPSLongitude'],
                            gps_info.get('GPSLongitudeRef', 'E')
                        )

                        self.pre_geocoded_images.append(PreGeocodedImage(
                            path=img_path,
                            center_lat=lat,
                            center_lon=lon,
                            extent_meters=buffer_meters
                        ))
                        loaded_count += 1

            except Exception as e:
                print(f"Error loading {img_path}: {e}")

        # Update status
        self.pre_geo_status_label.setText(
            f"Loaded {loaded_count} images with GPS data from {len(images)} total images"
        )
        self.input_mode = InputMode.PRE_GEOCODED

    def _convert_gps_to_decimal(self, gps_coords, ref):
        """Convert GPS coordinates from EXIF format to decimal."""
        d = float(gps_coords[0])
        m = float(gps_coords[1])
        s = float(gps_coords[2])

        decimal = d + (m / 60.0) + (s / 3600.0)

        if ref in ['S', 'W']:
            decimal = -decimal

        return decimal

    def _browse_flight_images(self):
        """Browse for flight images folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with Flight Images", str(Path.home())
        )
        if folder:
            self.flight_img_path_edit.setText(folder)

    def _browse_flight_shapefile(self):
        """Browse for flight shapefile."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Flight Path Shapefile", str(Path.home()),
            "Shapefiles (*.shp);;All Files (*)"
        )

        if filepath:
            self.flight_shp_path_edit.setText(filepath)
            self._load_shapefile_fields(filepath)

    def _load_shapefile_fields(self, filepath: str):
        """Load shapefile and populate field combo."""
        self.filename_field_combo.clear()

        try:
            layer = QgsVectorLayer(filepath, "temp", "ogr")
            if layer.isValid():
                fields = layer.fields()
                for field in fields:
                    self.filename_field_combo.addItem(field.name())

                feature_count = layer.featureCount()
                self.flight_status_label.setText(f"Loaded shapefile with {feature_count} features")
            else:
                self.flight_status_label.setText("Error: Invalid shapefile")
        except Exception as e:
            self.flight_status_label.setText(f"Error loading shapefile: {e}")

    def _update_image_count(self):
        """Update image count display."""
        count = len(self.image_paths)
        if count == 0:
            self.image_count_label.setText("No images selected")
        elif count == 1:
            self.image_count_label.setText("1 image selected")
        else:
            self.image_count_label.setText(f"{count} images selected")

        self._update_cost_estimate()

    def _browse_output_dir(self):
        """Browse for output directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(Path.home())
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def _load_settings(self):
        """Load saved settings."""
        # Provider
        provider_str = self.settings.value(SETTINGS_PROVIDER, "openai")
        provider_idx = 0 if provider_str == "openai" else 1
        self.provider_combo.setCurrentIndex(provider_idx)

        # Model
        saved_model = self.settings.value(SETTINGS_MODEL, "")

        # Remember API key
        remember = self.settings.value(SETTINGS_REMEMBER_KEY, False, type=bool)
        self.remember_key_checkbox.setChecked(remember)

        if remember:
            provider = self.provider_combo.currentData()
            if provider == APIProvider.OPENAI:
                key = self.settings.value(SETTINGS_API_KEY_OPENAI, "")
            else:
                key = self.settings.value(SETTINGS_API_KEY_ANTHROPIC, "")
            self.api_key_edit.setText(key)

        # Output suffix
        suffix = self.settings.value(SETTINGS_OUTPUT_SUFFIX, "_georef")
        self.suffix_edit.setText(suffix)

        # Basemap
        basemap = self.settings.value(SETTINGS_BASEMAP, "osm_standard")
        idx = self.basemap_combo.findData(basemap)
        if idx >= 0:
            self.basemap_combo.setCurrentIndex(idx)

        # Quality
        quality = self.settings.value(SETTINGS_QUALITY, "balanced")
        idx = self.quality_combo.findData(quality)
        if idx >= 0:
            self.quality_combo.setCurrentIndex(idx)

        # Custom prompt
        use_custom = self.settings.value(SETTINGS_USE_CUSTOM_PROMPT, False, type=bool)
        self.use_custom_prompt_checkbox.setChecked(use_custom)
        custom_prompt = self.settings.value(SETTINGS_CUSTOM_PROMPT, "")
        if custom_prompt:
            self.prompt_edit.setPlainText(custom_prompt)

        # Buffer
        buffer_val = self.settings.value(SETTINGS_BUFFER_METERS, 1000, type=int)
        self.buffer_spin.setValue(buffer_val)

        # Update model combo
        self._update_model_combo()
        if saved_model:
            idx = self.model_combo.findData(saved_model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

    def _save_settings(self):
        """Save current settings."""
        provider = self.provider_combo.currentData()
        self.settings.setValue(SETTINGS_PROVIDER, provider.value)

        model = self.model_combo.currentData()
        if model:
            self.settings.setValue(SETTINGS_MODEL, model)

        remember = self.remember_key_checkbox.isChecked()
        self.settings.setValue(SETTINGS_REMEMBER_KEY, remember)

        if remember:
            if provider == APIProvider.OPENAI:
                self.settings.setValue(SETTINGS_API_KEY_OPENAI, self.api_key_edit.text())
            else:
                self.settings.setValue(SETTINGS_API_KEY_ANTHROPIC, self.api_key_edit.text())
        else:
            self.settings.remove(SETTINGS_API_KEY_OPENAI)
            self.settings.remove(SETTINGS_API_KEY_ANTHROPIC)

        self.settings.setValue(SETTINGS_OUTPUT_SUFFIX, self.suffix_edit.text())
        self.settings.setValue(SETTINGS_BASEMAP, self.basemap_combo.currentData())
        self.settings.setValue(SETTINGS_QUALITY, self.quality_combo.currentData())

        # Custom prompt
        self.settings.setValue(SETTINGS_USE_CUSTOM_PROMPT, self.use_custom_prompt_checkbox.isChecked())
        self.settings.setValue(SETTINGS_CUSTOM_PROMPT, self.prompt_edit.toPlainText())
        self.settings.setValue(SETTINGS_BUFFER_METERS, self.buffer_spin.value())

    def _validate_inputs(self) -> tuple:
        """Validate inputs before processing."""
        current_tab = self.tab_widget.currentIndex()

        if current_tab == 0:  # Standard input
            if len(self.image_paths) == 0:
                return False, "Please select images to process"
            if not self.api_key_edit.text().strip():
                return False, "Please enter your API key"

        elif current_tab == 1:  # Pre-geocoded
            if len(self.pre_geocoded_images) == 0:
                return False, "No pre-geocoded images loaded"

        elif current_tab == 2:  # Flight path
            if not self.flight_img_path_edit.text():
                return False, "Please select the image folder"
            if not self.flight_shp_path_edit.text():
                return False, "Please select the flight path shapefile"

        if self.model_manager is None:
            return False, "Model manager not initialized"

        return True, ""

    def _start_processing(self):
        """Start batch processing."""
        is_valid, error_msg = self._validate_inputs()
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_msg)
            return

        self._save_settings()

        # Build config
        provider = self.provider_combo.currentData()
        model_key = self.model_combo.currentData()

        output_dir = self.output_dir_edit.text().strip()
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = None

        manual_mode = self.manual_control_checkbox.isChecked()

        config = BatchConfig(
            api_provider=provider,
            api_key=self.api_key_edit.text().strip(),
            model_key=model_key,
            output_directory=output_dir,
            output_suffix=self.suffix_edit.text(),
            auto_basemap=not manual_mode,
            auto_quality=not manual_mode,
            basemap_source=self.basemap_combo.currentData(),
            quality_preset=self.quality_combo.currentData(),
            auto_load_result=False,
            compression='DEFLATE',
            cloud_optimized=True
        )

        # Update UI
        self.is_processing = True
        self._update_ui_for_processing(True)
        self.log_text.clear()

        # Determine processing mode
        current_tab = self.tab_widget.currentIndex()
        custom_prompt = None
        if self.use_custom_prompt_checkbox.isChecked():
            custom_prompt = self.prompt_edit.toPlainText()

        pre_geo = None
        if current_tab == 1:  # Pre-geocoded
            pre_geo = self.pre_geocoded_images
            image_paths = [p.path for p in self.pre_geocoded_images]
        elif current_tab == 2:  # Flight path
            pre_geo, image_paths = self._build_flight_path_list()
            if not pre_geo:
                self.is_processing = False
                self._update_ui_for_processing(False)
                return
        else:
            image_paths = self.image_paths

        # Create thread
        self.processing_thread = BatchProcessingThread(
            self.model_manager,
            image_paths,
            config,
            self.iface,
            pre_geocoded_images=pre_geo,
            custom_prompt=custom_prompt
        )
        self.processing_thread.progress_updated.connect(self._on_progress_updated)
        self.processing_thread.log_message.connect(self._on_log_message)
        self.processing_thread.finished_processing.connect(self._on_processing_finished)
        self.processing_thread.start()

    def _build_flight_path_list(self) -> tuple:
        """Build list of pre-geocoded images from flight path shapefile."""
        try:
            shp_path = self.flight_shp_path_edit.text()
            img_folder = Path(self.flight_img_path_edit.text())
            field_name = self.filename_field_combo.currentText()
            buffer_meters = self.flight_buffer_spin.value()

            layer = QgsVectorLayer(shp_path, "flight", "ogr")
            if not layer.isValid():
                QMessageBox.critical(self, "Error", "Could not load shapefile")
                return None, []

            pre_geo = []
            image_paths = []

            for feature in layer.getFeatures():
                filename = feature[field_name]
                geom = feature.geometry()

                # Get centroid
                if geom.type() == 0:  # Point
                    centroid = geom.asPoint()
                else:
                    centroid = geom.centroid().asPoint()

                # Find matching image
                img_path = img_folder / filename
                if not img_path.exists():
                    # Try common extensions
                    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        test_path = img_folder / f"{Path(filename).stem}{ext}"
                        if test_path.exists():
                            img_path = test_path
                            break

                if img_path.exists():
                    pre_geo.append(PreGeocodedImage(
                        path=img_path,
                        center_lat=centroid.y(),
                        center_lon=centroid.x(),
                        extent_meters=buffer_meters
                    ))
                    image_paths.append(img_path)

            self.log_text.append(f"Matched {len(pre_geo)} images from flight path")
            return pre_geo, image_paths

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing flight path: {e}")
            return None, []

    def _on_cancel(self):
        """Handle cancel button click."""
        if self.processing_thread:
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")
            self.processing_thread.cancel()

    def _on_progress_updated(self, progress: BatchProgress):
        """Handle progress update."""
        if progress.total_items > 0:
            percentage = int((progress.current_item / progress.total_items) * 100)
            self.progress_bar.setValue(percentage)

            self.current_item_label.setText(
                f"Processing [{progress.current_item}/{progress.total_items}]: "
                f"{progress.current_item_name}"
            )

            self.progress_detail_label.setText(
                f"Step: {progress.current_step} | "
                f"Completed: {progress.completed_items} | "
                f"Failed: {progress.failed_items} | "
                f"Elapsed: {progress.elapsed_time:.0f}s"
            )

    def _on_log_message(self, message: str):
        """Handle log message."""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _on_processing_finished(self, results: List[BatchItemResult]):
        """Handle processing completion."""
        self.is_processing = False
        self._update_ui_for_processing(False)

        completed = sum(1 for r in results if r.status == BatchItemStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == BatchItemStatus.FAILED)
        skipped = sum(1 for r in results if r.status == BatchItemStatus.SKIPPED)

        summary = (
            f"Batch Processing Complete!\n\n"
            f"Total: {len(results)}\n"
            f"Completed: {completed}\n"
            f"Failed: {failed}\n"
            f"Skipped: {skipped}"
        )

        if failed > 0:
            summary += "\n\nFailed images:"
            for r in results:
                if r.status == BatchItemStatus.FAILED:
                    summary += f"\n- {r.image_path.name}: {r.error_message}"

        QMessageBox.information(self, "Batch Complete", summary)

    def _update_ui_for_processing(self, processing: bool):
        """Update UI state for processing."""
        self.tab_widget.setEnabled(not processing)
        self.manual_control_checkbox.setEnabled(not processing)
        self.output_dir_edit.setEnabled(not processing)
        self.suffix_edit.setEnabled(not processing)

        self.start_btn.setVisible(not processing)
        self.cancel_btn.setVisible(processing)
        self.cancel_btn.setEnabled(processing)
        self.cancel_btn.setText("Cancel")
        self.close_btn.setEnabled(not processing)

        if not processing:
            self.current_item_label.setText("Ready to process")
            self.progress_bar.setValue(0)

    def _on_close(self):
        """Handle close button."""
        if self.is_processing:
            result = QMessageBox.question(
                self, "Confirm Close",
                "Processing is still running. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if result != QMessageBox.Yes:
                return

            if self.processing_thread:
                self.processing_thread.cancel()
                self.processing_thread.wait(5000)

        self._save_settings()
        self.reject()

    def _show_help(self):
        """Show help dialog."""
        QMessageBox.information(
            self,
            "Bulk Georeference - Help",
            "Bulk Georeference - Multiple Input Modes\n\n"
            "STANDARD INPUT:\n"
            "Uses Vision AI to analyze each image and estimate its geographic location.\n"
            "Best for: Maps, aerial photos with visible landmarks.\n\n"
            "PRE-GEOCODED IMAGES:\n"
            "Uses existing location metadata (EXIF GPS, world files) to skip AI estimation.\n"
            "Best for: Drone imagery, GPS-tagged photos.\n\n"
            "FLIGHT PATH / CENTROIDS:\n"
            "Uses a shapefile with image locations/centroids.\n"
            "Best for: Flight surveys, systematic image capture.\n\n"
            "ADVANCED SETTINGS:\n"
            "- Customize the LLM prompt for location extraction\n"
            "- Export control points for manual refinement\n\n"
            "Tips:\n"
            "- Pre-geocoded mode is faster and doesn't require API key\n"
            "- Increase search radius for roughly positioned images\n"
            "- Use permissive quality for historical/challenging imagery"
        )

    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.is_processing:
            result = QMessageBox.question(
                self, "Confirm Close",
                "Processing is still running. Close anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if result != QMessageBox.Yes:
                event.ignore()
                return

            if self.processing_thread:
                self.processing_thread.cancel()
                self.processing_thread.wait(5000)

        self._save_settings()
        event.accept()

"""
Batch Processing Dialog for Magic Georeferencer

Dialog for batch georeferencing using vision AI for location estimation.
"""

import json
from pathlib import Path
from typing import Optional, List

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
    QApplication
)
from qgis.PyQt.QtCore import Qt, QSettings, QThread, pyqtSignal
from qgis.core import QgsCoordinateReferenceSystem

from ..core.vision_api import (
    APIProvider,
    VISION_MODELS,
    MODELS_BY_PROVIDER,
    estimate_batch_cost,
    get_provider_display_name
)
from ..core.batch_processor import (
    BatchProcessor,
    BatchConfig,
    BatchProgress,
    BatchItemResult,
    BatchItemStatus,
    find_images_in_path,
    SUPPORTED_IMAGE_EXTENSIONS
)


# Settings keys
SETTINGS_PREFIX = "MagicGeoreferencer/Batch"
SETTINGS_PROVIDER = f"{SETTINGS_PREFIX}/provider"
SETTINGS_MODEL = f"{SETTINGS_PREFIX}/model"
SETTINGS_REMEMBER_KEY = f"{SETTINGS_PREFIX}/remember_api_key"
SETTINGS_API_KEY_OPENAI = f"{SETTINGS_PREFIX}/api_key_openai"
SETTINGS_API_KEY_ANTHROPIC = f"{SETTINGS_PREFIX}/api_key_anthropic"
SETTINGS_OUTPUT_SUFFIX = f"{SETTINGS_PREFIX}/output_suffix"
SETTINGS_BASEMAP = f"{SETTINGS_PREFIX}/basemap"
SETTINGS_QUALITY = f"{SETTINGS_PREFIX}/quality"


class BatchProcessingThread(QThread):
    """Thread for running batch processing."""

    progress_updated = pyqtSignal(object)  # BatchProgress
    log_message = pyqtSignal(str)
    finished_processing = pyqtSignal(list)  # List[BatchItemResult]

    def __init__(self, model_manager, image_paths: List[Path], config: BatchConfig, iface=None):
        super().__init__()
        self.model_manager = model_manager
        self.image_paths = image_paths
        self.config = config
        self.iface = iface
        self.processor: Optional[BatchProcessor] = None

    def run(self):
        """Run batch processing in thread."""
        self.processor = BatchProcessor(
            model_manager=self.model_manager,
            iface=self.iface,
            progress_callback=lambda p: self.progress_updated.emit(p),
            log_callback=lambda m: self.log_message.emit(m)
        )

        results = self.processor.process_batch(self.image_paths, self.config)
        self.finished_processing.emit(results)

    def cancel(self):
        """Cancel processing."""
        if self.processor:
            self.processor.cancel()


class BatchDialog(QDialog):
    """Dialog for batch georeferencing."""

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
        self.processing_thread: Optional[BatchProcessingThread] = None
        self.is_processing = False

        # Load tile sources
        tile_config_path = Path(__file__).parent.parent / 'config' / 'tile_sources.json'
        with open(tile_config_path, 'r', encoding='utf-8') as f:
            self.tile_sources = json.load(f)

        # Setup UI
        self.setWindowTitle("Batch Georeference")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
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

        # Input section
        layout.addWidget(self._create_input_group())

        # API section
        layout.addWidget(self._create_api_group())

        # Matching section
        layout.addWidget(self._create_matching_group())

        # Output section
        layout.addWidget(self._create_output_group())

        # Processing section (with log)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._create_processing_group())
        layout.addWidget(splitter)

        # Action buttons
        layout.addLayout(self._create_action_buttons())

        self.setLayout(layout)

    def _create_input_group(self) -> QGroupBox:
        """Create input selection group."""
        group = QGroupBox("Input Images")
        layout = QVBoxLayout()

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

    def _create_api_group(self) -> QGroupBox:
        """Create API configuration group."""
        group = QGroupBox("Vision API")
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

        # Automatic mode explanation
        auto_info = QLabel(
            "By default, the AI will automatically select the best basemap for each image "
            "and try progressively relaxed quality thresholds until a match is found. "
            "Enable manual control below to override these settings."
        )
        auto_info.setWordWrap(True)
        auto_info.setStyleSheet("color: #666; font-style: italic; margin-bottom: 8px;")
        layout.addWidget(auto_info)

        # Manual control checkbox
        self.manual_control_checkbox = QCheckBox("Enable manual control (override automatic settings)")
        self.manual_control_checkbox.setChecked(False)
        self.manual_control_checkbox.stateChanged.connect(self._on_manual_control_changed)
        layout.addWidget(self.manual_control_checkbox)

        # Basemap source (label for display)
        self.basemap_label = QLabel("Basemap:")
        basemap_layout = QHBoxLayout()
        basemap_layout.addWidget(self.basemap_label)

        self.basemap_combo = QComboBox()
        for source_key, source_config in self.tile_sources.items():
            self.basemap_combo.addItem(source_config['name'], source_key)
        basemap_layout.addWidget(self.basemap_combo)

        basemap_layout.addStretch()
        layout.addLayout(basemap_layout)

        # Auto basemap indicator
        self.auto_basemap_label = QLabel("(AI will select best basemap per image)")
        self.auto_basemap_label.setStyleSheet("color: #080; font-style: italic;")
        layout.addWidget(self.auto_basemap_label)

        # Match quality (label for display)
        self.quality_label = QLabel("Quality:")
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(self.quality_label)

        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Strict (0.85)", "strict")
        self.quality_combo.addItem("Balanced (0.70)", "balanced")
        self.quality_combo.addItem("Permissive (0.55)", "permissive")
        self.quality_combo.addItem("Very Permissive (0.20)", "very_permissive")
        self.quality_combo.setCurrentIndex(1)  # Default to Balanced
        quality_layout.addWidget(self.quality_combo)

        quality_layout.addStretch()
        layout.addLayout(quality_layout)

        # Auto quality indicator
        self.auto_quality_label = QLabel("(Will try strict first, then relax until match found)")
        self.auto_quality_label.setStyleSheet("color: #080; font-style: italic;")
        layout.addWidget(self.auto_quality_label)

        # Set initial state (automatic mode - controls disabled)
        self._on_manual_control_changed()

        group.setLayout(layout)
        return group

    def _on_manual_control_changed(self):
        """Handle manual control checkbox state change."""
        manual_mode = self.manual_control_checkbox.isChecked()

        # Enable/disable manual controls
        self.basemap_combo.setEnabled(manual_mode)
        self.basemap_label.setEnabled(manual_mode)
        self.quality_combo.setEnabled(manual_mode)
        self.quality_label.setEnabled(manual_mode)

        # Show/hide automatic mode indicators
        self.auto_basemap_label.setVisible(not manual_mode)
        self.auto_quality_label.setVisible(not manual_mode)

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
        self.suffix_edit.setPlaceholderText("e.g., _georef")
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

    def _create_processing_group(self) -> QWidget:
        """Create processing status group."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Progress section
        progress_layout = QVBoxLayout()

        # Current item label
        self.current_item_label = QLabel("Ready to process")
        progress_layout.addWidget(self.current_item_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Progress detail
        self.progress_detail_label = QLabel("")
        progress_layout.addWidget(self.progress_detail_label)

        layout.addLayout(progress_layout)

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

    def _load_settings(self):
        """Load saved settings."""
        # Provider
        provider_str = self.settings.value(SETTINGS_PROVIDER, "openai")
        provider_idx = 0 if provider_str == "openai" else 1
        self.provider_combo.setCurrentIndex(provider_idx)

        # Model (loaded after provider is set)
        saved_model = self.settings.value(SETTINGS_MODEL, "")

        # Remember API key setting
        remember = self.settings.value(SETTINGS_REMEMBER_KEY, False, type=bool)
        self.remember_key_checkbox.setChecked(remember)

        # Load API key if remember is enabled
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

        # Update model combo and try to restore saved model
        self._update_model_combo()
        if saved_model:
            idx = self.model_combo.findData(saved_model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

    def _save_settings(self):
        """Save current settings."""
        # Provider
        provider = self.provider_combo.currentData()
        self.settings.setValue(SETTINGS_PROVIDER, provider.value)

        # Model
        model = self.model_combo.currentData()
        if model:
            self.settings.setValue(SETTINGS_MODEL, model)

        # Remember API key
        remember = self.remember_key_checkbox.isChecked()
        self.settings.setValue(SETTINGS_REMEMBER_KEY, remember)

        # Save API key if remember is enabled
        if remember:
            if provider == APIProvider.OPENAI:
                self.settings.setValue(SETTINGS_API_KEY_OPENAI, self.api_key_edit.text())
            else:
                self.settings.setValue(SETTINGS_API_KEY_ANTHROPIC, self.api_key_edit.text())
        else:
            # Clear saved keys if not remembering
            self.settings.remove(SETTINGS_API_KEY_OPENAI)
            self.settings.remove(SETTINGS_API_KEY_ANTHROPIC)

        # Output suffix
        self.settings.setValue(SETTINGS_OUTPUT_SUFFIX, self.suffix_edit.text())

        # Basemap
        self.settings.setValue(SETTINGS_BASEMAP, self.basemap_combo.currentData())

        # Quality
        self.settings.setValue(SETTINGS_QUALITY, self.quality_combo.currentData())

    def _on_provider_changed(self):
        """Handle provider selection change."""
        self._update_model_combo()

        # Load API key for this provider if remember is enabled
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
            self.output_warning_label.setText(
                "WARNING: Output will overwrite input files!"
            )
        elif not suffix:
            self.output_warning_label.setText(
                "WARNING: Files with same name in output folder will be overwritten!"
            )
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
            self,
            "Select Folder with Images",
            str(Path.home())
        )

        if folder:
            folder_path = Path(folder)
            self.input_path_edit.setText(str(folder_path))
            self.image_paths = find_images_in_path(folder_path)
            self._update_image_count()

    def _browse_files(self):
        """Browse for input files."""
        extensions = " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            str(Path.home()),
            f"Images ({extensions});;All Files (*)"
        )

        if files:
            self.image_paths = [Path(f) for f in files]
            if len(files) == 1:
                self.input_path_edit.setText(str(self.image_paths[0]))
            else:
                self.input_path_edit.setText(f"{len(files)} files selected")
            self._update_image_count()

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
            self,
            "Select Output Folder",
            str(Path.home())
        )

        if folder:
            self.output_dir_edit.setText(folder)

    def _validate_inputs(self) -> tuple:
        """Validate inputs before processing.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(self.image_paths) == 0:
            return False, "Please select images to process"

        if not self.api_key_edit.text().strip():
            return False, "Please enter your API key"

        # Check if model is loaded
        if self.model_manager is None:
            return False, "Model manager not initialized"

        # Warn about overwrite
        suffix = self.suffix_edit.text()
        output_dir = self.output_dir_edit.text().strip()
        if not suffix and not output_dir:
            result = QMessageBox.warning(
                self,
                "Confirm Overwrite",
                "Output files will overwrite input files!\n\n"
                "Are you sure you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if result != QMessageBox.Yes:
                return False, "Cancelled by user"

        return True, ""

    def _start_processing(self):
        """Start batch processing."""
        # Validate inputs
        is_valid, error_msg = self._validate_inputs()
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_msg)
            return

        # Save settings
        self._save_settings()

        # Create config
        provider = self.provider_combo.currentData()
        model_key = self.model_combo.currentData()

        output_dir = self.output_dir_edit.text().strip()
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = None

        # Determine automatic vs manual mode
        manual_mode = self.manual_control_checkbox.isChecked()

        config = BatchConfig(
            api_provider=provider,
            api_key=self.api_key_edit.text().strip(),
            model_key=model_key,
            output_directory=output_dir,
            output_suffix=self.suffix_edit.text(),
            # Automatic mode settings (inverted from manual_mode)
            auto_basemap=not manual_mode,
            auto_quality=not manual_mode,
            # Manual settings (used when auto modes are disabled)
            basemap_source=self.basemap_combo.currentData(),
            quality_preset=self.quality_combo.currentData(),
            # Never auto-load results in batch mode
            auto_load_result=False,
        )

        # Update UI for processing state
        self.is_processing = True
        self._update_ui_for_processing(True)

        # Clear log
        self.log_text.clear()

        # Create and start processing thread
        self.processing_thread = BatchProcessingThread(
            self.model_manager,
            self.image_paths,
            config,
            self.iface
        )
        self.processing_thread.progress_updated.connect(self._on_progress_updated)
        self.processing_thread.log_message.connect(self._on_log_message)
        self.processing_thread.finished_processing.connect(self._on_processing_finished)
        self.processing_thread.start()

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
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _on_processing_finished(self, results: List[BatchItemResult]):
        """Handle processing completion."""
        self.is_processing = False
        self._update_ui_for_processing(False)

        # Show summary
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
        # Disable/enable input controls
        self.provider_combo.setEnabled(not processing)
        self.model_combo.setEnabled(not processing)
        self.api_key_edit.setEnabled(not processing)
        self.remember_key_checkbox.setEnabled(not processing)
        self.manual_control_checkbox.setEnabled(not processing)
        self.output_dir_edit.setEnabled(not processing)
        self.suffix_edit.setEnabled(not processing)

        # Basemap and quality controls depend on manual control state when not processing
        if not processing:
            manual_mode = self.manual_control_checkbox.isChecked()
            self.basemap_combo.setEnabled(manual_mode)
            self.basemap_label.setEnabled(manual_mode)
            self.quality_combo.setEnabled(manual_mode)
            self.quality_label.setEnabled(manual_mode)
        else:
            self.basemap_combo.setEnabled(False)
            self.basemap_label.setEnabled(False)
            self.quality_combo.setEnabled(False)
            self.quality_label.setEnabled(False)

        # Toggle buttons
        self.start_btn.setVisible(not processing)
        self.cancel_btn.setVisible(processing)
        self.cancel_btn.setEnabled(processing)
        self.cancel_btn.setText("Cancel")
        self.close_btn.setEnabled(not processing)

        # Reset progress if not processing
        if not processing:
            self.current_item_label.setText("Ready to process")
            self.progress_bar.setValue(0)

    def _on_close(self):
        """Handle close button."""
        if self.is_processing:
            result = QMessageBox.question(
                self,
                "Confirm Close",
                "Processing is still running. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if result != QMessageBox.Yes:
                return

            # Cancel processing
            if self.processing_thread:
                self.processing_thread.cancel()
                self.processing_thread.wait(5000)  # Wait up to 5 seconds

        self._save_settings()
        self.reject()

    def _show_help(self):
        """Show help dialog."""
        QMessageBox.information(
            self,
            "Batch Georeference - Help",
            "Batch Georeference - AI-Powered Location Estimation\n\n"
            "This tool processes multiple images automatically by:\n"
            "1. Using a vision AI to analyze each image and estimate its location\n"
            "2. Fetching reference tiles for the estimated area\n"
            "3. Running feature matching to find control points\n"
            "4. Georeferencing the image\n\n"
            "API Keys:\n"
            "- OpenAI: Get your key at platform.openai.com/api-keys\n"
            "- Anthropic: Get your key at console.anthropic.com\n\n"
            "Cost Estimates:\n"
            "- GPT-4.1: ~$0.01-0.03 per image\n"
            "- GPT-4.1 Nano: ~$0.001-0.005 per image\n"
            "- Claude Sonnet 4.5: ~$0.01-0.03 per image\n"
            "- Claude Haiku 4.5: ~$0.002-0.008 per image\n\n"
            "Tips:\n"
            "- Use a cheaper model (GPT-4.1 Nano or Haiku) for large batches\n"
            "- Images with clear text labels work best\n"
            "- Historical maps may have lower success rates"
        )

    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.is_processing:
            result = QMessageBox.question(
                self,
                "Confirm Close",
                "Processing is still running. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if result != QMessageBox.Yes:
                event.ignore()
                return

            # Cancel processing
            if self.processing_thread:
                self.processing_thread.cancel()
                self.processing_thread.wait(5000)

        self._save_settings()
        event.accept()

"""
Main Dialog for Magic Georeferencer

Primary user interface for the plugin.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

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
    QSizePolicy
)
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsProject, QgsCoordinateReferenceSystem

from ..core import ModelManager, Matcher, TileFetcher, GCPGenerator, Georeferencer
from .progress_dialog import ProgressDialog


class MagicGeoreferencerDialog(QDialog):
    """Main dialog for Magic Georeferencer"""

    def __init__(self, iface, parent=None):
        """Initialize dialog.

        Args:
            iface: QGIS interface
            parent: Parent widget
        """
        super().__init__(parent)

        self.iface = iface
        self.source_image_path = None
        self.model_manager = None
        self.matcher = None

        # Load settings
        config_path = Path(__file__).parent.parent / 'config' / 'default_settings.json'
        with open(config_path, 'r') as f:
            self.settings = json.load(f)

        # Load tile sources
        tile_config_path = Path(__file__).parent.parent / 'config' / 'tile_sources.json'
        with open(tile_config_path, 'r') as f:
            self.tile_sources = json.load(f)

        # Setup UI
        self.setWindowTitle("Magic Georeferencer")
        self.setMinimumWidth(600)
        self._setup_ui()

        # Initialize model manager
        self._init_model_manager()

    def _setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout()

        # 1. Load Ungeoreferenced Image
        layout.addWidget(self._create_image_selection_group())

        # 2. Position Map Canvas
        layout.addWidget(self._create_map_position_group())

        # 3. Configure Matching
        layout.addWidget(self._create_matching_config_group())

        # 4. Action Buttons
        layout.addLayout(self._create_action_buttons())

        # Status bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def _create_image_selection_group(self) -> QGroupBox:
        """Create image selection group"""
        group = QGroupBox("1. Load Ungeoreferenced Image")
        layout = QVBoxLayout()

        # File selection row
        file_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("Select an image file...")
        file_layout.addWidget(self.image_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_image)
        file_layout.addWidget(browse_btn)

        layout.addLayout(file_layout)

        # Image info label
        self.image_info_label = QLabel("")
        layout.addWidget(self.image_info_label)

        group.setLayout(layout)
        return group

    def _create_map_position_group(self) -> QGroupBox:
        """Create map positioning group"""
        group = QGroupBox("2. Position Map Canvas")
        layout = QVBoxLayout()

        # Overlay preview checkbox
        self.overlay_preview_checkbox = QCheckBox("Show image overlay preview")
        self.overlay_preview_checkbox.setEnabled(False)  # TODO: Implement overlay
        layout.addWidget(self.overlay_preview_checkbox)

        # Instructions
        info_label = QLabel(
            "9 Navigate the map to the approximate location of your image.\n"
            "The visible map extent will be used for matching."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group.setLayout(layout)
        return group

    def _create_matching_config_group(self) -> QGroupBox:
        """Create matching configuration group"""
        group = QGroupBox("3. Configure Matching")
        layout = QVBoxLayout()

        # Basemap source
        basemap_layout = QHBoxLayout()
        basemap_layout.addWidget(QLabel("Basemap Source:"))

        self.basemap_combo = QComboBox()
        for source_key, source_config in self.tile_sources.items():
            self.basemap_combo.addItem(source_config['name'], source_key)
        basemap_layout.addWidget(self.basemap_combo)
        layout.addLayout(basemap_layout)

        # Basemap description
        self.basemap_desc_label = QLabel("")
        self.basemap_desc_label.setWordWrap(True)
        self.basemap_combo.currentIndexChanged.connect(self._update_basemap_description)
        self._update_basemap_description()
        layout.addWidget(self.basemap_desc_label)

        # Match quality
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Match Quality:"))

        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Strict (0.85)", "strict")
        self.quality_combo.addItem("Balanced (0.70)", "balanced")
        self.quality_combo.addItem("Permissive (0.55)", "permissive")
        self.quality_combo.setCurrentIndex(1)  # Default to Balanced
        quality_layout.addWidget(self.quality_combo)
        layout.addLayout(quality_layout)

        # Progressive refinement
        self.progressive_checkbox = QCheckBox("Progressive refinement (slower but more accurate)")
        self.progressive_checkbox.setChecked(
            self.settings['matching']['enable_progressive_refinement']
        )
        layout.addWidget(self.progressive_checkbox)

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons"""
        layout = QHBoxLayout()

        # Help button
        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self._show_help)
        layout.addWidget(help_btn)

        # Settings button
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._show_settings)
        settings_btn.setEnabled(False)  # TODO: Implement settings
        layout.addWidget(settings_btn)

        layout.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)

        # Match button
        self.match_btn = QPushButton("Match && Generate GCPs")
        self.match_btn.clicked.connect(self._run_matching)
        self.match_btn.setDefault(True)
        layout.addWidget(self.match_btn)

        return layout

    def _init_model_manager(self):
        """Initialize model manager and check for first run"""
        try:
            self.model_manager = ModelManager()

            if self.model_manager.check_first_run():
                self._show_first_run_dialog()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize model manager:\n{str(e)}"
            )

    def _show_first_run_dialog(self):
        """Show first-run dialog for weight download"""
        device_info = self.model_manager.get_device_info()

        msg = QMessageBox(self)
        msg.setWindowTitle("First Run - Model Weights Required")
        msg.setIcon(QMessageBox.Information)

        text = (
            "Magic Georeferencer uses an AI model to automatically match images.\n\n"
            f"The model weights (~{self.settings['model']['weights_size_mb']} MB) "
            "need to be downloaded once.\n\n"
        )

        if device_info['cuda_available']:
            text += f" CUDA GPU detected: {device_info['cuda_device_name']}\n"
            text += "  Recommended version: GPU (faster)\n"
        else:
            text += " No CUDA GPU detected\n"
            text += "  Will use CPU (slower)\n"

        text += "\nWould you like to download the model weights now?"

        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)

        result = msg.exec_()

        if result == QMessageBox.Yes:
            self._download_weights()

    def _download_weights(self):
        """Download model weights with progress"""
        progress = ProgressDialog(self, "Downloading Model Weights")

        def progress_callback(current, total):
            progress.set_progress(current, total)
            progress.set_status("Downloading model weights...")

        progress.show()

        try:
            success, message = self.model_manager.download_weights(progress_callback)

            progress.close()

            if success:
                QMessageBox.information(
                    self,
                    "Download Complete",
                    "Model weights downloaded successfully!"
                )
                self._load_model()
            else:
                QMessageBox.critical(
                    self,
                    "Download Failed",
                    f"Failed to download model weights:\n{message}"
                )

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Download Error",
                f"Error during download:\n{str(e)}"
            )

    def _load_model(self):
        """Load AI model"""
        progress = ProgressDialog(self, "Loading Model")
        progress.set_status("Loading AI model into memory...")
        progress.set_indeterminate(True)
        progress.show()

        try:
            success, message = self.model_manager.load_model()

            progress.close()

            if success:
                self.status_label.setText(f"Ready - {message}")
                self.matcher = Matcher(self.model_manager)
            else:
                QMessageBox.critical(
                    self,
                    "Model Loading Failed",
                    f"Failed to load model:\n{message}"
                )
                self.status_label.setText("Error: Model not loaded")

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Model Loading Error",
                f"Error loading model:\n{str(e)}"
            )

    def _browse_image(self):
        """Browse for image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ungeoreferenced Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
        )

        if file_path:
            self.source_image_path = Path(file_path)
            self.image_path_edit.setText(str(self.source_image_path))

            # Get image info
            try:
                from PIL import Image
                img = Image.open(self.source_image_path)
                width, height = img.size
                self.image_info_label.setText(f"Image size: {width} × {height} px")
            except:
                self.image_info_label.setText("Image loaded")

    def _update_basemap_description(self):
        """Update basemap description label"""
        source_key = self.basemap_combo.currentData()
        if source_key and source_key in self.tile_sources:
            desc = self.tile_sources[source_key].get('description', '')
            self.basemap_desc_label.setText(f"9 {desc}")

    def _validate_inputs(self) -> Tuple[bool, str]:
        """Validate inputs before processing"""
        if self.source_image_path is None or not self.source_image_path.exists():
            return False, "Please select a valid source image"

        if self.model_manager is None or self.matcher is None:
            return False, "Model not loaded. Please restart the plugin."

        # Check map canvas has valid extent
        if self.iface.mapCanvas().extent().isEmpty():
            return False, "Map canvas extent is empty. Please zoom to a location."

        return True, ""

    def _run_matching(self):
        """Run the matching workflow"""
        # Validate inputs
        is_valid, error_msg = self._validate_inputs()
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_msg)
            return

        # TODO: Implement full matching workflow
        QMessageBox.information(
            self,
            "Not Implemented",
            "Full matching workflow is not yet implemented.\n\n"
            "This is the initial plugin structure. The matching workflow will:\n"
            "1. Capture map canvas or fetch tiles\n"
            "2. Run MatchAnything inference\n"
            "3. Filter and validate matches\n"
            "4. Generate GCPs\n"
            "5. Show confidence viewer\n"
            "6. Georeference the image"
        )

    def _show_help(self):
        """Show help dialog"""
        QMessageBox.information(
            self,
            "Magic Georeferencer - Help",
            "Magic Georeferencer - AI-Powered Image Georeferencing\n\n"
            "1. Load an ungeoreferenced image (map, aerial photo, sketch)\n"
            "2. Navigate the QGIS map to the approximate location\n"
            "3. Select a basemap source that matches your image type\n"
            "4. Click 'Match & Generate GCPs' to automatically georeference\n\n"
            "Tips:\n"
            "- Use OSM Standard for road maps\n"
            "- Use ESRI World Imagery for aerial photos\n"
            "- Navigate close to the actual location for best results\n"
            "- Progressive refinement improves accuracy but takes longer"
        )

    def _show_settings(self):
        """Show settings dialog"""
        # TODO: Implement settings dialog
        pass

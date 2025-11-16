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
        with open(config_path, 'r', encoding='utf-8') as f:
            self.settings = json.load(f)

        # Load tile sources
        tile_config_path = Path(__file__).parent.parent / 'config' / 'tile_sources.json'
        with open(tile_config_path, 'r', encoding='utf-8') as f:
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
        group = QGroupBox("2. Align Map View")
        layout = QVBoxLayout()

        # Instructions
        info_label = QLabel(
            "You will be guided to align your map view with the source image.\n\n"
            "The plugin will show an alignment dialog where you can:\n"
            "• Center the map on your image location\n"
            "• Match the scale by adjusting zoom\n"
            "• Choose which dimension (horizontal/vertical) to match\n\n"
            "The plugin will then automatically find the best zoom level for matching."
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
        self.quality_combo.addItem("Very Permissive (0.20) - for historical imagery", "very_permissive")
        self.quality_combo.setCurrentIndex(1)  # Default to Balanced
        quality_layout.addWidget(self.quality_combo)
        layout.addLayout(quality_layout)

        # Progressive refinement
        self.progressive_checkbox = QCheckBox("Progressive refinement (slower but more accurate)")
        self.progressive_checkbox.setChecked(
            self.settings['matching']['enable_progressive_refinement']
        )
        layout.addWidget(self.progressive_checkbox)

        # Debug output directory
        debug_layout = QVBoxLayout()
        debug_label = QLabel("Debug Output Directory (images saved here for inspection):")
        debug_layout.addWidget(debug_label)

        debug_path_layout = QHBoxLayout()
        self.debug_dir_edit = QLineEdit()
        self.debug_dir_edit.setText(str(Path.home() / "georefio_debug"))
        self.debug_dir_edit.setReadOnly(True)
        debug_path_layout.addWidget(self.debug_dir_edit)

        browse_debug_btn = QPushButton("Browse...")
        browse_debug_btn.clicked.connect(self._browse_debug_dir)
        debug_path_layout.addWidget(browse_debug_btn)

        open_debug_btn = QPushButton("Open Folder")
        open_debug_btn.clicked.connect(self._open_debug_dir)
        debug_path_layout.addWidget(open_debug_btn)

        debug_layout.addLayout(debug_path_layout)
        layout.addLayout(debug_layout)

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
            f"The model weights (~{self.settings['model']['estimated_size_mb']} MB) "
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

        def progress_callback(status, current, total):
            progress.set_status(status)
            progress.set_progress(current, total)
            # Process events to keep UI responsive
            from qgis.PyQt.QtCore import QCoreApplication
            QCoreApplication.processEvents()

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
                self.image_info_label.setText(f"Image size: {width} x {height} px")
            except:
                self.image_info_label.setText("Image loaded")

    def _update_basemap_description(self):
        """Update basemap description label"""
        source_key = self.basemap_combo.currentData()
        if source_key and source_key in self.tile_sources:
            desc = self.tile_sources[source_key].get('description', '')
            self.basemap_desc_label.setText(f"9 {desc}")

    def _browse_debug_dir(self):
        """Browse for debug output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Debug Output Directory",
            str(Path.home())
        )

        if dir_path:
            self.debug_dir_edit.setText(dir_path)

    def _open_debug_dir(self):
        """Open the debug directory in file explorer"""
        import subprocess
        import platform

        debug_dir = Path(self.debug_dir_edit.text())

        # Create directory if it doesn't exist
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Open in file explorer based on OS
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.run(['explorer', str(debug_dir)])
            elif system == "Darwin":  # macOS
                subprocess.run(['open', str(debug_dir)])
            else:  # Linux
                subprocess.run(['xdg-open', str(debug_dir)])
        except Exception as e:
            QMessageBox.information(
                self,
                "Debug Directory",
                f"Debug directory: {debug_dir}\n\nCouldn't open automatically: {e}"
            )

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
        """Run the full matching workflow"""
        # Validate inputs
        is_valid, error_msg = self._validate_inputs()
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_msg)
            return

        # Step 1: Show alignment dialog
        from .alignment_dialog import AlignmentDialog

        alignment_dialog = AlignmentDialog(self.source_image_path, self.iface, self)
        alignment_info = alignment_dialog.get_alignment_info()

        if alignment_info is None:
            # User cancelled
            return

        center_lat, center_lon, extent_meters, extent_dimension = alignment_info

        print(f"\nAlignment info:")
        print(f"  Center: {center_lat:.6f}, {center_lon:.6f}")
        print(f"  Extent: {extent_meters:.0f} meters ({extent_dimension})")

        # Create progress dialog
        progress = ProgressDialog(self, "Magic Georeferencer")

        try:
            # Step 2: Load source image
            progress.set_status("Loading source image...")
            progress.set_progress(0, 100)
            progress.show()

            from PIL import Image
            import numpy as np

            # Load source image and handle large images
            src_image_pil = Image.open(self.source_image_path).convert('RGB')
            original_size = src_image_pil.size  # (width, height)

            # Check if image is too large and needs downsampling
            MAX_DIMENSION = 4096  # Maximum size for matching
            downsample_factor = 1.0

            if original_size[0] > MAX_DIMENSION or original_size[1] > MAX_DIMENSION:
                # Calculate downsample factor
                downsample_factor = max(original_size[0] / MAX_DIMENSION,
                                       original_size[1] / MAX_DIMENSION)
                new_width = int(original_size[0] / downsample_factor)
                new_height = int(original_size[1] / downsample_factor)

                print(f"\n{'='*80}")
                print("LARGE IMAGE DETECTED - Downsampling for matching")
                print(f"{'='*80}")
                print(f"Original size: {original_size[0]} x {original_size[1]} px")
                print(f"Downsample factor: {downsample_factor:.2f}x")
                print(f"Matching size: {new_width} x {new_height} px")
                print(f"Note: GCPs will be scaled back to original resolution for georeferencing")
                print(f"{'='*80}\n")

                # Downsample using high-quality Lanczos resampling
                src_image_pil = src_image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

            src_image = np.array(src_image_pil)
            src_image_size = (src_image.shape[1], src_image.shape[0])  # (width, height)

            # Store original size for GCP scaling
            original_image_size = original_size

            progress.set_progress(10, 100)

            # Step 3: Calculate optimal zoom level
            progress.set_status("Calculating optimal zoom level...")

            tile_fetcher = TileFetcher()
            base_zoom = tile_fetcher.calculate_optimal_zoom(extent_meters, target_pixels=832)

            print(f"Calculated base zoom level: {base_zoom}")

            progress.set_progress(15, 100)

            # Step 4: Run multi-zoom matching
            progress.set_status("Running AI matching at multiple zoom levels...")

            basemap_source = self.basemap_combo.currentData()

            # Get quality threshold
            quality_preset = self.quality_combo.currentData()
            confidence_threshold = self.settings['matching']['confidence_thresholds'][quality_preset]

            # Determine zoom range based on progressive setting
            if self.progressive_checkbox.isChecked():
                zoom_range = 2  # Try base_zoom-2 to base_zoom+2 (5 zoom levels)
            else:
                zoom_range = 1  # Try base_zoom-1 to base_zoom+1 (3 zoom levels)

            # Set debug directory for matcher
            self.matcher.debug_dir = Path(self.debug_dir_edit.text())

            # Run multi-zoom matching
            match_result, ref_image, ref_extent, best_zoom = self.matcher.match_multi_zoom(
                image_src=src_image,
                center_lat=center_lat,
                center_lon=center_lon,
                extent_meters=extent_meters,
                extent_dimension=extent_dimension,
                tile_fetcher=tile_fetcher,
                basemap_source=basemap_source,
                base_zoom=base_zoom,
                zoom_range=zoom_range
            )

            ref_image_size = (ref_image.shape[1], ref_image.shape[0])

            progress.set_progress(60, 100)

            # Step 4: Filter matches
            progress.set_status("Filtering matches...")

            match_result = self.matcher.filter_matches(
                match_result,
                confidence_threshold=confidence_threshold,
                min_gcps=self.settings['matching']['min_gcps']
            )

            # Check if we have enough matches
            if match_result.num_matches() < self.settings['matching']['min_gcps']:
                progress.close()

                # Build helpful error message
                suggestions = [
                    "- Try 'Very Permissive (0.20)' quality setting for historical imagery",
                    "- Use a different basemap source (try both OSM and aerial imagery)",
                    "- Zoom to an area with stable features (roads, coastlines, buildings)",
                    "- Ensure you're viewing the correct geographic location",
                ]

                # Add specific advice based on match count
                if match_result.num_matches() == 0:
                    suggestions.insert(0, "- The images may be too different or location incorrect")
                elif match_result.num_matches() < 4:
                    suggestions.insert(0, "- Very few matches suggest challenging imagery (historical photos, major changes)")
                    suggestions.append("- Consider preprocessing the historical image (contrast, rotation)")

                QMessageBox.warning(
                    self,
                    "Insufficient Matches",
                    f"Only {match_result.num_matches()} matches found "
                    f"(confidence: {match_result.mean_confidence():.3f}).\n"
                    f"Minimum required: {self.settings['matching']['min_gcps']}\n\n"
                    "Suggestions:\n" + "\n".join(suggestions)
                )
                return

            progress.set_progress(70, 100)

            # Step 5: Generate GCPs
            progress.set_status("Generating Ground Control Points...")

            # ref_extent is in EPSG:3857 (Web Mercator)
            web_mercator_crs = QgsCoordinateReferenceSystem('EPSG:3857')

            gcp_generator = GCPGenerator()

            # Use original image size for GCPs (not downsampled size)
            # If image was downsampled, GCP coordinates will be scaled automatically
            gcps = gcp_generator.matches_to_gcps(
                match_result,
                ref_extent,
                web_mercator_crs,
                ref_image_size,
                original_image_size,  # Use original size, not downsampled
                downsample_factor=downsample_factor  # Pass scale factor
            )

            progress.set_progress(80, 100)

            # Step 6: Show confidence viewer (if enabled)
            progress.close()

            if self.settings['ui']['show_confidence_viewer']:
                from .confidence_viewer import ConfidenceViewer

                viewer = ConfidenceViewer(match_result, self)
                result = viewer.exec_()

                # If user cancelled, stop here
                if result != viewer.Accepted:
                    return

            # Step 7: Georeference
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Georeferenced Image",
                str(self.source_image_path.with_name(
                    self.source_image_path.stem + "_georef.tif"
                )),
                "GeoTIFF (*.tif *.tiff)"
            )

            if not output_path:
                return  # User cancelled

            output_path = Path(output_path)

            progress = ProgressDialog(self, "Georeferencing")
            progress.set_status("Initializing georeferencing...")
            progress.set_progress(0, 100)
            progress.show()

            # Process events to show progress dialog
            from qgis.PyQt.QtCore import QCoreApplication
            QCoreApplication.processEvents()

            georeferencer = Georeferencer(self.iface)

            # Suggest transform type
            transform_type = georeferencer.suggest_transform_type(
                match_result.num_matches(),
                match_result.distribution_quality
            )

            # Define progress callback
            def update_progress(status, current, total):
                progress.set_status(status)
                progress.set_progress(current, total)
                QCoreApplication.processEvents()  # Keep UI responsive

            # Perform georeferencing with progress updates
            success, message = georeferencer.georeference_image(
                self.source_image_path,
                gcps,
                output_path,
                web_mercator_crs,  # Use the CRS we defined earlier
                transform_type=transform_type,
                resampling=self.settings['georeferencing']['default_resampling'],
                compression=self.settings['georeferencing']['default_compression'],
                progress_callback=update_progress
            )

            progress.close()

            if success:
                QMessageBox.information(
                    self,
                    "Success!",
                    f"Image georeferenced successfully!\n\n"
                    f"Output: {output_path}\n"
                    f"Matches: {match_result.num_matches()}\n"
                    f"Mean confidence: {match_result.mean_confidence():.2f}\n"
                    f"Transform type: {transform_type}\n\n"
                    f"The georeferenced image has been added to the map."
                )

                # Update status
                self.status_label.setText("✓ Georeferencing complete!")

            else:
                QMessageBox.critical(
                    self,
                    "Georeferencing Failed",
                    f"Failed to georeference image:\n{message}"
                )

        except Exception as e:
            if 'progress' in locals():
                progress.close()

            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred during processing:\n\n{str(e)}\n\n"
                f"Check the QGIS Python console for details."
            )

            # Print detailed error to console
            import traceback
            traceback.print_exc()

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

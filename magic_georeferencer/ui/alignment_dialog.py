"""
Alignment Dialog for Magic Georeferencer

Guides the user to align their QGIS map view with the source image by:
1. Centering the map on the source image location
2. Adjusting zoom so horizontal OR vertical extent matches
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np


class AlignmentDialog(QDialog):
    """Dialog to guide user through map alignment"""

    # Signal emitted when alignment is confirmed
    # Emits: (center_lat, center_lon, extent_meters, extent_dimension)
    # extent_dimension is 'horizontal' or 'vertical'
    alignment_confirmed = pyqtSignal(float, float, float, str)

    def __init__(self, source_image_path, iface, parent=None):
        """
        Initialize alignment dialog.

        Args:
            source_image_path: Path to source image
            iface: QGIS interface
            parent: Parent widget
        """
        super().__init__(parent)
        self.source_image_path = Path(source_image_path)
        self.iface = iface
        self.extent_dimension = 'horizontal'  # Default

        self.setWindowTitle("Align Map View with Source Image")
        self.resize(800, 900)

        self._setup_ui()
        self._load_source_image()

    def _setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("<h2>Step 1: Align Your Map View</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "<p><b>Before we can match your image, we need you to roughly align "
            "the QGIS map view with your source image.</b></p>"
            "<p>This doesn't need to be perfect - just get it close!</p>"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Source image preview
        preview_group = QGroupBox("Your Source Image")
        preview_layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        preview_layout.addWidget(self.image_label)

        self.image_info_label = QLabel()
        self.image_info_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.image_info_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Alignment instructions
        align_group = QGroupBox("Alignment Instructions")
        align_layout = QVBoxLayout()

        step1 = QLabel(
            "<b>Step 1: Center the Map</b><br>"
            "Pan your QGIS map so the <b>center</b> of the visible area is roughly "
            "at the <b>center</b> of where your source image is located."
        )
        step1.setWordWrap(True)
        align_layout.addWidget(step1)

        step2 = QLabel(
            "<b>Step 2: Match the Scale</b><br>"
            "Zoom in or out so that features in your QGIS view are at approximately "
            "the same scale as in your source image. Choose ONE dimension to match:"
        )
        step2.setWordWrap(True)
        align_layout.addWidget(step2)

        # Radio buttons for extent dimension choice
        radio_layout = QHBoxLayout()
        self.radio_group = QButtonGroup()

        self.radio_horizontal = QRadioButton(
            "Match Horizontal Extent\n"
            "(left-to-right width of visible area)"
        )
        self.radio_horizontal.setChecked(True)
        self.radio_group.addButton(self.radio_horizontal)
        radio_layout.addWidget(self.radio_horizontal)

        self.radio_vertical = QRadioButton(
            "Match Vertical Extent\n"
            "(top-to-bottom height of visible area)"
        )
        self.radio_group.addButton(self.radio_vertical)
        radio_layout.addWidget(self.radio_vertical)

        align_layout.addLayout(radio_layout)

        # Connect radio buttons
        self.radio_horizontal.toggled.connect(
            lambda checked: self._set_extent_dimension('horizontal') if checked else None
        )
        self.radio_vertical.toggled.connect(
            lambda checked: self._set_extent_dimension('vertical') if checked else None
        )

        # Example
        example = QLabel(
            "<i>Example: If your source image shows a 2km wide area, zoom your QGIS map "
            "so the visible width is also about 2km. The height doesn't need to match - "
            "just pick one dimension.</i>"
        )
        example.setWordWrap(True)
        example.setStyleSheet("color: #666; margin-top: 10px;")
        align_layout.addWidget(example)

        align_group.setLayout(align_layout)
        layout.addWidget(align_group)

        # Current map info
        self.map_info_group = QGroupBox("Current Map View Info")
        map_info_layout = QVBoxLayout()

        self.map_info_label = QLabel()
        self.map_info_label.setWordWrap(True)
        map_info_layout.addWidget(self.map_info_label)

        # Info about auto-refresh
        auto_refresh_note = QLabel(
            "<i style='color: #666; font-size: 9pt;'>"
            "Map info updates automatically as you pan/zoom QGIS"
            "</i>"
        )
        auto_refresh_note.setWordWrap(True)
        map_info_layout.addWidget(auto_refresh_note)

        self.map_info_group.setLayout(map_info_layout)
        layout.addWidget(self.map_info_group)

        # Initial map info
        self._update_map_info()

        # Set up timer for auto-refresh (update every 1 second)
        from PyQt5.QtCore import QTimer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._update_map_info)
        self.refresh_timer.start(1000)  # 1000ms = 1 second

        # Attribution note
        attribution_label = QLabel(
            '<p style="color: #666; font-size: 10pt; margin-top: 15px;">'
            '<b>Note:</b> When using OpenStreetMap tiles, please ensure proper attribution '
            '(© OpenStreetMap contributors) is included in any published work. '
            'See <a href="https://www.openstreetmap.org/copyright">osmorg/copyright</a> for details.'
            '</p>'
        )
        attribution_label.setWordWrap(True)
        attribution_label.setOpenExternalLinks(True)
        layout.addWidget(attribution_label)

        # Buttons
        button_layout = QHBoxLayout()

        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self._show_help)
        button_layout.addWidget(help_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self.confirm_btn = QPushButton("Confirm Alignment - Continue to Matching")
        self.confirm_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.confirm_btn.clicked.connect(self._confirm_alignment)
        button_layout.addWidget(self.confirm_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _load_source_image(self):
        """Load and display the source image"""
        try:
            # Load image
            img = Image.open(self.source_image_path)
            width, height = img.size

            # Update info
            self.image_info_label.setText(
                f"Size: {width} × {height} pixels | "
                f"Aspect ratio: {width/height:.2f}:1"
            )

            # Create thumbnail for display
            # Scale to fit in label while maintaining aspect ratio
            max_width = 700
            max_height = 400

            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            img_thumb = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to QPixmap
            img_array = np.array(img_thumb)
            if len(img_array.shape) == 2:
                # Grayscale
                height, width = img_array.shape
                bytes_per_line = width
                qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                # RGB/RGBA
                height, width, channels = img_array.shape
                bytes_per_line = channels * width
                if channels == 3:
                    qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
                else:
                    qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap)

        except Exception as e:
            self.image_label.setText(f"Error loading image: {e}")
            self.image_info_label.setText("Could not load image")

    def _set_extent_dimension(self, dimension):
        """Set which extent dimension to use"""
        self.extent_dimension = dimension
        self._update_map_info()

    def _update_map_info(self):
        """Update the current map view information"""
        try:
            canvas = self.iface.mapCanvas()
            extent = canvas.extent()
            crs = canvas.mapSettings().destinationCrs()
            scale = canvas.scale()

            # Get center point
            center = extent.center()

            # Calculate extent in meters (convert to EPSG:3857 if needed)
            from qgis.core import QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsProject

            web_mercator = QgsCoordinateReferenceSystem('EPSG:3857')
            if crs != web_mercator:
                transform = QgsCoordinateTransform(crs, web_mercator, QgsProject.instance())
                extent_3857 = transform.transformBoundingBox(extent)
                center_3857 = transform.transform(center)
            else:
                extent_3857 = extent
                center_3857 = center

            # Calculate extents in meters
            width_m = extent_3857.width()
            height_m = extent_3857.height()

            # Convert center to lat/lon for display
            wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')
            if crs != wgs84:
                transform_wgs = QgsCoordinateTransform(crs, wgs84, QgsProject.instance())
                center_wgs = transform_wgs.transform(center)
            else:
                center_wgs = center

            # Format display
            info_text = (
                f"<b>Map CRS:</b> {crs.authid()}<br>"
                f"<b>Map Scale:</b> 1:{int(scale):,}<br>"
                f"<b>Center Point:</b> {center_wgs.y():.6f}°, {center_wgs.x():.6f}°<br>"
                f"<br>"
                f"<b>Visible Extent:</b><br>"
                f"  • Horizontal: {width_m:,.0f} meters ({width_m/1000:.2f} km)<br>"
                f"  • Vertical: {height_m:,.0f} meters ({height_m/1000:.2f} km)<br>"
            )

            # Highlight the selected dimension
            if self.extent_dimension == 'horizontal':
                info_text += f"<br><b style='color: green;'>→ Will use HORIZONTAL extent ({width_m/1000:.2f} km) for matching</b>"
            else:
                info_text += f"<br><b style='color: green;'>→ Will use VERTICAL extent ({height_m/1000:.2f} km) for matching</b>"

            self.map_info_label.setText(info_text)

        except Exception as e:
            self.map_info_label.setText(f"Error reading map info: {e}")

    def _show_help(self):
        """Show help dialog"""
        help_text = """
<h3>How to Align Your Map</h3>

<p><b>Why do we need this?</b><br>
To automatically match features between your image and the map, we need to know
approximately where to look and at what scale. This alignment step helps us
fetch the right basemap tiles for comparison.</p>

<p><b>Step-by-step:</b></p>
<ol>
<li><b>Know your location:</b> You should have a rough idea of where your source
image is located geographically.</li>

<li><b>Center the map:</b> Pan the QGIS map so the center of your screen is
roughly at the center of where your source image is located.</li>

<li><b>Match the scale:</b> Zoom in or out so that the visible extent of your
QGIS map roughly matches your source image extent in ONE dimension (horizontal
or vertical - your choice).</li>

<li><b>Example:</b> If your source image shows a 3km × 2km area, you might:
   - Choose "Match Horizontal Extent"
   - Zoom QGIS until the visible width is about 3km
   - The height doesn't need to match exactly</li>

<li><b>Confirm:</b> Click "Confirm Alignment" and the plugin will automatically
fetch basemap tiles at the appropriate zoom level(s) for matching.</li>
</ol>

<p><b>Tips:</b></p>
<ul>
<li>Use the "Refresh Map Info" button to see your current extent</li>
<li>The alignment doesn't need to be perfect - rough is fine!</li>
<li>If matching fails, you can come back and try a different alignment</li>
<li>For rotated images, just align based on the geographic extent, not the rotation</li>
</ul>
"""

        msg = QMessageBox(self)
        msg.setWindowTitle("Alignment Help")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def _confirm_alignment(self):
        """Confirm alignment and emit signal"""
        try:
            canvas = self.iface.mapCanvas()
            extent = canvas.extent()
            crs = canvas.mapSettings().destinationCrs()

            # Get center point in lat/lon
            from qgis.core import QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsProject

            center = extent.center()
            wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')

            if crs != wgs84:
                transform = QgsCoordinateTransform(crs, wgs84, QgsProject.instance())
                center_wgs = transform.transform(center)
            else:
                center_wgs = center

            # Get extent in meters
            web_mercator = QgsCoordinateReferenceSystem('EPSG:3857')
            if crs != web_mercator:
                transform_3857 = QgsCoordinateTransform(crs, web_mercator, QgsProject.instance())
                extent_3857 = transform_3857.transformBoundingBox(extent)
            else:
                extent_3857 = extent

            # Get the appropriate extent dimension
            if self.extent_dimension == 'horizontal':
                extent_meters = extent_3857.width()
            else:
                extent_meters = extent_3857.height()

            # Emit signal
            self.alignment_confirmed.emit(
                center_wgs.y(),  # latitude
                center_wgs.x(),  # longitude
                extent_meters,
                self.extent_dimension
            )

            # Accept dialog
            self.accept()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to get alignment info: {e}"
            )

    def get_alignment_info(self):
        """
        Get the current alignment information.

        Returns:
            Tuple of (center_lat, center_lon, extent_meters, extent_dimension)
            or None if dialog was cancelled
        """
        if self.exec_() == QDialog.Accepted:
            try:
                canvas = self.iface.mapCanvas()
                extent = canvas.extent()
                crs = canvas.mapSettings().destinationCrs()

                from qgis.core import QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsProject

                center = extent.center()
                wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')

                if crs != wgs84:
                    transform = QgsCoordinateTransform(crs, wgs84, QgsProject.instance())
                    center_wgs = transform.transform(center)
                else:
                    center_wgs = center

                web_mercator = QgsCoordinateReferenceSystem('EPSG:3857')
                if crs != web_mercator:
                    transform_3857 = QgsCoordinateTransform(crs, web_mercator, QgsProject.instance())
                    extent_3857 = transform_3857.transformBoundingBox(extent)
                else:
                    extent_3857 = extent

                if self.extent_dimension == 'horizontal':
                    extent_meters = extent_3857.width()
                else:
                    extent_meters = extent_3857.height()

                return (center_wgs.y(), center_wgs.x(), extent_meters, self.extent_dimension)

            except Exception as e:
                print(f"Error getting alignment info: {e}")
                return None

        return None

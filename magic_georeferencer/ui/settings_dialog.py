"""
Settings Dialog for Magic Georeferencer

Configure plugin settings.

TODO: Implement full settings dialog with:
- Performance settings (GPU/CPU, threads)
- Matching settings (quality presets, thresholds)
- Tile fetching settings (cache size, expiry)
- Georeferencing settings (transform type, resampling)
- Output settings (directory, naming scheme)
"""

from qgis.PyQt.QtWidgets import QDialog, QLabel, QVBoxLayout


class SettingsDialog(QDialog):
    """Plugin settings dialog (placeholder implementation)"""

    def __init__(self, parent=None):
        """Initialize settings dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setWindowTitle("Magic Georeferencer - Settings")
        self.setMinimumSize(600, 400)

        # Placeholder UI
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Settings Dialog - Coming Soon"))
        self.setLayout(layout)

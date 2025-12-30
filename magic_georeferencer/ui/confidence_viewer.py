"""
Confidence Viewer for Magic Georeferencer

Visualizes match results before georeferencing.

TODO: Implement full confidence viewer with:
- Side-by-side image display
- Interactive filtering
- Keypoint visualization
- Quality metrics display
"""

from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QDialogButtonBox, QFileDialog, QMessageBox
)
from qgis.PyQt.QtCore import Qt


class ConfidenceViewer(QDialog):
    """Visualize match results (placeholder implementation)"""

    def __init__(self, match_result, gcps=None, gcp_generator=None, parent=None):
        """Initialize confidence viewer.

        Args:
            match_result: MatchResult object
            gcps: List of Ground Control Points (optional, for export)
            gcp_generator: GCPGenerator instance (optional, for export)
            parent: Parent widget
        """
        super().__init__(parent)

        self.match_result = match_result
        self.gcps = gcps
        self.gcp_generator = gcp_generator
        self.setWindowTitle("Match Results - Review Before Georeferencing")
        self.setMinimumSize(600, 400)

        # Main layout
        layout = QVBoxLayout()

        # Title
        title = QLabel("<h2>Match Results</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Info message
        info = QLabel(
            "<p>The AI has found matching features between your image and the basemap.</p>"
            "<p>Review the statistics below, then click <b>Continue to Georeference</b> "
            "to create the georeferenced output.</p>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Statistics
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(QLabel(f"<b>Total Matches Found:</b> {match_result.num_matches()}"))
        stats_layout.addWidget(QLabel(f"<b>Mean Confidence:</b> {match_result.mean_confidence():.3f}"))
        stats_layout.addWidget(QLabel(f"<b>Geometric Consistency Score:</b> {match_result.geometric_score:.3f}"))
        stats_layout.addWidget(QLabel(f"<b>Spatial Distribution Quality:</b> {match_result.distribution_quality:.3f}"))

        layout.addLayout(stats_layout)

        # Quality assessment
        layout.addWidget(QLabel(""))  # Spacer
        quality_label = QLabel()
        if match_result.mean_confidence() > 0.5 and match_result.geometric_score > 0.6:
            quality_label.setText("✓ <b style='color: green;'>High quality matches - excellent for georeferencing</b>")
        elif match_result.mean_confidence() > 0.3 and match_result.geometric_score > 0.4:
            quality_label.setText("⚠ <b style='color: orange;'>Moderate quality matches - should work but may need refinement</b>")
        else:
            quality_label.setText("⚠ <b style='color: red;'>Low quality matches - georeferencing may be less accurate</b>")
        quality_label.setWordWrap(True)
        layout.addWidget(quality_label)

        # Note about future features
        layout.addWidget(QLabel(""))  # Spacer
        note = QLabel(
            "<i>Note: Future versions will show side-by-side visualizations of the matches "
            "and allow you to interactively adjust the confidence threshold.</i>"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(note)

        # Spacer
        layout.addStretch()

        # GCP Export section
        if self.gcps and self.gcp_generator:
            gcp_layout = QHBoxLayout()
            gcp_info = QLabel(f"<b>Control Points:</b> {len(self.gcps)} GCPs available for export")
            gcp_layout.addWidget(gcp_info)

            export_btn = QPushButton("Export GCPs...")
            export_btn.setToolTip(
                "Save control points to a .points file for later use or manual editing"
            )
            export_btn.clicked.connect(self._export_gcps)
            gcp_layout.addWidget(export_btn)
            gcp_layout.addStretch()

            layout.addLayout(gcp_layout)
            layout.addWidget(QLabel(""))  # Spacer

        # Buttons
        button_box = QDialogButtonBox()

        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_box.addButton(cancel_btn, QDialogButtonBox.RejectRole)

        # Continue button (default, highlighted)
        continue_btn = QPushButton("Continue to Georeference")
        continue_btn.setDefault(True)
        continue_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        continue_btn.clicked.connect(self.accept)
        button_box.addButton(continue_btn, QDialogButtonBox.AcceptRole)

        layout.addWidget(button_box)

        self.setLayout(layout)

    def _export_gcps(self):
        """Export GCPs to a .points file."""
        if not self.gcps or not self.gcp_generator:
            QMessageBox.warning(
                self,
                "Export Error",
                "No control points available to export."
            )
            return

        # Get save location
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Control Points",
            "",
            "QGIS Points Files (*.points);;All Files (*)"
        )

        if not filepath:
            return

        # Ensure .points extension
        if not filepath.endswith('.points'):
            filepath += '.points'

        try:
            self.gcp_generator.export_gcp_file(self.gcps, Path(filepath))
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {len(self.gcps)} control points to:\n{filepath}\n\n"
                "You can load this file in the QGIS Georeferencer tool for manual editing."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export control points:\n{str(e)}"
            )

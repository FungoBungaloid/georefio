"""
Confidence Viewer for Magic Georeferencer

Visualizes match results before georeferencing.

TODO: Implement full confidence viewer with:
- Side-by-side image display
- Interactive filtering
- Keypoint visualization
- Quality metrics display
"""

from qgis.PyQt.QtWidgets import QDialog, QLabel, QVBoxLayout


class ConfidenceViewer(QDialog):
    """Visualize match results (placeholder implementation)"""

    def __init__(self, match_result, parent=None):
        """Initialize confidence viewer.

        Args:
            match_result: MatchResult object
            parent: Parent widget
        """
        super().__init__(parent)

        self.match_result = match_result
        self.setWindowTitle("Match Results - Confidence Viewer")
        self.setMinimumSize(800, 600)

        # Placeholder UI
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Confidence Viewer - Coming Soon"))
        layout.addWidget(QLabel(f"Matches: {match_result.num_matches()}"))
        layout.addWidget(QLabel(f"Mean Confidence: {match_result.mean_confidence():.2f}"))
        layout.addWidget(QLabel(f"Geometric Score: {match_result.geometric_score:.2f}"))
        layout.addWidget(QLabel(f"Distribution Quality: {match_result.distribution_quality:.2f}"))
        self.setLayout(layout)

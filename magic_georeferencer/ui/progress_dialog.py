"""
Progress Dialog for Magic Georeferencer

Shows progress during long operations (download, matching).
"""

from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton
)
from qgis.PyQt.QtCore import Qt, pyqtSignal


class ProgressDialog(QDialog):
    """Progress dialog for long-running operations"""

    # Signal emitted when cancel is clicked
    cancelled = pyqtSignal()

    def __init__(self, parent=None, title="Processing..."):
        """Initialize progress dialog.

        Args:
            parent: Parent widget
            title: Dialog title
        """
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(500, 150)

        # Setup UI
        self._setup_ui()

        # Track cancellation
        self._is_cancelled = False

    def _setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Detail label (size, speed, time remaining)
        self.detail_label = QLabel("")
        self.detail_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detail_label)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def set_status(self, message: str):
        """Set status message.

        Args:
            message: Status message to display
        """
        self.status_label.setText(message)

    def set_progress(self, current: int, total: int):
        """Set progress value.

        Args:
            current: Current progress value
            total: Total progress value
        """
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)

            # Update detail label for downloads
            if current > 0 and total > 0:
                current_mb = current / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                self.detail_label.setText(
                    f"{current_mb:.1f} / {total_mb:.1f} MB ({percentage}%)"
                )
        else:
            # Indeterminate progress
            self.progress_bar.setMaximum(0)
            self.progress_bar.setMinimum(0)

    def set_indeterminate(self, indeterminate: bool = True):
        """Set progress bar to indeterminate mode.

        Args:
            indeterminate: True for indeterminate, False for normal
        """
        if indeterminate:
            self.progress_bar.setMaximum(0)
            self.progress_bar.setMinimum(0)
        else:
            self.progress_bar.setMaximum(100)
            self.progress_bar.setMinimum(0)

    def _on_cancel(self):
        """Handle cancel button click"""
        self._is_cancelled = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        self.cancelled.emit()

    def is_cancelled(self) -> bool:
        """Check if operation was cancelled.

        Returns:
            True if cancelled
        """
        return self._is_cancelled

    def reset(self):
        """Reset dialog to initial state"""
        self._is_cancelled = False
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        self.detail_label.setText("")
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")

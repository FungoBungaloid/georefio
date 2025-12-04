"""
UI components for Magic Georeferencer
"""

from .main_dialog import MagicGeoreferencerDialog
from .progress_dialog import ProgressDialog
from .confidence_viewer import ConfidenceViewer
from .settings_dialog import SettingsDialog
from .batch_dialog import BatchDialog

__all__ = [
    'MagicGeoreferencerDialog',
    'ProgressDialog',
    'ConfidenceViewer',
    'SettingsDialog',
    'BatchDialog'
]

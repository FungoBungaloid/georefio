"""
Main plugin class for Bulk Georeferencer
"""

from pathlib import Path
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox


class BulkGeoreferencer:
    """QGIS Plugin Implementation for Bulk Georeferencer"""

    def __init__(self, iface):
        """Constructor.

        Args:
            iface: An interface instance that will be passed to this class
                which provides the hook by which you can manipulate the QGIS
                application at run time.
        """
        self.iface = iface
        self.plugin_dir = Path(__file__).parent

        # Initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = self.plugin_dir / 'i18n' / f'BulkGeoreferencer_{locale}.qm'

        if locale_path.exists():
            self.translator = QTranslator()
            self.translator.load(str(locale_path))
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr('&Bulk Georeferencer')
        self.toolbar = self.iface.addToolBar('BulkGeoreferencer')
        self.toolbar.setObjectName('BulkGeoreferencer')

        # Plugin dialogs
        self.batch_dialog = None

        # Shared model manager (from magic_georeferencer)
        self._model_manager = None

        # Cache dependency check result
        self._dependencies_checked = False
        self._dependencies_available = False
        self._magic_georeferencer_available = False

    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        return QCoreApplication.translate('BulkGeoreferencer', message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None
    ):
        """Add a toolbar icon to the toolbar."""
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToRasterMenu(
                self.menu,
                action
            )

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        icon_path = str(self.plugin_dir / 'icon.png')

        # Create a default icon if it doesn't exist
        if not Path(icon_path).exists():
            icon_path = ''

        # Batch processing action
        self.add_action(
            icon_path,
            text=self.tr('Bulk Georeference'),
            callback=self.run,
            parent=self.iface.mainWindow(),
            status_tip=self.tr('Batch georeference multiple images using AI location estimation'),
            whats_this=self.tr('Process multiple images automatically using vision AI for location estimation')
        )

        # Settings action
        self.add_action(
            icon_path,
            text=self.tr('Settings'),
            callback=self.show_settings,
            parent=self.iface.mainWindow(),
            add_to_toolbar=False,
            status_tip=self.tr('Configure Bulk Georeferencer settings'),
            whats_this=self.tr('Configure API keys, default settings, and LLM prompts')
        )

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(
                self.tr('&Bulk Georeferencer'),
                action
            )
            self.iface.removeToolBarIcon(action)

        # Remove the toolbar
        del self.toolbar

    def _check_magic_georeferencer(self) -> bool:
        """Check if Magic Georeferencer plugin is available."""
        try:
            # Try to import core components from magic_georeferencer
            from magic_georeferencer.core import ModelManager
            from magic_georeferencer.core import Matcher
            from magic_georeferencer.core import TileFetcher
            from magic_georeferencer.core import GCPGenerator
            from magic_georeferencer.core import Georeferencer
            return True
        except ImportError:
            return False

    def _check_dependencies(self) -> bool:
        """Check all required dependencies."""
        if not self._check_magic_georeferencer():
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Missing Dependency",
                "Bulk Georeferencer requires the Magic Georeferencer plugin.\n\n"
                "Please install Magic Georeferencer from the QGIS Plugin Manager first."
            )
            return False

        # Check Magic Georeferencer's dependencies
        try:
            from magic_georeferencer.dependency_installer import check_and_prompt_install
            return check_and_prompt_install(self.iface.mainWindow())
        except ImportError:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Import Error",
                "Could not load Magic Georeferencer dependency checker.\n\n"
                "Please ensure Magic Georeferencer is properly installed."
            )
            return False

    def run(self):
        """Run the batch georeferencing dialog."""
        # Check dependencies
        if not self._dependencies_checked:
            self._dependencies_available = self._check_dependencies()
            self._dependencies_checked = True

        if not self._dependencies_available:
            return

        try:
            # Import from magic_georeferencer
            from magic_georeferencer.core import ModelManager
            from magic_georeferencer.ui.progress_dialog import ProgressDialog

            # Import our batch dialog
            from .ui.batch_dialog import BulkBatchDialog

            # Initialize model manager if not already done
            if self._model_manager is None:
                self._model_manager = ModelManager()

                # Check if weights need to be downloaded
                if self._model_manager.check_first_run():
                    device_info = self._model_manager.get_device_info()

                    msg_text = (
                        "The AI model weights (~800 MB) need to be downloaded before processing.\n\n"
                    )
                    if device_info.get('cuda_available'):
                        msg_text += f"CUDA GPU detected: {device_info.get('cuda_device_name', 'Unknown')}\n"
                        msg_text += "Will use GPU for faster processing.\n\n"
                    else:
                        msg_text += "No CUDA GPU detected. Will use CPU (slower).\n\n"

                    msg_text += "Would you like to download the model weights now?"

                    result = QMessageBox.question(
                        self.iface.mainWindow(),
                        "Model Weights Required",
                        msg_text,
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    if result != QMessageBox.Yes:
                        return

                    # Download weights with progress dialog
                    progress = ProgressDialog(self.iface.mainWindow(), "Downloading Model Weights")

                    def progress_callback(status, current, total):
                        progress.set_status(status)
                        progress.set_progress(current, total)
                        QCoreApplication.processEvents()

                    progress.show()

                    try:
                        success, message = self._model_manager.download_weights(progress_callback)
                        progress.close()

                        if not success:
                            QMessageBox.critical(
                                self.iface.mainWindow(),
                                "Download Failed",
                                f"Failed to download model weights:\n{message}"
                            )
                            return

                        QMessageBox.information(
                            self.iface.mainWindow(),
                            "Download Complete",
                            "Model weights downloaded successfully!"
                        )

                    except Exception as e:
                        progress.close()
                        QMessageBox.critical(
                            self.iface.mainWindow(),
                            "Download Error",
                            f"Error during download:\n{str(e)}"
                        )
                        return

            # Create the batch dialog if it doesn't exist
            if self.batch_dialog is None:
                self.batch_dialog = BulkBatchDialog(
                    self.iface,
                    self._model_manager,
                    self.iface.mainWindow()
                )

            # Show the dialog
            self.batch_dialog.show()
            self.batch_dialog.exec_()

        except ImportError as e:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Bulk Georeferencer - Import Error",
                f"Failed to load batch processing components:\n\n{str(e)}\n\n"
                "Please ensure Magic Georeferencer is properly installed."
            )

    def show_settings(self):
        """Show the settings dialog."""
        try:
            from .ui.settings_dialog import BulkSettingsDialog

            dialog = BulkSettingsDialog(self.iface.mainWindow())
            dialog.exec_()

        except ImportError as e:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Bulk Georeferencer - Import Error",
                f"Failed to load settings dialog:\n\n{str(e)}"
            )

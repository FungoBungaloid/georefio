"""
Main plugin class for Magic Georeferencer
"""

import os
from pathlib import Path
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.core import QgsApplication


class MagicGeoreferencer:
    """QGIS Plugin Implementation for Magic Georeferencer"""

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
        locale_path = self.plugin_dir / 'i18n' / f'MagicGeoreferencer_{locale}.qm'

        if locale_path.exists():
            self.translator = QTranslator()
            self.translator.load(str(locale_path))
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr('&Magic Georeferencer')
        self.toolbar = self.iface.addToolBar('MagicGeoreferencer')
        self.toolbar.setObjectName('MagicGeoreferencer')

        # Plugin dialogs
        self.dialog = None
        self.batch_dialog = None

        # Shared model manager (cached across dialogs)
        self._model_manager = None

        # Cache dependency check result to avoid repeated checks
        self._dependencies_checked = False
        self._dependencies_available = False

    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        return QCoreApplication.translate('MagicGeoreferencer', message)

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
        """Add a toolbar icon to the toolbar.

        Args:
            icon_path: Path to the icon for this action.
            text: Text that should be shown in menu items for this action.
            callback: Function to be called when the action is triggered.
            enabled_flag: A flag indicating if the action should be enabled by default.
            add_to_menu: Flag indicating whether the action should also be added to the menu.
            add_to_toolbar: Flag indicating whether the action should also be added to the toolbar.
            status_tip: Optional text to show in a popup when mouse pointer hovers over the action.
            whats_this: Optional text to show in the status bar when the mouse pointer hovers over the action.
            parent: Parent widget for the new action.

        Returns:
            The action that was created.
        """
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

        # Main georeferencer action
        self.add_action(
            icon_path,
            text=self.tr('Magic Georeferencer'),
            callback=self.run,
            parent=self.iface.mainWindow(),
            status_tip=self.tr('AI-powered automatic image georeferencing'),
            whats_this=self.tr('Automatically georeference images using AI')
        )

        # Batch processing action
        self.add_action(
            icon_path,
            text=self.tr('Batch Georeference'),
            callback=self.run_batch,
            parent=self.iface.mainWindow(),
            add_to_toolbar=False,  # Only in menu, not toolbar
            status_tip=self.tr('Batch georeference multiple images using AI location estimation'),
            whats_this=self.tr('Process multiple images automatically using vision AI for location estimation')
        )

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(
                self.tr('&Magic Georeferencer'),
                action
            )
            self.iface.removeToolBarIcon(action)

        # Remove the toolbar
        del self.toolbar

    def run(self):
        """Run method that performs all the real work"""
        # Check dependencies before importing (cache result to avoid repeated checks)
        if not self._dependencies_checked:
            from .dependency_installer import check_and_prompt_install
            self._dependencies_available = check_and_prompt_install(self.iface.mainWindow())
            self._dependencies_checked = True

        if not self._dependencies_available:
            # Dependencies not available or user cancelled
            return

        try:
            # Lazy import to speed up plugin loading
            from .ui.main_dialog import MagicGeoreferencerDialog

            # Create the dialog if it doesn't exist
            if self.dialog is None:
                self.dialog = MagicGeoreferencerDialog(self.iface)

            # Show the dialog
            self.dialog.show()
            # Run the dialog event loop
            self.dialog.exec_()

        except ImportError as e:
            # Show error if imports still fail after dependency check
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Magic Georeferencer - Import Error",
                f"Failed to load plugin components:\n\n{str(e)}\n\n"
                "Please restart QGIS after installing dependencies."
            )

    def run_batch(self):
        """Run batch georeferencing dialog"""
        # Check dependencies before importing (cache result to avoid repeated checks)
        if not self._dependencies_checked:
            from .dependency_installer import check_and_prompt_install
            self._dependencies_available = check_and_prompt_install(self.iface.mainWindow())
            self._dependencies_checked = True

        if not self._dependencies_available:
            # Dependencies not available or user cancelled
            return

        try:
            # Lazy import to speed up plugin loading
            from .ui.batch_dialog import BatchDialog
            from .core.model_manager import ModelManager

            # Initialize model manager if not already done
            if self._model_manager is None:
                self._model_manager = ModelManager()

                # Check if weights need to be downloaded
                if self._model_manager.check_first_run():
                    result = QMessageBox.question(
                        self.iface.mainWindow(),
                        "Model Weights Required",
                        "The AI model weights need to be downloaded before batch processing.\n\n"
                        "Would you like to open the main dialog to download them first?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    if result == QMessageBox.Yes:
                        self.run()  # Open main dialog to download weights
                    return

            # Create the batch dialog if it doesn't exist
            if self.batch_dialog is None:
                self.batch_dialog = BatchDialog(
                    self.iface,
                    self._model_manager,
                    self.iface.mainWindow()
                )

            # Show the dialog
            self.batch_dialog.show()
            # Run the dialog event loop
            self.batch_dialog.exec_()

        except ImportError as e:
            # Show error if imports still fail after dependency check
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Magic Georeferencer - Import Error",
                f"Failed to load batch processing components:\n\n{str(e)}\n\n"
                "Please restart QGIS after installing dependencies."
            )

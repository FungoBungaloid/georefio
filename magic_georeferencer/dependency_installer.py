"""
Dependency installer for Magic Georeferencer plugin.

Handles installation of Python packages required by the plugin into QGIS's Python environment.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


class DependencyInstaller:
    """Manages installation of required Python packages."""

    # Required packages with minimum versions
    REQUIRED_PACKAGES = [
        'opencv-python>=4.8.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'transformers>=4.30.0',
        'huggingface-hub>=0.16.0',
        'scipy>=1.9.0',
    ]

    # Package import names (different from pip names in some cases)
    IMPORT_NAMES = {
        'opencv-python': 'cv2',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'transformers': 'transformers',
        'huggingface-hub': 'huggingface_hub',
        'scipy': 'scipy',
    }

    def __init__(self):
        self.missing_packages = []
        self.python_exe = sys.executable

    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Check if all required dependencies are installed.

        Returns:
            (all_installed, missing_packages)
        """
        self.missing_packages = []

        for package_spec in self.REQUIRED_PACKAGES:
            # Extract package name (before >= or ==)
            package_name = package_spec.split('>=')[0].split('==')[0]
            import_name = self.IMPORT_NAMES.get(package_name, package_name)

            try:
                __import__(import_name)
            except ImportError:
                self.missing_packages.append(package_spec)

        return len(self.missing_packages) == 0, self.missing_packages

    def install_dependencies(self, progress_callback=None) -> Tuple[bool, str]:
        """
        Install missing dependencies using pip.

        Args:
            progress_callback: Optional callback function(message: str)

        Returns:
            (success, message)
        """
        if not self.missing_packages:
            return True, "All dependencies already installed"

        try:
            for i, package in enumerate(self.missing_packages, 1):
                if progress_callback:
                    progress_callback(f"Installing {package} ({i}/{len(self.missing_packages)})...")

                # Use pip to install the package
                result = subprocess.run(
                    [self.python_exe, '-m', 'pip', 'install', '--user', package],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per package
                )

                if result.returncode != 0:
                    error_msg = f"Failed to install {package}:\n{result.stderr}"
                    return False, error_msg

            if progress_callback:
                progress_callback("All dependencies installed successfully!")

            return True, "Dependencies installed successfully. Please restart QGIS."

        except subprocess.TimeoutExpired:
            return False, "Installation timed out. Please check your internet connection."
        except Exception as e:
            return False, f"Installation error: {str(e)}"

    def get_installation_instructions(self) -> str:
        """
        Get manual installation instructions for missing packages.

        Returns:
            Formatted instruction string
        """
        if not self.missing_packages:
            return "All dependencies are installed."

        packages_str = ' '.join(self.missing_packages)

        instructions = f"""
Magic Georeferencer requires additional Python packages.

Missing packages:
{chr(10).join(f'  - {pkg}' for pkg in self.missing_packages)}

AUTOMATIC INSTALLATION:
The plugin can install these automatically. Click 'Install Dependencies' when prompted.

MANUAL INSTALLATION:
If automatic installation fails, you can install manually:

Windows:
1. Open OSGeo4W Shell (from Start Menu)
2. Run: python -m pip install --user {packages_str}

macOS/Linux:
1. Open Terminal
2. Find QGIS Python: which python3
3. Run: /path/to/qgis/python3 -m pip install --user {packages_str}

After installation, restart QGIS.
"""
        return instructions


def check_and_prompt_install(parent_widget=None):
    """
    Check dependencies and prompt user to install if needed.

    Args:
        parent_widget: Parent Qt widget for dialogs (optional)

    Returns:
        True if all dependencies available, False otherwise
    """
    installer = DependencyInstaller()
    all_installed, missing = installer.check_dependencies()

    if all_installed:
        return True

    # Try to import Qt for dialog
    try:
        from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog
        from qgis.PyQt.QtCore import Qt

        # Show missing packages dialog
        msg = QMessageBox(parent_widget)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Magic Georeferencer - Missing Dependencies")
        msg.setText("Magic Georeferencer requires additional Python packages.")
        msg.setInformativeText(
            f"Missing packages:\n" +
            "\n".join(f"  • {pkg}" for pkg in missing[:5]) +
            (f"\n  ... and {len(missing)-5} more" if len(missing) > 5 else "")
        )
        msg.setDetailedText(installer.get_installation_instructions())

        install_btn = msg.addButton("Install Dependencies", QMessageBox.AcceptRole)
        manual_btn = msg.addButton("Show Manual Instructions", QMessageBox.HelpRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)

        msg.exec_()
        clicked = msg.clickedButton()

        if clicked == install_btn:
            # Show progress dialog
            progress = QProgressDialog(
                "Installing dependencies...",
                "Cancel",
                0,
                0,  # Indeterminate progress
                parent_widget
            )
            progress.setWindowTitle("Magic Georeferencer")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            def update_progress(message):
                progress.setLabelText(message)
                from qgis.PyQt.QtWidgets import QApplication
                QApplication.processEvents()

            success, message = installer.install_dependencies(update_progress)
            progress.close()

            # Show result
            result_msg = QMessageBox(parent_widget)
            if success:
                result_msg.setIcon(QMessageBox.Information)
                result_msg.setWindowTitle("Installation Complete")
                result_msg.setText("Dependencies installed successfully!")
                result_msg.setInformativeText(
                    "Please restart QGIS for changes to take effect."
                )
            else:
                result_msg.setIcon(QMessageBox.Critical)
                result_msg.setWindowTitle("Installation Failed")
                result_msg.setText("Failed to install dependencies.")
                result_msg.setInformativeText(message)
                result_msg.setDetailedText(installer.get_installation_instructions())

            result_msg.exec_()
            return False  # Require restart

        elif clicked == manual_btn:
            # Show manual instructions
            info_msg = QMessageBox(parent_widget)
            info_msg.setIcon(QMessageBox.Information)
            info_msg.setWindowTitle("Manual Installation Instructions")
            info_msg.setText("Manual Dependency Installation")
            info_msg.setDetailedText(installer.get_installation_instructions())
            info_msg.exec_()
            return False
        else:
            # User cancelled
            return False

    except ImportError:
        # Can't show GUI, print to console
        print("=" * 60)
        print("Magic Georeferencer - Missing Dependencies")
        print("=" * 60)
        print(installer.get_installation_instructions())
        print("=" * 60)
        return False

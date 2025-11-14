"""
Dependency installer for Magic Georeferencer plugin.

Handles installation of Python packages required by the plugin into QGIS's Python environment.
"""

import sys
import subprocess
import os
from pathlib import Path
from typing import List, Tuple
import importlib


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
        self.python_exe = self._find_python_executable()

    def _find_python_executable(self) -> str:
        """
        Find the correct Python executable for QGIS.

        On Windows, sys.executable often points to qgis-bin.exe, not python.exe.
        We need to find the actual Python executable.
        """
        # First, check if sys.executable is actually Python
        if 'python' in os.path.basename(sys.executable).lower():
            return sys.executable

        # Try to find Python in QGIS installation
        # Common Windows QGIS paths
        possible_paths = []

        if sys.platform == 'win32':
            # Get QGIS installation directory
            qgis_exe_dir = Path(sys.executable).parent

            # Common locations for python.exe in QGIS
            possible_paths.extend([
                qgis_exe_dir / 'python.exe',
                qgis_exe_dir / 'python3.exe',
                qgis_exe_dir / '..' / 'bin' / 'python.exe',
                qgis_exe_dir / '..' / 'bin' / 'python3.exe',
                qgis_exe_dir / '..' / '..' / 'bin' / 'python.exe',
                qgis_exe_dir / '..' / '..' / 'bin' / 'python3.exe',
            ])

            # OSGeo4W paths
            for osgeo_root in ['C:\\OSGeo4W64', 'C:\\OSGeo4W', 'C:\\Program Files\\QGIS 3.34', 'C:\\Program Files\\QGIS 3.28']:
                possible_paths.extend([
                    Path(osgeo_root) / 'bin' / 'python.exe',
                    Path(osgeo_root) / 'bin' / 'python3.exe',
                    Path(osgeo_root) / 'apps' / 'Python39' / 'python.exe',
                    Path(osgeo_root) / 'apps' / 'Python310' / 'python.exe',
                ])
        else:
            # macOS/Linux - sys.executable should be fine
            possible_paths = [
                Path(sys.executable),
                Path('/usr/bin/python3'),
            ]

        # Check which Python exists
        for path in possible_paths:
            if path.exists():
                return str(path.resolve())

        # Fallback to sys.executable
        return sys.executable

    def check_dependencies(self, force_recheck=False) -> Tuple[bool, List[str]]:
        """
        Check if all required dependencies are installed.

        Args:
            force_recheck: If True, reload modules to check for newly installed packages

        Returns:
            (all_installed, missing_packages)
        """
        if force_recheck:
            # Reload sys.path to pick up newly installed packages
            importlib.invalidate_caches()

        self.missing_packages = []

        for package_spec in self.REQUIRED_PACKAGES:
            # Extract package name (before >= or ==)
            package_name = package_spec.split('>=')[0].split('==')[0]
            import_name = self.IMPORT_NAMES.get(package_name, package_name)

            try:
                if force_recheck:
                    # Try to reload if already imported
                    if import_name in sys.modules:
                        importlib.reload(sys.modules[import_name])
                    else:
                        __import__(import_name)
                else:
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

        # Verify Python executable
        if not Path(self.python_exe).exists():
            return False, f"Python executable not found: {self.python_exe}\n\nPlease install dependencies manually using OSGeo4W Shell."

        try:
            errors = []

            for i, package in enumerate(self.missing_packages, 1):
                if progress_callback:
                    progress_callback(f"Installing {package} ({i}/{len(self.missing_packages)})...\n\nThis may take several minutes, especially for PyTorch.")

                try:
                    # Use pip to install the package
                    result = subprocess.run(
                        [self.python_exe, '-m', 'pip', 'install', '--user', package],
                        capture_output=True,
                        text=True,
                        timeout=600,  # 10 minute timeout per package (PyTorch is large!)
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )

                    if result.returncode != 0:
                        errors.append(f"{package}: {result.stderr[:200]}")
                        if progress_callback:
                            progress_callback(f"Failed to install {package}. Continuing with others...")
                    else:
                        if progress_callback:
                            progress_callback(f"✓ Successfully installed {package}")

                except subprocess.TimeoutExpired:
                    errors.append(f"{package}: Installation timed out (>10 minutes)")
                except Exception as e:
                    errors.append(f"{package}: {str(e)}")

            if errors:
                return False, "Some packages failed to install:\n\n" + "\n".join(errors) + "\n\nTry manual installation using OSGeo4W Shell."

            if progress_callback:
                progress_callback("All dependencies installed successfully!")

            return True, "Dependencies installed successfully. Please restart QGIS."

        except Exception as e:
            return False, f"Installation error: {str(e)}\n\nTry manual installation using OSGeo4W Shell."

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

RECOMMENDED: Manual Installation via OSGeo4W Shell

Windows:
1. Open "OSGeo4W Shell" from Start Menu (installed with QGIS)
2. Run these commands ONE AT A TIME:

   python -m pip install --user opencv-python>=4.8.0
   python -m pip install --user torch>=2.0.0 torchvision>=0.15.0
   python -m pip install --user transformers>=4.30.0
   python -m pip install --user huggingface-hub>=0.16.0
   python -m pip install --user scipy>=1.9.0

3. Wait for each to complete (PyTorch takes ~5 minutes)
4. Restart QGIS

macOS/Linux:
1. Open Terminal
2. Find QGIS Python path:

   Open QGIS > Plugins > Python Console
   Type: import sys; print(sys.executable)

3. Use that path to install packages:

   /path/to/python -m pip install --user {packages_str}

4. Restart QGIS

TROUBLESHOOTING:
- If you see "python: command not found", make sure OSGeo4W Shell is used (not regular CMD)
- If you see "No module named pip", install pip first: python -m ensurepip
- For network errors, check firewall/proxy settings
- PyTorch download is ~2GB - requires good internet connection

Detected Python: {self.python_exe}
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
        from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog, QApplication
        from qgis.PyQt.QtCore import Qt

        # Show missing packages dialog
        msg = QMessageBox(parent_widget)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Magic Georeferencer - Missing Dependencies")
        msg.setText("Magic Georeferencer requires additional Python packages.")
        msg.setInformativeText(
            f"Missing {len(missing)} package(s):\n\n" +
            "\n".join(f"  • {pkg.split('>=')[0]}" for pkg in missing[:5]) +
            (f"\n  ... and {len(missing)-5} more" if len(missing) > 5 else "") +
            "\n\nWARNING: Automatic installation may not work on all systems.\nManual installation via OSGeo4W Shell is recommended."
        )
        msg.setDetailedText(installer.get_installation_instructions())

        # Make buttons more explicit
        manual_btn = msg.addButton("Show Manual Instructions (Recommended)", QMessageBox.AcceptRole)
        install_btn = msg.addButton("Try Automatic Install (Experimental)", QMessageBox.ActionRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)

        msg.exec_()
        clicked = msg.clickedButton()

        if clicked == install_btn:
            # Show progress dialog
            progress = QProgressDialog(
                "Installing dependencies...\n\nThis may take 10-15 minutes.\nPlease be patient.",
                None,  # No cancel button
                0,
                0,  # Indeterminate progress
                parent_widget
            )
            progress.setWindowTitle("Magic Georeferencer - Installing")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumWidth(400)
            progress.show()

            def update_progress(message):
                progress.setLabelText(message)
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
                    "Please restart QGIS for changes to take effect.\n\n"
                    "After restarting, click the plugin icon again."
                )
            else:
                result_msg.setIcon(QMessageBox.Critical)
                result_msg.setWindowTitle("Installation Failed")
                result_msg.setText("Automatic installation failed.")
                result_msg.setInformativeText(
                    "Please use manual installation instead.\n\n"
                    "Click 'Show Details' for instructions."
                )
                result_msg.setDetailedText(message + "\n\n" + installer.get_installation_instructions())

            result_msg.exec_()
            return False  # Require restart

        elif clicked == manual_btn:
            # Show manual instructions
            info_msg = QMessageBox(parent_widget)
            info_msg.setIcon(QMessageBox.Information)
            info_msg.setWindowTitle("Manual Installation Instructions")
            info_msg.setText("Follow these steps to install dependencies:")
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

# Magic Georeferencer - Installation Guide

## Prerequisites

### System Requirements

- **QGIS**: Version 3.22 or higher
- **Python**: 3.9+ (bundled with QGIS)
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk Space**: ~2 GB (model cache + tiles)

### Optional (for GPU Acceleration)

- **NVIDIA GPU**: CUDA Compute Capability 3.5+
- **CUDA Toolkit**: 11.7 or higher
- **GPU RAM**: 4 GB minimum, 8 GB recommended

## Installation Methods

### Method 1: QGIS Plugin Repository (Recommended)

Once published to the official QGIS plugin repository, this is the easiest method:

1. Open QGIS
2. Go to `Plugins > Manage and Install Plugins`
3. Search for "Magic Georeferencer"
4. Click "Install Plugin"
5. Dependencies will be automatically installed

### Method 2: Install from ZIP (Manual Fallback)

If the plugin is not yet available in the repository, or you need a development version:

#### Step 1: Download the Repository

1. Go to the GitHub repository
2. Click the green "Code" button
3. Select "Download ZIP"
4. Unzip the downloaded file to a temporary location

#### Step 2: Prepare the Plugin ZIP

**Important**: You need to create a ZIP of just the `magic_georeferencer` folder, not the entire repository.

1. Open the extracted folder (e.g., `georefio-main`)
2. Find the `magic_georeferencer` folder inside
3. Create a ZIP file containing **only** the `magic_georeferencer` folder:
   - **Windows**: Right-click `magic_georeferencer` folder > "Send to" > "Compressed (zipped) folder"
   - **macOS**: Right-click `magic_georeferencer` folder > "Compress magic_georeferencer"
   - **Linux**: `cd georefio-main && zip -r magic_georeferencer.zip magic_georeferencer/`

The resulting ZIP file should be named `magic_georeferencer.zip` and when opened should show the `magic_georeferencer` folder at the top level.

#### Step 3: Install in QGIS

1. Open QGIS
2. Go to `Plugins > Manage and Install Plugins`
3. Click on "Install from ZIP" tab
4. Click the "..." button and select your `magic_georeferencer.zip` file
5. Click "Install Plugin"

#### Step 4: Install Dependencies (AUTOMATIC)

When you first click the plugin icon in QGIS:

1. A dialog will appear showing missing Python packages
2. Click **"Install Dependencies"** button
3. Wait for installation to complete (may take 5-10 minutes)
4. **Restart QGIS** when prompted
5. The plugin is now ready to use!

**What gets installed:**
- PyTorch and TorchVision (deep learning framework)
- Transformers and HuggingFace Hub (AI model support)
- OpenCV (image processing)
- SciPy and NumPy (scientific computing)

### Method 3: Manual Dependency Installation

If automatic installation doesn't work, you can install dependencies manually:

#### Windows (OSGeo4W Shell)

1. Open **OSGeo4W Shell** from Start Menu (comes with QGIS)
2. Run the following commands:

```bash
# Navigate to plugin directory
cd "C:\Users\YourName\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\magic_georeferencer"

# Install dependencies
python -m pip install --user -r requirements.txt
```

Or install packages individually:
```bash
python -m pip install --user opencv-python>=4.8.0
python -m pip install --user torch>=2.0.0 torchvision>=0.15.0
python -m pip install --user transformers>=4.30.0 huggingface-hub>=0.16.0
python -m pip install --user scipy>=1.9.0
```

#### macOS/Linux

1. Open Terminal
2. Find QGIS Python path:
```bash
# Method 1: Check QGIS Python
which python3

# Method 2: Check in QGIS Python Console
# Open QGIS > Plugins > Python Console > Type: import sys; print(sys.executable)
```

3. Install dependencies:
```bash
# Replace /path/to/qgis/python3 with actual path
/path/to/qgis/python3 -m pip install --user -r requirements.txt
```

**Alternative: Use QGIS Python Console**
```python
import subprocess
import sys

packages = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'transformers>=4.30.0',
    'huggingface-hub>=0.16.0',
    'scipy>=1.9.0'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
```

#### Step 4: Enable the Plugin

1. Restart QGIS
2. Go to `Plugins � Manage and Install Plugins`
3. Click on "Installed" tab
4. Find "Magic Georeferencer" and check the box to enable it

## First Run Setup

### Model Download

On first use, the plugin will prompt you to download the AI model weights (~400 MB):

1. Click "Yes" when prompted to download
2. Wait for the download to complete (may take several minutes)
3. The model will be cached in the plugin's `weights` directory

### GPU Detection

The plugin automatically detects CUDA-capable GPUs. If found:
-  GPU acceleration will be used (much faster)
- Model will use ~4 GB of VRAM

If no GPU is found:
- CPU mode will be used (slower but functional)
- Processing time will be longer

## Verification

### Check Installation

1. Open QGIS
2. Look for the Magic Georeferencer icon in the toolbar
3. Or go to `Raster � Magic Georeferencer`

### Test the Plugin

1. Click the plugin icon
2. The dialog should open without errors
3. Check the status bar - it should say "Ready" after model loading

## Troubleshooting Installation

### Plugin Not Appearing

**Problem:** Plugin doesn't show up in QGIS

**Solutions:**
1. Check the plugin directory path is correct
2. Ensure the folder is named exactly `magic_georeferencer`
3. Check QGIS error log: `Plugins � Python Console � Show Errors`
4. Verify QGIS version is 3.22 or higher

### Import Errors

**Problem:** "ModuleNotFoundError: No module named 'torch'"

**Solutions:**
1. Install dependencies in QGIS Python environment (see Step 3 above)
2. Check Python path in QGIS: `Settings � Options � System � Environment`
3. Try installing with pip in QGIS Python Console

### CUDA Not Detected

**Problem:** Plugin says "No CUDA GPU detected" even though you have one

**Solutions:**
1. Install CUDA Toolkit 11.7+
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Restart QGIS
4. Verify CUDA in Python Console:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

### Model Download Fails

**Problem:** Model download hangs or fails

**Solutions:**
1. Check internet connection
2. Try downloading again (downloads can resume)
3. Manual download:
   - Go to https://huggingface.co/zju-community/matchanything_eloftr
   - Download model files manually
   - Place in `magic_georeferencer/weights/` directory
4. Check firewall/proxy settings

### Permission Errors

**Problem:** "Permission denied" when accessing plugin directory

**Solutions:**
1. Run QGIS as administrator (Windows) or with sudo (Linux)
2. Check directory permissions
3. Install to user directory instead of system directory

## Uninstallation

### Using Plugin Manager

1. Go to `Plugins � Manage and Install Plugins`
2. Find "Magic Georeferencer"
3. Click "Uninstall Plugin"

### Manual Uninstallation

1. Close QGIS
2. Delete the `magic_georeferencer` folder from the plugins directory
3. Optionally remove cached model weights (can be several GB):
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\magic_georeferencer\weights`
   - macOS/Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/magic_georeferencer/weights`

## Updating the Plugin

### Via Plugin Manager (When Published)

1. Go to `Plugins � Manage and Install Plugins`
2. Click "Upgradeable" tab
3. Find "Magic Georeferencer" if an update is available
4. Click "Upgrade Plugin"

### Manual Update

1. Backup your current installation
2. Remove old version
3. Install new version following Method 2 above
4. Model weights will be preserved (no need to re-download)

## Getting Help

If you encounter issues:

1. **Check the logs**: `Plugins � Python Console` in QGIS
2. **Read the documentation**: See `USAGE.md` and `README.md`
3. **Report issues**: https://github.com/yourusername/georefio/issues
4. **Ask for help**: QGIS community forums or plugin GitHub discussions

## System-Specific Notes

### Windows

- Use OSGeo4W Shell for command-line operations
- May need to run as Administrator for first installation
- Antivirus may flag the download - add exception if needed

### macOS

- May need to allow Python to access files in System Preferences � Security
- Use Homebrew Python if QGIS Python has issues
- M1/M2 Macs: Use CPU mode (PyTorch MPS support coming soon)

### Linux

- Install via package manager when available: `apt install qgis-plugin-magic-georeferencer`
- Some distributions may need additional dependencies
- Check SELinux permissions if installation fails

## Next Steps

After successful installation, see `USAGE.md` for how to use the plugin to georeference your images!

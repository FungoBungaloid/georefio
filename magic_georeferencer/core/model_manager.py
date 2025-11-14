"""
Model Manager for Magic Georeferencer

Handles model weight download, CUDA detection, and model loading.
"""

import os
import hashlib
import json
import requests
import zipfile
from pathlib import Path
from typing import Callable, Optional, Dict, Tuple
from urllib.parse import urlparse

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelManager:
    """Manages MatchAnything model weights and loading"""

    def __init__(self, weights_dir: Optional[Path] = None):
        """Initialize ModelManager.

        Args:
            weights_dir: Directory to store model weights. If None, uses default.
        """
        if weights_dir is None:
            # Default to plugin directory / weights
            plugin_dir = Path(__file__).parent.parent
            self.weights_dir = plugin_dir / 'weights'
        else:
            self.weights_dir = Path(weights_dir)

        # Ensure weights directory exists
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Load settings
        config_path = Path(__file__).parent.parent / 'config' / 'default_settings.json'
        with open(config_path, 'r') as f:
            self.settings = json.load(f)

        self.device = None
        self.model = None
        self.weights_url = self.settings['model']['weights_url']

    def is_cuda_available(self) -> bool:
        """Check if CUDA-capable GPU is available.

        Returns:
            True if CUDA is available, False otherwise
        """
        if not TORCH_AVAILABLE:
            return False

        return torch.cuda.is_available()

    def get_device_info(self) -> Dict[str, any]:
        """Get detailed device information.

        Returns:
            Dictionary with device information
        """
        info = {
            'torch_available': TORCH_AVAILABLE,
            'cuda_available': False,
            'cuda_device_count': 0,
            'cuda_device_name': None,
            'recommended_device': 'cpu'
        }

        if TORCH_AVAILABLE:
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available']:
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['recommended_device'] = 'cuda'

        return info

    def weights_exist(self) -> bool:
        """Check if model weights are already downloaded.

        Returns:
            True if weights exist, False otherwise
        """
        # Check for common weight file patterns
        weight_files = [
            'model_checkpoint.pth',
            'model.pth',
            'checkpoint.pth',
            'weights.pth'
        ]

        for weight_file in weight_files:
            if (self.weights_dir / weight_file).exists():
                return True

        # Also check if directory has any .pth files
        pth_files = list(self.weights_dir.glob('*.pth'))
        return len(pth_files) > 0

    def download_weights(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[bool, str]:
        """Download model weights from HuggingFace.

        Args:
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Download weights.zip
            zip_path = self.weights_dir / 'weights.zip'

            # Check if already exists
            if zip_path.exists():
                # Verify if it's a complete download by checking size
                expected_size = self.settings['model']['weights_size_mb'] * 1024 * 1024
                actual_size = zip_path.stat().st_size

                if actual_size < expected_size * 0.9:  # Allow 10% variance
                    # Incomplete download, remove it
                    zip_path.unlink()

            # Download with progress
            response = requests.get(self.weights_url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback:
                            progress_callback(downloaded, total_size)

            # Verify download (basic size check)
            if total_size > 0 and downloaded < total_size * 0.9:
                return False, f"Download incomplete: {downloaded}/{total_size} bytes"

            # Extract weights
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.weights_dir)

            # Cleanup zip file
            zip_path.unlink()

            return True, "Weights downloaded successfully"

        except requests.RequestException as e:
            return False, f"Download failed: {str(e)}"
        except zipfile.BadZipFile as e:
            return False, f"Invalid zip file: {str(e)}"
        except Exception as e:
            return False, f"Error downloading weights: {str(e)}"

    def load_model(self) -> Tuple[bool, str]:
        """Load MatchAnything model into memory.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not TORCH_AVAILABLE:
            return False, "PyTorch is not installed. Please install torch to use this plugin."

        if not self.weights_exist():
            return False, "Model weights not found. Please download them first."

        try:
            # Auto-detect device
            if self.is_cuda_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'

            # Import MatchAnything inference wrapper
            from ..matchanything.inference import MatchAnythingInference

            # Load model
            self.model = MatchAnythingInference(
                weights_path=self.weights_dir,
                device=self.device
            )

            return True, f"Model loaded successfully on {self.device}"

        except ImportError as e:
            return False, f"Failed to import MatchAnything: {str(e)}"
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    def get_inference_config(self) -> Dict[str, any]:
        """Get device-appropriate inference configuration.

        Returns:
            Dictionary with inference configuration
        """
        if self.device == 'cuda':
            return {
                'size': 1024,
                'num_keypoints': 2048,
                'device': 'cuda'
            }
        else:
            return {
                'size': 512,
                'num_keypoints': 512,
                'device': 'cpu'
            }

    def unload_model(self):
        """Unload model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None

        if TORCH_AVAILABLE and self.device == 'cuda':
            torch.cuda.empty_cache()

    def check_first_run(self) -> bool:
        """Check if this is the first run (weights not downloaded).

        Returns:
            True if first run (weights missing), False otherwise
        """
        return not self.weights_exist()

    def get_weights_info(self) -> Dict[str, any]:
        """Get information about model weights.

        Returns:
            Dictionary with weights information
        """
        info = {
            'exists': self.weights_exist(),
            'directory': str(self.weights_dir),
            'url': self.weights_url,
            'expected_size_mb': self.settings['model']['weights_size_mb']
        }

        if info['exists']:
            # Calculate actual size
            total_size = sum(
                f.stat().st_size
                for f in self.weights_dir.rglob('*')
                if f.is_file()
            )
            info['actual_size_mb'] = total_size / (1024 * 1024)

        return info

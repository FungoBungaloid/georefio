"""
Model Manager for Magic Georeferencer

Handles model weight download, CUDA detection, and model loading using HuggingFace Hub.
"""

import json
from pathlib import Path
from typing import Callable, Optional, Dict, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ModelManager:
    """Manages MatchAnything-ELoFTR model weights and loading via HuggingFace"""

    MODEL_REPO = "zju-community/matchanything_eloftr"

    def __init__(self, weights_dir: Optional[Path] = None):
        """Initialize ModelManager.

        Args:
            weights_dir: Directory to store model cache. If None, uses default.
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
        with open(config_path, 'r', encoding='utf-8') as f:
            self.settings = json.load(f)

        self.device = None
        self.model = None
        self.processor = None

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
        """Check if model weights are already cached.

        Returns:
            True if weights exist in HuggingFace cache, False otherwise
        """
        # Check if HuggingFace cache has the model
        # The cache will be in weights_dir with HuggingFace structure
        if not self.weights_dir.exists():
            return False

        # Look for HuggingFace model files (config.json, pytorch_model.bin, etc.)
        has_config = False
        has_weights = False

        # Recursively search cache directory
        for path in self.weights_dir.rglob('*'):
            if path.name == 'config.json':
                has_config = True
            if path.name.startswith('pytorch_model') or path.name.startswith('model'):
                if path.suffix in ['.bin', '.safetensors', '.pth', '.pt']:
                    has_weights = True

        return has_config and has_weights

    def download_weights(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Tuple[bool, str]:
        """Download model weights from HuggingFace Hub.

        Args:
            progress_callback: Optional callback function(status, current, total) for progress updates
                             status: string describing current operation
                             current: bytes downloaded so far
                             total: total bytes to download

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not TRANSFORMERS_AVAILABLE:
            return False, "Transformers library not installed. Please install: pip install transformers"

        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import tqdm as hf_tqdm

            # Download model and processor using transformers
            # This will download to cache_dir automatically
            print(f"Downloading model from HuggingFace: {self.MODEL_REPO}")
            print(f"Cache directory: {self.weights_dir}")

            if progress_callback:
                # Use snapshot_download with progress tracking
                # This downloads all model files at once
                progress_callback("Downloading model files...", 0, 100)

                # We'll use a custom progress wrapper
                class ProgressWrapper:
                    def __init__(self, callback):
                        self.callback = callback
                        self.total_files = 0
                        self.completed_files = 0

                    def __call__(self, filename):
                        self.completed_files += 1
                        if self.total_files > 0:
                            percent = int((self.completed_files / self.total_files) * 100)
                            self.callback(f"Downloading {filename}...", percent, 100)

                wrapper = ProgressWrapper(progress_callback)

                # Download using snapshot_download for better progress tracking
                snapshot_download(
                    repo_id=self.MODEL_REPO,
                    cache_dir=self.weights_dir,
                    resume_download=True
                )

                progress_callback("Download complete", 100, 100)
            else:
                # Simple download without progress
                # Download processor (smaller, downloads first)
                AutoImageProcessor.from_pretrained(
                    self.MODEL_REPO,
                    cache_dir=self.weights_dir
                )

                # Download model (larger)
                AutoModel.from_pretrained(
                    self.MODEL_REPO,
                    cache_dir=self.weights_dir
                )

            return True, f"Model downloaded successfully from {self.MODEL_REPO}"

        except Exception as e:
            return False, f"Failed to download model from HuggingFace: {str(e)}"

    def load_model(self) -> Tuple[bool, str]:
        """Load MatchAnything-ELoFTR model from HuggingFace cache.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not TORCH_AVAILABLE:
            return False, "PyTorch is not installed. Please install torch to use this plugin."

        if not TRANSFORMERS_AVAILABLE:
            return False, "Transformers library not installed. Please install: pip install transformers"

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

            # Load model using HuggingFace transformers
            self.model = MatchAnythingInference(
                weights_path=self.weights_dir,
                device=self.device
            )

            return True, f"Model loaded successfully on {self.device}"

        except ImportError as e:
            return False, f"Failed to import dependencies: {str(e)}"
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
            'repo': self.MODEL_REPO,
            'expected_size_mb': self.settings['model']['estimated_size_mb']
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

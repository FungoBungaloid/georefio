"""
MatchAnything-ELoFTR Inference Wrapper for Magic Georeferencer

Inference interface using HuggingFace Transformers for the MatchAnything-ELoFTR model.
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image

try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class MatchAnythingInference:
    """MatchAnything-ELoFTR inference wrapper using HuggingFace Transformers"""

    MODEL_REPO = "zju-community/matchanything_eloftr"

    def __init__(self, weights_path: Path, device: str = 'auto'):
        """Initialize MatchAnything-ELoFTR model.

        Args:
            weights_path: Path to model cache directory (HuggingFace cache)
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Please install: pip install transformers")

        self.weights_path = Path(weights_path)

        # Auto-detect device if needed
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model from HuggingFace
        self.processor, self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        """Load MatchAnything-ELoFTR model from HuggingFace Hub.

        Returns:
            Tuple of (processor, model)
        """
        try:
            # Load image processor and model from HuggingFace
            print(f"Loading model from HuggingFace: {self.MODEL_REPO}")
            print(f"Cache directory: {self.weights_path}")

            processor = AutoImageProcessor.from_pretrained(
                self.MODEL_REPO,
                cache_dir=self.weights_path
            )

            model = AutoModel.from_pretrained(
                self.MODEL_REPO,
                cache_dir=self.weights_path
            )

            print(f"Model loaded successfully")
            return processor, model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from HuggingFace: {e}")

    def match_images(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        max_keypoints: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match two images and return corresponding keypoints.

        Args:
            image1: First image [H, W, 3] RGB numpy array
            image2: Second image [H, W, 3] RGB numpy array
            max_keypoints: Maximum number of keypoints to detect

        Returns:
            Tuple of (keypoints1, keypoints2, confidence):
            - keypoints1: [N, 2] coordinates in image1 (x, y)
            - keypoints2: [N, 2] coordinates in image2 (x, y)
            - confidence: [N] confidence score per match (0-1)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot perform matching.")

        # Preprocess images
        img1_tensor = self._preprocess_image(image1)
        img2_tensor = self._preprocess_image(image2)

        # Run inference
        with torch.no_grad():
            # TODO: Complete implementation based on ELoFTR/MatchAnything API
            # The exact inference API needs to be determined from the model documentation

            # Expected approach (to be confirmed):
            # inputs = self.processor([image1, image2], return_tensors="pt").to(self.device)
            # outputs = self.model(**inputs)
            # keypoints1 = outputs.keypoints0  # or similar
            # keypoints2 = outputs.keypoints1
            # confidence = outputs.confidence

            # Placeholder: return dummy results for initial testing
            # This will be replaced when integrating actual ELoFTR inference
            print("WARNING: Using placeholder match_images implementation")
            print("TODO: Implement actual ELoFTR inference based on model API")

            num_matches = min(max_keypoints, 100)
            keypoints1 = np.random.rand(num_matches, 2) * np.array([image1.shape[1], image1.shape[0]])
            keypoints2 = np.random.rand(num_matches, 2) * np.array([image2.shape[1], image2.shape[0]])
            confidence = np.random.rand(num_matches)

        return keypoints1, keypoints2, confidence

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for ELoFTR inference using HuggingFace processor.

        Args:
            image: Input image as numpy array [H, W, 3] in RGB format

        Returns:
            Preprocessed image tensor ready for model input
        """
        # Use the HuggingFace processor for preprocessing
        # Convert numpy to PIL Image for processor
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            # Ensure it's in the correct format (uint8, RGB)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        # Use processor to preprocess
        # Note: Actual preprocessing may need adjustment based on model requirements
        inputs = self.processor(images=pil_image, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def estimate_homography(
        self,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        confidence: np.ndarray,
        ransac_threshold: float = 3.0
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Estimate homography matrix using RANSAC.

        Args:
            keypoints1: [N, 2] keypoints from image 1
            keypoints2: [N, 2] keypoints from image 2
            confidence: [N] confidence scores
            ransac_threshold: RANSAC reprojection threshold in pixels

        Returns:
            Tuple of (H, inlier_mask):
            - H: [3, 3] homography matrix (or None if failed)
            - inlier_mask: [N] boolean mask of inliers
        """
        if len(keypoints1) < 4:
            # Need at least 4 points for homography
            return None, np.zeros(len(keypoints1), dtype=bool)

        try:
            # Use OpenCV's RANSAC homography estimation
            H, mask = cv2.findHomography(
                keypoints1,
                keypoints2,
                cv2.RANSAC,
                ransac_threshold
            )

            if H is None:
                return None, np.zeros(len(keypoints1), dtype=bool)

            inlier_mask = mask.ravel().astype(bool)
            return H, inlier_mask

        except cv2.error as e:
            print(f"Homography estimation failed: {e}")
            return None, np.zeros(len(keypoints1), dtype=bool)

    def resize_image(
        self,
        image: np.ndarray,
        target_size: int,
        maintain_aspect: bool = True
    ) -> Tuple[np.ndarray, float]:
        """Resize image to target size.

        Args:
            image: Input image [H, W, 3]
            target_size: Target size (longest edge)
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Tuple of (resized_image, scale_factor)
        """
        h, w = image.shape[:2]

        if maintain_aspect:
            # Scale based on longest edge
            scale = target_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
        else:
            new_h = target_size
            new_w = target_size
            scale = target_size / max(h, w)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return resized, scale

    def scale_keypoints(
        self,
        keypoints: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """Scale keypoint coordinates.

        Args:
            keypoints: [N, 2] keypoint coordinates
            scale: Scale factor

        Returns:
            Scaled keypoints [N, 2]
        """
        return keypoints * scale

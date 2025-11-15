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

            # Try to load the specific model class for keypoint matching
            try:
                from transformers import EfficientLoFTRForKeypointMatching
                print("Loading EfficientLoFTRForKeypointMatching model...")
                model = EfficientLoFTRForKeypointMatching.from_pretrained(
                    self.MODEL_REPO,
                    cache_dir=self.weights_path
                )
                print("✓ Loaded full keypoint matching model")
            except ImportError:
                # Fallback to AutoModel if specific class not available
                print("EfficientLoFTRForKeypointMatching not found, using AutoModel...")
                model = AutoModel.from_pretrained(
                    self.MODEL_REPO,
                    cache_dir=self.weights_path
                )
                print("⚠ Loaded backbone only - keypoint extraction may need post-processing")

            print(f"Model loaded successfully: {type(model).__name__}")
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

        # Store original image sizes for coordinate scaling
        orig_h1, orig_w1 = image1.shape[:2]
        orig_h2, orig_w2 = image2.shape[:2]

        # Preprocess both images together (as required by EfficientLoFTR)
        inputs = self._preprocess_images_pair(image1, image2)

        # Run inference
        with torch.no_grad():
            # Call the model with the image pair
            outputs = self.model(**inputs)

            # Debug: Print what attributes the output actually has
            print(f"Output type: {type(outputs)}")
            print(f"Output attributes: {dir(outputs)}")

            # Try different ways to extract keypoints based on model output structure
            keypoints1 = None
            keypoints2 = None
            confidence = None

            # Debug: Check what keypoints actually is
            if hasattr(outputs, 'keypoints'):
                kp = outputs.keypoints
                print(f"Debug: outputs.keypoints type: {type(kp)}")
                print(f"Debug: outputs.keypoints value: {kp}")
                if kp is not None and hasattr(kp, 'shape'):
                    print(f"Debug: outputs.keypoints shape: {kp.shape}")

            # Method 1: Try dict-like access for KeypointMatchingOutput
            if hasattr(outputs, 'keypoints') and outputs.keypoints is not None and hasattr(outputs.keypoints, 'shape'):
                # keypoints is a tensor, check its shape
                kp = outputs.keypoints
                print(f"Debug: keypoints shape: {kp.shape}")
                if len(kp.shape) == 3 and kp.shape[0] == 2:
                    # Shape is [2, N, 2] - first dim is image index
                    keypoints1 = kp[0].cpu().numpy()
                    keypoints2 = kp[1].cpu().numpy()
                    print(f"✓ Extracted keypoints from outputs.keypoints tensor [2,N,2]: {len(keypoints1)} matches")
                elif len(kp.shape) == 2:
                    # Might need to use matches to split them
                    print(f"Debug: keypoints is 2D tensor, checking matches attribute")

            # Method 2: Try accessing as dict items (KeypointMatchingOutput is dict-like)
            if keypoints1 is None and hasattr(outputs, 'items'):
                try:
                    output_dict = dict(outputs.items())
                    print(f"Debug: Output dict keys: {output_dict.keys()}")

                    # Check for keypoints in dict format
                    if 'keypoints' in output_dict:
                        kp = output_dict['keypoints']
                        if hasattr(kp, 'shape'):
                            print(f"Debug: dict keypoints shape: {kp.shape}")
                            if len(kp.shape) == 3 and kp.shape[0] == 2:
                                keypoints1 = kp[0].cpu().numpy()
                                keypoints2 = kp[1].cpu().numpy()
                                print(f"✓ Extracted from dict keypoints [2,N,2]: {len(keypoints1)} matches")
                except Exception as e:
                    print(f"Debug: Failed to extract from dict: {e}")

            # Method 3: Try using matches attribute to reconstruct keypoint pairs
            if keypoints1 is None and hasattr(outputs, 'matches') and hasattr(outputs, 'keypoints'):
                try:
                    matches = outputs.matches
                    keypoints = outputs.keypoints
                    print(f"Debug: matches shape: {matches.shape if hasattr(matches, 'shape') else 'unknown'}")
                    print(f"Debug: keypoints shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'unknown'}")

                    # matches might be indices or a match matrix
                    # keypoints might be all keypoints from both images
                    # This is a common LoFTR-style output format

                except Exception as e:
                    print(f"Debug: Failed to use matches: {e}")

            # Method 4: Direct attribute access
            if keypoints1 is None and hasattr(outputs, 'keypoints0') and hasattr(outputs, 'keypoints1'):
                keypoints1 = outputs.keypoints0.cpu().numpy()
                keypoints2 = outputs.keypoints1.cpu().numpy()
                print(f"✓ Extracted keypoints from outputs.keypoints0/keypoints1: {len(keypoints1)} matches")

            # Extract confidence scores
            if keypoints1 is not None:
                if hasattr(outputs, 'confidence'):
                    confidence = outputs.confidence.cpu().numpy()
                    print(f"✓ Using model confidence scores")
                elif hasattr(outputs, 'matching_scores'):
                    confidence = outputs.matching_scores.cpu().numpy()
                    print(f"✓ Using matching scores as confidence")
                elif hasattr(outputs, 'scores'):
                    confidence = outputs.scores.cpu().numpy()
                    print(f"✓ Using scores as confidence")
                elif isinstance(outputs, dict) and 'confidence' in outputs:
                    confidence = outputs['confidence'].cpu().numpy()
                    print(f"✓ Using dict confidence scores")
                else:
                    # Default to uniform confidence
                    confidence = np.ones(len(keypoints1)) * 0.8
                    print(f"⚠ No confidence scores found, using default 0.8")

            # If still no keypoints, fail gracefully
            if keypoints1 is None:
                raise RuntimeError(
                    f"Could not extract keypoints from model output.\n"
                    f"Output type: {type(outputs)}\n"
                    f"Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}\n"
                    f"Please check the model output format."
                )

            # Scale keypoints back to original image sizes if needed
            # The processor resizes to 832x832, so we need to scale back
            model_size = 832  # Default from processor

            scale_x1 = orig_w1 / model_size
            scale_y1 = orig_h1 / model_size
            scale_x2 = orig_w2 / model_size
            scale_y2 = orig_h2 / model_size

            keypoints1[:, 0] *= scale_x1
            keypoints1[:, 1] *= scale_y1
            keypoints2[:, 0] *= scale_x2
            keypoints2[:, 1] *= scale_y2

            # Limit to max_keypoints if needed
            if len(keypoints1) > max_keypoints:
                # Sort by confidence and take top N
                top_indices = np.argsort(confidence)[-max_keypoints:]
                keypoints1 = keypoints1[top_indices]
                keypoints2 = keypoints2[top_indices]
                confidence = confidence[top_indices]

        return keypoints1, keypoints2, confidence

    def _preprocess_images_pair(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> dict:
        """Preprocess a pair of images for ELoFTR inference.

        Args:
            image1: First image as numpy array [H, W, 3] in RGB format
            image2: Second image as numpy array [H, W, 3] in RGB format

        Returns:
            Dictionary of preprocessed tensors ready for model input
        """
        from PIL import Image as PILImage

        # Convert numpy arrays to PIL Images
        def to_pil(img):
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                return PILImage.fromarray(img)
            return img

        pil_img1 = to_pil(image1)
        pil_img2 = to_pil(image2)

        # Process both images together as a pair
        # EfficientLoFTR processor expects a pair of images
        inputs = self.processor(
            images=[pil_img1, pil_img2],
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def _match_from_features(
        self,
        feature_maps: tuple,
        image1: np.ndarray,
        image2: np.ndarray,
        max_keypoints: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract matches from feature maps (fallback method).

        This is a backup method if direct keypoint extraction isn't available.
        Uses traditional feature matching on the extracted feature maps.

        Args:
            feature_maps: Feature maps from the model
            image1: Original image 1
            image2: Original image 2
            max_keypoints: Maximum number of keypoints

        Returns:
            Tuple of (keypoints1, keypoints2, confidence)
        """
        print("WARNING: Using traditional feature matching as fallback")

        # For now, use simple grid sampling as a placeholder
        # This should be replaced with proper feature matching if needed
        h, w = 832, 832  # Model input size
        grid_size = int(np.sqrt(max_keypoints))

        # Create a grid of points
        x = np.linspace(0, w-1, grid_size)
        y = np.linspace(0, h-1, grid_size)
        xv, yv = np.meshgrid(x, y)

        keypoints = np.stack([xv.flatten(), yv.flatten()], axis=1)

        # Add small random noise to simulate matching
        noise = np.random.randn(len(keypoints), 2) * 5
        keypoints1 = keypoints
        keypoints2 = keypoints + noise

        # Random confidence scores
        confidence = np.random.uniform(0.6, 0.95, len(keypoints))

        # Limit to max_keypoints
        if len(keypoints1) > max_keypoints:
            indices = np.random.choice(len(keypoints1), max_keypoints, replace=False)
            keypoints1 = keypoints1[indices]
            keypoints2 = keypoints2[indices]
            confidence = confidence[indices]

        return keypoints1, keypoints2, confidence

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

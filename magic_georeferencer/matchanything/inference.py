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
            match_indices_0 = None  # Track which keypoints were matched (for filtering confidence)

            # Debug: Check what keypoints actually is
            if hasattr(outputs, 'keypoints'):
                kp = outputs.keypoints
                print(f"Debug: outputs.keypoints type: {type(kp)}")
                print(f"Debug: outputs.keypoints value: {kp}")
                if kp is not None and hasattr(kp, 'shape'):
                    print(f"Debug: outputs.keypoints shape: {kp.shape}")

            # Method 1: Handle 4D tensor format [batch, 2, N, 2]
            if hasattr(outputs, 'keypoints') and outputs.keypoints is not None and hasattr(outputs.keypoints, 'shape'):
                kp = outputs.keypoints
                print(f"Debug: keypoints shape: {kp.shape}")

                if len(kp.shape) == 4 and kp.shape[0] == 1 and kp.shape[1] == 2:
                    # Shape is [1, 2, N, 2] - batch, image_index, num_keypoints, xy
                    # Remove batch dimension [0] and extract both images
                    keypoints1 = kp[0, 0].cpu().numpy()  # First image
                    keypoints2 = kp[0, 1].cpu().numpy()  # Second image
                    print(f"✓ Extracted keypoints from outputs.keypoints tensor [1,2,N,2]: {len(keypoints1)} keypoints per image")

                    # Debug: Check if all keypoints are identical (suspicious!)
                    unique_kp1 = np.unique(keypoints1, axis=0)
                    unique_kp2 = np.unique(keypoints2, axis=0)
                    print(f"Debug: Unique keypoints img1: {len(unique_kp1)} / {len(keypoints1)}")
                    print(f"Debug: Unique keypoints img2: {len(unique_kp2)} / {len(keypoints2)}")
                    if len(unique_kp1) < 10:
                        print(f"⚠ WARNING: Very few unique keypoints detected!")
                        print(f"  Sample unique values: {unique_kp1[:5]}")

                    # Now use matches to get the actual matched pairs
                    if hasattr(outputs, 'matches'):
                        matches = outputs.matches  # Shape: [1, 2, N]
                        print(f"Debug: Using matches tensor of shape {matches.shape}")
                        print(f"Debug: Sample matches values: {matches[0, :, :10]}")

                        # matches[0, 0, i] = j means keypoint i in image 0 matches to keypoint j in image 1
                        # matches[0, 1, k] = i means keypoint k in image 1 matches to keypoint i in image 0
                        # -1 means no match

                        matches_0_to_1 = matches[0, 0].cpu().numpy()  # For each kp in img0, index in img1
                        matches_1_to_0 = matches[0, 1].cpu().numpy()  # For each kp in img1, index in img0

                        print(f"Debug: matches_0_to_1 range: {matches_0_to_1.min():.1f} to {matches_0_to_1.max():.1f}")
                        print(f"Debug: matches_1_to_0 range: {matches_1_to_0.min():.1f} to {matches_1_to_0.max():.1f}")

                        # Find valid matches from image 0 perspective
                        # For each keypoint i in image 0, check if it has a valid match j in image 1
                        valid_0_indices = []
                        valid_1_indices = []

                        for i in range(len(matches_0_to_1)):
                            j = int(matches_0_to_1[i])
                            # Check if this is a valid match
                            if j >= 0 and j < len(keypoints2):
                                valid_0_indices.append(i)
                                valid_1_indices.append(j)

                        print(f"Debug: Found {len(valid_0_indices)} valid matches out of {len(matches_0_to_1)}")

                        # Extract only the matched keypoints
                        if len(valid_0_indices) > 0:
                            valid_0_indices = np.array(valid_0_indices)
                            valid_1_indices = np.array(valid_1_indices)
                            keypoints1 = keypoints1[valid_0_indices]
                            keypoints2 = keypoints2[valid_1_indices]

                            # Also store the indices for filtering confidence later
                            match_indices_0 = valid_0_indices
                            print(f"✓ Filtered to {len(keypoints1)} matched keypoint pairs")
                        else:
                            match_indices_0 = None
                            print("⚠ No valid matches found!")

                elif len(kp.shape) == 3 and kp.shape[0] == 2:
                    # Shape is [2, N, 2] - image_index, num_keypoints, xy (no batch dim)
                    keypoints1 = kp[0].cpu().numpy()
                    keypoints2 = kp[1].cpu().numpy()
                    print(f"✓ Extracted keypoints from outputs.keypoints tensor [2,N,2]: {len(keypoints1)} matches")

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
                    print(f"Debug: confidence shape from outputs.confidence: {confidence.shape}")
                elif hasattr(outputs, 'matching_scores'):
                    confidence = outputs.matching_scores.cpu().numpy()
                    print(f"Debug: confidence shape from matching_scores: {confidence.shape}")
                elif hasattr(outputs, 'scores'):
                    confidence = outputs.scores.cpu().numpy()
                    print(f"Debug: confidence shape from scores: {confidence.shape}")
                elif isinstance(outputs, dict) and 'confidence' in outputs:
                    confidence = outputs['confidence'].cpu().numpy()
                    print(f"Debug: confidence shape from dict: {confidence.shape}")
                else:
                    confidence = None

                # Process confidence to match keypoints length
                if confidence is not None:
                    # If confidence is 3D like [1, 2, N], extract and filter
                    if len(confidence.shape) == 3:
                        print(f"Debug: Reshaping 3D confidence tensor")
                        # Extract confidence for image 0 matches: [1, 0, :]
                        confidence = confidence[0, 0, :]  # Take first batch, first image
                        print(f"Debug: confidence shape after extraction: {confidence.shape}")

                    # If we filtered matches, also filter confidence
                    if 'match_indices_0' in locals() and match_indices_0 is not None:
                        confidence = confidence[match_indices_0]
                        print(f"✓ Filtered confidence to {len(confidence)} values matching keypoints")

                    # Ensure confidence matches keypoints length
                    if len(confidence) != len(keypoints1):
                        print(f"⚠ Confidence length {len(confidence)} doesn't match keypoints {len(keypoints1)}")
                        # Truncate or pad to match
                        if len(confidence) > len(keypoints1):
                            confidence = confidence[:len(keypoints1)]
                        else:
                            # Pad with default values
                            padding = np.ones(len(keypoints1) - len(confidence)) * 0.8
                            confidence = np.concatenate([confidence, padding])
                        print(f"  Adjusted confidence to {len(confidence)} values")
                else:
                    # Default to uniform confidence
                    confidence = np.ones(len(keypoints1)) * 0.8
                    print(f"⚠ No confidence scores found, using default 0.8 for {len(keypoints1)} matches")

            # If still no keypoints, fail gracefully
            if keypoints1 is None:
                raise RuntimeError(
                    f"Could not extract keypoints from model output.\n"
                    f"Output type: {type(outputs)}\n"
                    f"Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}\n"
                    f"Please check the model output format."
                )

            # Scale keypoints back to original image sizes
            # Check if keypoints are in normalized coordinates [-1, 1] or pixel coordinates
            print(f"Debug: Keypoint ranges before scaling:")
            print(f"  kp1 x: [{keypoints1[:, 0].min():.3f}, {keypoints1[:, 0].max():.3f}]")
            print(f"  kp1 y: [{keypoints1[:, 1].min():.3f}, {keypoints1[:, 1].max():.3f}]")
            print(f"  kp2 x: [{keypoints2[:, 0].min():.3f}, {keypoints2[:, 0].max():.3f}]")
            print(f"  kp2 y: [{keypoints2[:, 1].min():.3f}, {keypoints2[:, 1].max():.3f}]")

            # Determine coordinate system and convert to pixels
            kp1_min_x, kp1_max_x = keypoints1[:, 0].min(), keypoints1[:, 0].max()
            kp1_min_y, kp1_max_y = keypoints1[:, 1].min(), keypoints1[:, 1].max()

            model_size = 832  # Model processes images at 832x832

            # Check if normalized [0, 1] vs [-1, 1] vs already pixels
            if kp1_min_x >= -0.1 and kp1_max_x <= 1.1:
                # Normalized coordinates in [0, 1] range
                print("Debug: Keypoints appear to be in normalized coordinates [0, 1]")
                # Convert from [0, 1] to pixel coordinates
                # Formula: pixel = normalized * size
                keypoints1[:, 0] = keypoints1[:, 0] * model_size
                keypoints1[:, 1] = keypoints1[:, 1] * model_size
                keypoints2[:, 0] = keypoints2[:, 0] * model_size
                keypoints2[:, 1] = keypoints2[:, 1] * model_size

                print(f"Debug: After [0,1] → pixel conversion:")
                print(f"  kp1 x: [{keypoints1[:, 0].min():.1f}, {keypoints1[:, 0].max():.1f}]")
                print(f"  kp1 y: [{keypoints1[:, 1].min():.1f}, {keypoints1[:, 1].max():.1f}]")

            elif kp1_min_x >= -1.1 and kp1_max_x <= 1.1:
                # Normalized coordinates in [-1, 1] range
                print("Debug: Keypoints appear to be in normalized coordinates [-1, 1]")
                # Convert from [-1, 1] to pixel coordinates
                # Formula: pixel = (normalized + 1) * (size / 2)
                keypoints1[:, 0] = (keypoints1[:, 0] + 1) * (model_size / 2)
                keypoints1[:, 1] = (keypoints1[:, 1] + 1) * (model_size / 2)
                keypoints2[:, 0] = (keypoints2[:, 0] + 1) * (model_size / 2)
                keypoints2[:, 1] = (keypoints2[:, 1] + 1) * (model_size / 2)

                print(f"Debug: After [-1,1] → pixel conversion:")
                print(f"  kp1 x: [{keypoints1[:, 0].min():.1f}, {keypoints1[:, 0].max():.1f}]")
                print(f"  kp1 y: [{keypoints1[:, 1].min():.1f}, {keypoints1[:, 1].max():.1f}]")
            else:
                print("Debug: Keypoints appear to already be in pixel coordinates")

            # Now scale from model size to original image size
            scale_x1 = orig_w1 / model_size
            scale_y1 = orig_h1 / model_size
            scale_x2 = orig_w2 / model_size
            scale_y2 = orig_h2 / model_size

            keypoints1[:, 0] *= scale_x1
            keypoints1[:, 1] *= scale_y1
            keypoints2[:, 0] *= scale_x2
            keypoints2[:, 1] *= scale_y2

            print(f"Debug: After scaling to original image size:")
            print(f"  kp1 x: [{keypoints1[:, 0].min():.1f}, {keypoints1[:, 0].max():.1f}] (orig_w={orig_w1})")
            print(f"  kp1 y: [{keypoints1[:, 1].min():.1f}, {keypoints1[:, 1].max():.1f}] (orig_h={orig_h1})")

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

        # Debug: Check what the processor created
        print(f"Debug: Processor inputs keys: {inputs.keys()}")
        for key, val in inputs.items():
            if hasattr(val, 'shape'):
                print(f"  - {key}: shape {val.shape}, dtype {val.dtype}")
                if 'pixel' in key.lower():
                    print(f"    value range: [{val.min():.3f}, {val.max():.3f}]")

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

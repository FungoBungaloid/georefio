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

from ..core.logging_utils import get_logger

try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = get_logger('inference')


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

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.processor, self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        """Load MatchAnything-ELoFTR model from HuggingFace Hub.

        Returns:
            Tuple of (processor, model)
        """
        try:
            logger.info(f"Loading model from HuggingFace: {self.MODEL_REPO} (cache: {self.weights_path})")

            processor = AutoImageProcessor.from_pretrained(
                self.MODEL_REPO,
                cache_dir=self.weights_path
            )

            # Prefer the dedicated keypoint-matching head; AutoModel only
            # provides the backbone and needs manual post-processing.
            try:
                from transformers import EfficientLoFTRForKeypointMatching
                model = EfficientLoFTRForKeypointMatching.from_pretrained(
                    self.MODEL_REPO,
                    cache_dir=self.weights_path
                )
                logger.info("Loaded EfficientLoFTRForKeypointMatching model")
            except ImportError:
                model = AutoModel.from_pretrained(
                    self.MODEL_REPO,
                    cache_dir=self.weights_path
                )
                logger.warning(
                    "EfficientLoFTRForKeypointMatching not available in this transformers "
                    "version; loaded AutoModel backbone. Consider upgrading transformers."
                )

            logger.info(f"Model loaded successfully: {type(model).__name__}")
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

        orig_h1, orig_w1 = image1.shape[:2]
        orig_h2, orig_w2 = image2.shape[:2]

        inputs = self._preprocess_images_pair(image1, image2)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Preferred path: the official post-processing API scales keypoints back
        # to the original image sizes and filters to valid matches for us.
        result = self._extract_with_postprocessing(outputs, (orig_h1, orig_w1), (orig_h2, orig_w2))

        if result is None:
            # Fallback for older transformers versions without
            # post_process_keypoint_matching: parse the raw output tensors.
            result = self._extract_manually(outputs, inputs, (orig_h1, orig_w1), (orig_h2, orig_w2))

        keypoints1, keypoints2, confidence = result

        if keypoints1 is None or len(keypoints1) == 0:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 2), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )

        # Limit to max_keypoints, keeping the most confident matches
        if len(keypoints1) > max_keypoints:
            top_indices = np.argsort(confidence)[-max_keypoints:]
            keypoints1 = keypoints1[top_indices]
            keypoints2 = keypoints2[top_indices]
            confidence = confidence[top_indices]

        logger.debug(
            f"match_images: {len(keypoints1)} matches, "
            f"confidence range [{confidence.min():.3f}, {confidence.max():.3f}]"
        )

        return keypoints1, keypoints2, confidence

    def _extract_with_postprocessing(
        self,
        outputs,
        size1: Tuple[int, int],
        size2: Tuple[int, int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Extract matches via processor.post_process_keypoint_matching.

        Returns None if the processor doesn't support post-processing
        (older transformers versions).
        """
        if not hasattr(self.processor, 'post_process_keypoint_matching'):
            return None

        try:
            # target_sizes: one (h, w) pair per image in the pair
            processed = self.processor.post_process_keypoint_matching(
                outputs,
                target_sizes=[(size1, size2)],
                threshold=0.0  # keep everything; filtering happens downstream
            )
        except TypeError:
            # Some versions expect a flat list of sizes
            try:
                processed = self.processor.post_process_keypoint_matching(
                    outputs,
                    target_sizes=[size1, size2],
                    threshold=0.0
                )
            except Exception as e:
                logger.warning(f"post_process_keypoint_matching failed: {e}")
                return None
        except Exception as e:
            logger.warning(f"post_process_keypoint_matching failed: {e}")
            return None

        if not processed:
            return None

        pair = processed[0]
        keypoints1 = pair['keypoints0'].cpu().numpy().astype(np.float64)
        keypoints2 = pair['keypoints1'].cpu().numpy().astype(np.float64)
        confidence = pair['matching_scores'].cpu().numpy().astype(np.float64)

        return keypoints1, keypoints2, confidence

    def _extract_manually(
        self,
        outputs,
        inputs: dict,
        size1: Tuple[int, int],
        size2: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse raw model outputs into matched keypoint pairs.

        Fallback used when the processor lacks post_process_keypoint_matching.
        Keypoints are scaled from the processed tensor size back to the
        original image sizes.
        """
        keypoints1 = None
        keypoints2 = None
        confidence = None
        match_indices_0 = None

        kp = getattr(outputs, 'keypoints', None)

        if kp is not None and hasattr(kp, 'shape'):
            if len(kp.shape) == 4 and kp.shape[0] == 1 and kp.shape[1] == 2:
                # [1, 2, N, 2]: batch, image index, keypoints, xy
                keypoints1 = kp[0, 0].cpu().numpy().astype(np.float64)
                keypoints2 = kp[0, 1].cpu().numpy().astype(np.float64)

                matches = getattr(outputs, 'matches', None)
                if matches is not None:
                    # matches[0, 0, i] = j: keypoint i in image 0 matches j in image 1 (-1 = none)
                    matches_0_to_1 = matches[0, 0].cpu().numpy()
                    valid = (matches_0_to_1 >= 0) & (matches_0_to_1 < len(keypoints2))
                    match_indices_0 = np.nonzero(valid)[0]
                    target_indices = matches_0_to_1[valid].astype(int)

                    keypoints1 = keypoints1[match_indices_0]
                    keypoints2 = keypoints2[target_indices]

            elif len(kp.shape) == 3 and kp.shape[0] == 2:
                # [2, N, 2]: image index, keypoints, xy (already paired)
                keypoints1 = kp[0].cpu().numpy().astype(np.float64)
                keypoints2 = kp[1].cpu().numpy().astype(np.float64)

        if keypoints1 is None and hasattr(outputs, 'keypoints0') and hasattr(outputs, 'keypoints1'):
            keypoints1 = outputs.keypoints0.cpu().numpy().astype(np.float64)
            keypoints2 = outputs.keypoints1.cpu().numpy().astype(np.float64)

        if keypoints1 is None:
            raise RuntimeError(
                f"Could not extract keypoints from model output.\n"
                f"Output type: {type(outputs)}\n"
                f"Available attributes: {[a for a in dir(outputs) if not a.startswith('_')]}\n"
                f"Please check the model output format (or upgrade transformers)."
            )

        # Confidence scores
        for attr in ('confidence', 'matching_scores', 'scores'):
            value = getattr(outputs, attr, None)
            if value is not None:
                confidence = value.cpu().numpy().astype(np.float64)
                break

        if confidence is not None:
            if confidence.ndim == 3:
                # [1, 2, N] -> scores for image 0's keypoints
                confidence = confidence[0, 0, :]
            if match_indices_0 is not None and len(confidence) > len(keypoints1):
                confidence = confidence[match_indices_0]
            if len(confidence) != len(keypoints1):
                logger.warning(
                    f"Confidence length {len(confidence)} doesn't match "
                    f"{len(keypoints1)} keypoints; padding/truncating"
                )
                if len(confidence) > len(keypoints1):
                    confidence = confidence[:len(keypoints1)]
                else:
                    confidence = np.concatenate([
                        confidence,
                        np.full(len(keypoints1) - len(confidence), 0.5)
                    ])
        else:
            confidence = np.full(len(keypoints1), 0.5)
            logger.warning("No confidence scores in model output; using uniform 0.5")

        if len(keypoints1) == 0:
            return keypoints1, keypoints2, confidence

        # Determine the coordinate space of the raw keypoints and convert to
        # pixels in the processed tensor, then scale to the original images.
        proc_h, proc_w = self._processed_size(inputs)

        kp_max = max(keypoints1.max(), keypoints2.max())
        kp_min = min(keypoints1.min(), keypoints2.min())

        if kp_min >= -0.1 and kp_max <= 1.1:
            # Normalized [0, 1]
            keypoints1 = keypoints1 * [proc_w, proc_h]
            keypoints2 = keypoints2 * [proc_w, proc_h]
        elif kp_min >= -1.1 and kp_max <= 1.1:
            # Normalized [-1, 1]
            keypoints1 = (keypoints1 + 1) * [proc_w / 2, proc_h / 2]
            keypoints2 = (keypoints2 + 1) * [proc_w / 2, proc_h / 2]
        # else: already pixel coordinates in the processed tensor

        orig_h1, orig_w1 = size1
        orig_h2, orig_w2 = size2

        keypoints1 = keypoints1 * [orig_w1 / proc_w, orig_h1 / proc_h]
        keypoints2 = keypoints2 * [orig_w2 / proc_w, orig_h2 / proc_h]

        return keypoints1, keypoints2, confidence

    def _processed_size(self, inputs: dict) -> Tuple[int, int]:
        """Get the (height, width) of the tensors fed to the model."""
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None and pixel_values.dim() >= 2:
            return int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        # Last-resort default matching EfficientLoFTR's published config
        return 832, 832

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
        def to_pil(img):
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
                return Image.fromarray(img)
            return img

        pil_img1 = to_pil(image1)
        pil_img2 = to_pil(image2)

        # EfficientLoFTR processor expects the images as a pair
        inputs = self.processor(
            images=[pil_img1, pil_img2],
            return_tensors="pt"
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

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
            return None, np.zeros(len(keypoints1), dtype=bool)

        try:
            H, mask = cv2.findHomography(
                keypoints1,
                keypoints2,
                cv2.RANSAC,
                ransac_threshold
            )

            if H is None:
                return None, np.zeros(len(keypoints1), dtype=bool)

            return H, mask.ravel().astype(bool)

        except cv2.error as e:
            logger.warning(f"Homography estimation failed: {e}")
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
        scale = target_size / max(h, w)

        if maintain_aspect:
            new_h = int(h * scale)
            new_w = int(w * scale)
        else:
            new_h = target_size
            new_w = target_size

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

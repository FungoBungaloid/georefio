#!/usr/bin/env python3
"""
Test script for MatchAnything-ELoFTR model

This script explores the model API and tests inference outside of QGIS.
It helps understand the model's input/output format before integrating into the plugin.

Usage:
    python test_model.py [--image1 path] [--image2 path]

If no images provided, uses dummy test images.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

def test_model_loading():
    """Test loading the MatchAnything-ELoFTR model from HuggingFace"""
    print("=" * 80)
    print("Testing MatchAnything-ELoFTR Model Loading")
    print("=" * 80)

    try:
        from transformers import AutoImageProcessor, AutoModel
        import torch

        print("\n Dependencies imported successfully")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")

        # Model repository
        model_repo = "zju-community/matchanything_eloftr"
        cache_dir = Path("./model_cache")

        print(f"\n=å Loading model from HuggingFace: {model_repo}")
        print(f"   Cache directory: {cache_dir}")

        # Load processor
        print("\n  Loading image processor...")
        processor = AutoImageProcessor.from_pretrained(
            model_repo,
            cache_dir=cache_dir
        )
        print("   Image processor loaded")

        # Load model
        print("\n  Loading model...")
        model = AutoModel.from_pretrained(
            model_repo,
            cache_dir=cache_dir
        )
        print("   Model loaded")

        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        print(f"\n Model ready on device: {device}")

        # Inspect model
        print("\n=Ë Model Information:")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Model config: {model.config if hasattr(model, 'config') else 'N/A'}")

        # Inspect processor
        print("\n=Ë Processor Information:")
        print(f"  - Processor type: {type(processor).__name__}")
        if hasattr(processor, 'size'):
            print(f"  - Default size: {processor.size}")

        return processor, model, device

    except ImportError as e:
        print(f"\nL Import error: {e}")
        print("\nPlease install required packages:")
        print("  pip install torch transformers huggingface-hub")
        return None, None, None

    except Exception as e:
        print(f"\nL Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def create_test_images():
    """Create simple test images for matching"""
    print("\n" + "=" * 80)
    print("Creating Test Images")
    print("=" * 80)

    # Create a simple pattern image
    size = 512

    # Image 1: Grid pattern
    img1 = np.ones((size, size, 3), dtype=np.uint8) * 255
    for i in range(0, size, 64):
        img1[i:i+32, :] = [100, 150, 200]
        img1[:, i:i+32] = [100, 150, 200]

    # Image 2: Same pattern with slight offset and noise
    img2 = np.roll(img1, shift=20, axis=0)
    img2 = np.roll(img2, shift=15, axis=1)
    noise = np.random.randint(-20, 20, img2.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    print(f" Created test images: {size}x{size} pixels")

    # Save test images
    Image.fromarray(img1).save("test_image1.png")
    Image.fromarray(img2).save("test_image2.png")
    print(" Saved test images: test_image1.png, test_image2.png")

    return img1, img2


def test_inference(processor, model, device, img1, img2):
    """Test model inference on two images"""
    print("\n" + "=" * 80)
    print("Testing Model Inference")
    print("=" * 80)

    try:
        import torch
        from PIL import Image as PILImage

        # Convert numpy arrays to PIL Images
        pil_img1 = PILImage.fromarray(img1)
        pil_img2 = PILImage.fromarray(img2)

        print(f"\n=Ê Input images:")
        print(f"  - Image 1: {img1.shape}")
        print(f"  - Image 2: {img2.shape}")

        # Preprocess images
        print("\n= Preprocessing images...")

        # Try different preprocessing approaches
        print("\n  Approach 1: Process images separately")
        try:
            inputs1 = processor(images=pil_img1, return_tensors="pt")
            inputs2 = processor(images=pil_img2, return_tensors="pt")

            # Move to device
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}

            print(f"   Preprocessed separately")
            print(f"    - Input 1 keys: {list(inputs1.keys())}")
            print(f"    - Input 2 keys: {list(inputs2.keys())}")

            for key in inputs1.keys():
                print(f"    - {key} shape: {inputs1[key].shape}")

        except Exception as e:
            print(f"  L Separate processing failed: {e}")
            inputs1, inputs2 = None, None

        print("\n  Approach 2: Process images together")
        try:
            inputs_combined = processor(images=[pil_img1, pil_img2], return_tensors="pt")
            inputs_combined = {k: v.to(device) for k, v in inputs_combined.items()}

            print(f"   Preprocessed together")
            print(f"    - Input keys: {list(inputs_combined.keys())}")

            for key in inputs_combined.keys():
                print(f"    - {key} shape: {inputs_combined[key].shape}")

        except Exception as e:
            print(f"  L Combined processing failed: {e}")
            inputs_combined = None

        # Run inference
        print("\n=€ Running inference...")

        with torch.no_grad():
            # Try inference with separate inputs
            if inputs1 is not None and inputs2 is not None:
                print("\n  Attempt 1: Separate inputs")
                try:
                    # Try calling model with both inputs
                    output = model(**inputs1, **inputs2)
                    print(f"   Inference successful!")
                    print(f"    - Output type: {type(output)}")

                    if hasattr(output, 'keys'):
                        print(f"    - Output keys: {list(output.keys())}")
                        for key in output.keys():
                            val = output[key]
                            if torch.is_tensor(val):
                                print(f"      - {key}: shape {val.shape}")
                            else:
                                print(f"      - {key}: {type(val)}")
                    else:
                        print(f"    - Output: {output}")

                except Exception as e:
                    print(f"  L Failed: {e}")

            # Try inference with combined inputs
            if inputs_combined is not None:
                print("\n  Attempt 2: Combined inputs")
                try:
                    output = model(**inputs_combined)
                    print(f"   Inference successful!")
                    print(f"    - Output type: {type(output)}")

                    if hasattr(output, 'keys'):
                        print(f"    - Output keys: {list(output.keys())}")
                        for key in output.keys():
                            val = output[key]
                            if torch.is_tensor(val):
                                print(f"      - {key}: shape {val.shape}")
                            else:
                                print(f"      - {key}: {type(val)}")
                    else:
                        print(f"    - Output: {output}")

                except Exception as e:
                    print(f"  L Failed: {e}")

        print("\n Inference testing complete")
        print("\n=¡ Next steps:")
        print("  1. Study the output format to extract keypoints")
        print("  2. Implement proper preprocessing in inference.py")
        print("  3. Update match_images() method with correct API calls")

    except Exception as e:
        print(f"\nL Inference test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("\n" + "=," * 40)
    print("MatchAnything-ELoFTR Model Test Script")
    print("=," * 40)

    # Test model loading
    processor, model, device = test_model_loading()

    if processor is None or model is None:
        print("\nL Model loading failed. Exiting.")
        return 1

    # Create or load test images
    if len(sys.argv) >= 3:
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        print(f"\n=Á Loading images from command line:")
        print(f"  - Image 1: {img1_path}")
        print(f"  - Image 2: {img2_path}")

        img1 = np.array(Image.open(img1_path).convert('RGB'))
        img2 = np.array(Image.open(img2_path).convert('RGB'))
    else:
        img1, img2 = create_test_images()

    # Test inference
    test_inference(processor, model, device, img1, img2)

    print("\n" + "=" * 80)
    print(" Test Complete!")
    print("=" * 80)
    print("\nCheck the output above to understand the model API.")
    print("Use this information to complete the inference implementation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

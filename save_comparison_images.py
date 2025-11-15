"""
Save the actual source and reference images being compared by the matching model.
This helps debug whether the images are the right quality, zoom level, etc.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add magic_georeferencer to path
sys.path.insert(0, str(Path(__file__).parent / 'magic_georeferencer'))

from core.tile_fetcher import TileFetcher
from core.model_manager import ModelManager
from matchanything.inference import MatchAnythingInference

try:
    from qgis.utils import iface
    from qgis.core import QgsProject
except:
    print("ERROR: Must be run from QGIS Python console!")
    sys.exit(1)


def save_images_for_inspection(source_image_path, output_dir="/tmp/georefio_inspect"):
    """
    Save the source and reference images that will be compared.

    Args:
        source_image_path: Path to the ungeoreferenced source image
        output_dir: Where to save the output images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\nSaving images to: {output_dir}")
    print("="*80)

    # 1. Load source image
    print("\n1. Loading source image...")
    source_img = Image.open(source_image_path)
    print(f"   Original size: {source_img.width} × {source_img.height}")

    # Save original
    source_orig_path = output_dir / "1_source_original.png"
    source_img.save(source_orig_path)
    print(f"   Saved: {source_orig_path}")

    # Resize to model input size (832x832)
    source_resized = source_img.resize((832, 832), Image.Resampling.LANCZOS)
    source_resized_path = output_dir / "2_source_resized_832x832.png"
    source_resized.save(source_resized_path)
    print(f"   Saved resized (832x832): {source_resized_path}")

    # 2. Capture reference from QGIS
    print("\n2. Capturing reference image from QGIS canvas...")
    tile_fetcher = TileFetcher()

    try:
        # Capture canvas at 832x832
        ref_array, ref_extent = tile_fetcher.capture_canvas(iface, size=832)
        print(f"   Captured size: {ref_array.shape[1]} × {ref_array.shape[0]}")
        print(f"   Extent: {ref_extent.toString()}")

        # Get CRS info
        canvas = iface.mapCanvas()
        crs = canvas.mapSettings().destinationCrs()
        scale = canvas.scale()

        print(f"   CRS: {crs.authid()}")
        print(f"   Scale: 1:{int(scale)}")

        # Save reference
        ref_img = Image.fromarray(ref_array.astype(np.uint8))
        ref_path = output_dir / "3_reference_qgis_canvas_832x832.png"
        ref_img.save(ref_path)
        print(f"   Saved: {ref_path}")

        # Also save at higher resolution for comparison
        ref_array_high, _ = tile_fetcher.capture_canvas(iface, size=2048)
        ref_img_high = Image.fromarray(ref_array_high.astype(np.uint8))
        ref_high_path = output_dir / "4_reference_qgis_canvas_2048.png"
        ref_img_high.save(ref_high_path)
        print(f"   Saved high-res: {ref_high_path}")

    except Exception as e:
        print(f"   ERROR capturing canvas: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Create side-by-side comparison
    print("\n3. Creating side-by-side comparison...")
    comparison = Image.new('RGB', (832 * 2, 832))
    comparison.paste(source_resized, (0, 0))
    comparison.paste(ref_img, (832, 0))

    comparison_path = output_dir / "5_comparison_source_vs_reference.png"
    comparison.save(comparison_path)
    print(f"   Saved: {comparison_path}")

    # 4. Quick analysis
    print("\n4. Quick image analysis...")

    # Convert to grayscale for analysis
    source_gray = np.array(source_resized.convert('L'))
    ref_gray = np.array(ref_img.convert('L'))

    print(f"   Source image:")
    print(f"     - Mean brightness: {source_gray.mean():.1f}")
    print(f"     - Std dev: {source_gray.std():.1f}")
    print(f"     - Min: {source_gray.min()}, Max: {source_gray.max()}")

    print(f"   Reference image:")
    print(f"     - Mean brightness: {ref_gray.mean():.1f}")
    print(f"     - Std dev: {ref_gray.std():.1f}")
    print(f"     - Min: {ref_gray.min()}, Max: {ref_gray.max()}")

    # Check if images are too different
    if abs(source_gray.mean() - ref_gray.mean()) > 100:
        print(f"   ⚠ WARNING: Images have very different brightness!")

    if source_gray.std() < 20 or ref_gray.std() < 20:
        print(f"   ⚠ WARNING: One or both images have very low contrast (std < 20)")

    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)
    print(f"\nPlease visually inspect the images in: {output_dir}")
    print("\nLook for:")
    print("  1. Are the source and reference showing similar geographic features?")
    print("  2. Are they at similar scales/zoom levels?")
    print("  3. Is the reference image too blurry or too detailed?")
    print("  4. Are there recognizable common features (roads, buildings, etc.)?")
    print("  5. Is one image much darker/lighter than the other?")
    print("\nIf the images look very different, the matching will fail.")
    print("Try:")
    print("  - Zoom QGIS in/out to match the source image scale")
    print("  - Change the basemap (OSM vs aerial)")
    print("  - Navigate to the correct location")


def main():
    """Main entry point"""
    if not iface:
        print("ERROR: Must be run from QGIS Python console!")
        return

    # Get source image path from user
    print("\nEnter the path to your source image:")
    print("(Or edit this script to set it directly)")

    # Default path for testing
    source_path = "/tmp/test_image.png"

    # Check if default exists, otherwise ask
    if not Path(source_path).exists():
        print(f"\nDefault path {source_path} doesn't exist.")
        print("Please edit this script and set source_path to your image.")
        return

    save_images_for_inspection(source_path)


if __name__ == "__main__":
    # You can set your source image path here:
    SOURCE_IMAGE = "/tmp/test_image.png"  # <-- EDIT THIS

    if iface and Path(SOURCE_IMAGE).exists():
        save_images_for_inspection(SOURCE_IMAGE)
    else:
        main()

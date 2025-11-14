# Magic Georeferencer - QGIS Plugin Specification

## Project Overview

**Magic Georeferencer** is a QGIS plugin that leverages the MatchAnything AI model to automatically georeference ungeoreferenced images (maps, aerial photos, sketches) by matching them against real-world basemap data.

### Core Concept

1. User loads an ungeoreferenced image into QGIS
2. User navigates map canvas to approximate location
3. Plugin captures current map view (OSM or aerial tiles)
4. MatchAnything model finds correspondence points between images
5. Confidence filtering generates high-quality Ground Control Points (GCPs)
6. QGIS georeferencer applies transformation and outputs georeferenced image

### Key Features

- **Zero manual GCP placement** - fully automated matching
- **Cross-modality support** - handles maps→aerial, aerial→maps, sketches→maps, etc.
- **Progressive refinement** - multi-scale matching for accuracy
- **GPU acceleration** with CPU fallback
- **Confidence visualization** - review matches before committing
- **Flexible basemap sources** - OSM standard, aerial imagery, etc.

---

## Technical Architecture

### Component Overview

```
magic_georeferencer/
├── __init__.py                      # QGIS plugin entry point
├── magic_georeferencer.py           # Main plugin class
├── metadata.txt                     # QGIS plugin metadata
├── icon.png                         # Plugin icon
│
├── core/
│   ├── __init__.py
│   ├── model_manager.py             # Weight download, CUDA detection, model loading
│   ├── matcher.py                   # MatchAnything inference wrapper
│   ├── tile_fetcher.py              # Basemap tile capture/fetching
│   ├── gcp_generator.py             # Match → GCP conversion
│   └── georeferencer.py             # QGIS georeferencer integration
│
├── ui/
│   ├── __init__.py
│   ├── main_dialog.py               # Primary UI dialog
│   ├── main_dialog.ui               # Qt Designer UI file
│   ├── progress_dialog.py           # Download/processing progress
│   ├── confidence_viewer.py         # Match confidence visualization
│   └── resources.qrc                # Qt resources
│
├── matchanything/                   # MatchAnything integration
│   ├── __init__.py
│   ├── inference.py                 # Simplified inference wrapper
│   └── requirements.txt             # Model dependencies
│
├── config/
│   ├── tile_sources.json            # Basemap tile configurations
│   └── default_settings.json        # Plugin default settings
│
└── weights/                         # Downloaded on first run
    └── .gitkeep
```

---

## Core Components Specification

### 1. Model Manager (`core/model_manager.py`)

**Responsibilities:**
- Detect CUDA availability
- Download model weights on first run
- Load MatchAnything model into memory
- Manage model inference device (GPU/CPU)

**Key Functions:**

```python
class ModelManager:
    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        self.device = None
        self.model = None
        
    def is_cuda_available(self) -> bool:
        """Check for CUDA-capable GPU"""
        
    def weights_exist(self) -> bool:
        """Check if model weights are downloaded"""
        
    def download_weights(self, progress_callback: Callable) -> bool:
        """
        Download weights from HuggingFace with progress updates
        URL: https://huggingface.co/LittleFrog/MatchAnything/resolve/main/weights.zip
        Returns: True on success, False on failure
        """
        
    def load_model(self) -> bool:
        """
        Load MatchAnything model
        - Auto-detect device (CUDA/CPU)
        - Load appropriate config
        - Return success status
        """
        
    def get_inference_config(self) -> dict:
        """
        Return device-appropriate inference settings
        GPU: {'size': 1024, 'num_keypoints': 2048}
        CPU: {'size': 512, 'num_keypoints': 512}
        """
        
    def unload_model(self):
        """Free memory"""
```

**Download Process:**
```python
def download_weights(self, progress_callback):
    """
    1. Create weights directory if needed
    2. Download weights.zip with resume support
    3. Verify checksum (MD5/SHA256)
    4. Extract to weights/
    5. Cleanup zip file
    6. Call progress_callback(current, total) during download
    """
```

**First Run Detection:**
```python
def check_first_run() -> bool:
    """
    Check if this is first plugin activation
    - Look for weights/model_checkpoint.pth (or whatever the model file is named)
    - Return True if missing
    """
```

---

### 2. Tile Fetcher (`core/tile_fetcher.py`)

**Responsibilities:**
- Capture current QGIS map canvas
- Fetch tiles from various basemap sources
- Handle coordinate transformations
- Manage tile caching

**Key Functions:**

```python
class TileFetcher:
    def __init__(self):
        self.tile_sources = load_tile_sources()
        self.cache_dir = Path(QgsApplication.qgisSettingsDirPath()) / "magic_georeferencer" / "tiles"
        
    def capture_canvas(self, iface: QgisInterface, size: int = 1024) -> Tuple[np.ndarray, QgsRectangle]:
        """
        Capture current QGIS map canvas as numpy array
        
        Args:
            iface: QGIS interface
            size: Target image size (will maintain aspect ratio)
            
        Returns:
            (image_array, extent_rectangle)
            - image_array: RGB numpy array [H, W, 3]
            - extent_rectangle: QgsRectangle in map CRS
        """
        
    def fetch_tiles(self, extent: QgsRectangle, crs: QgsCoordinateReferenceSystem, 
                   source_name: str, zoom_level: int = 17, size: int = 1024) -> Tuple[np.ndarray, QgsRectangle]:
        """
        Fetch and stitch tiles for given extent
        
        Args:
            extent: Geographic extent to fetch
            crs: Coordinate system of extent
            source_name: Key from tile_sources.json
            zoom_level: Tile zoom level
            size: Target output size
            
        Returns:
            (stitched_image, actual_extent)
        """
        
    def get_tile_url(self, source_name: str, x: int, y: int, z: int) -> str:
        """Generate tile URL for given TMS coordinates"""
        
    def extent_to_tile_coords(self, extent: QgsRectangle, zoom: int) -> List[Tuple[int, int]]:
        """Convert geographic extent to tile coordinates"""
```

**Tile Sources Configuration** (`config/tile_sources.json`):
```json
{
  "osm_standard": {
    "name": "OpenStreetMap Standard",
    "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "type": "map",
    "attribution": "© OpenStreetMap contributors",
    "max_zoom": 19,
    "description": "Best for: Road maps, building footprints, labeled features"
  },
  "esri_world_imagery": {
    "name": "ESRI World Imagery",
    "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "type": "aerial",
    "attribution": "Esri, Maxar, Earthstar Geographics",
    "max_zoom": 19,
    "description": "Best for: Aerial photos, satellite imagery"
  },
  "osm_humanitarian": {
    "name": "OpenStreetMap Humanitarian",
    "url": "https://tile-{s}.openstreetmap.fr/hot/{z}/{x}/{y}.png",
    "subdomains": ["a", "b", "c"],
    "type": "map",
    "attribution": "© OpenStreetMap contributors, Tiles courtesy of Humanitarian OSM Team",
    "max_zoom": 18,
    "description": "Best for: High contrast, simplified features"
  }
}
```

---

### 3. Matcher (`core/matcher.py`)

**Responsibilities:**
- Interface with MatchAnything model
- Progressive multi-scale matching
- Match confidence scoring
- Outlier filtering

**Key Functions:**

```python
class Matcher:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.config = model_manager.get_inference_config()
        
    def match_progressive(self, image_src: np.ndarray, image_ref: np.ndarray,
                         scales: List[int] = None) -> MatchResult:
        """
        Multi-scale progressive matching
        
        Args:
            image_src: Source (ungeoreferenced) image [H, W, 3]
            image_ref: Reference (basemap) image [H, W, 3]
            scales: List of image sizes to try, e.g., [512, 768, 1024]
                   If None, uses single scale from config
                   
        Returns:
            MatchResult containing keypoints, confidence scores, etc.
            
        Process:
        1. Start at lowest resolution
        2. Get initial matches
        3. Filter by confidence threshold
        4. If > min_gcps found, proceed to next scale
        5. Use previous matches to refine search space
        6. Repeat until highest resolution or failure
        """
        
    def match_single_scale(self, image_src: np.ndarray, image_ref: np.ndarray,
                          size: int) -> MatchResult:
        """
        Single-scale matching
        
        1. Resize images to target size (maintain aspect ratio)
        2. Run MatchAnything inference
        3. Parse keypoint matches
        4. Calculate confidence scores
        5. Apply geometric consistency check (RANSAC)
        6. Return MatchResult
        """
        
    def filter_matches(self, matches: MatchResult, 
                      confidence_threshold: float = 0.7,
                      geometric_threshold: float = 0.8,
                      min_gcps: int = 6) -> MatchResult:
        """
        Filter matches based on quality criteria
        
        - Remove low-confidence matches
        - RANSAC outlier removal
        - Ensure minimum GCP count
        - Check spatial distribution (avoid clustering)
        """
        
    def estimate_spatial_distribution_quality(self, keypoints: np.ndarray) -> float:
        """
        Score how well-distributed keypoints are (0-1)
        
        - Quadrant coverage (divide image into 3x3 grid)
        - Minimum spanning tree length
        - Clustering metric
        
        Returns: Quality score (1.0 = perfectly distributed, 0.0 = all clustered)
        """
```

**MatchResult Data Structure:**
```python
@dataclass
class MatchResult:
    """Container for match results"""
    
    # Keypoints in source image coordinates
    keypoints_src: np.ndarray  # [N, 2] (x, y)
    
    # Keypoints in reference image coordinates  
    keypoints_ref: np.ndarray  # [N, 2] (x, y)
    
    # Per-match confidence scores
    confidence: np.ndarray  # [N]
    
    # Geometric consistency (post-RANSAC)
    geometric_score: float  # 0-1
    
    # Spatial distribution quality
    distribution_quality: float  # 0-1
    
    # Estimated transformation matrix (if computed)
    transform_matrix: Optional[np.ndarray]  # [3, 3] homography
    
    # Processing metadata
    processing_time: float  # seconds
    device: str  # 'cuda' or 'cpu'
    resolution: int  # Image size used for matching
    
    def num_matches(self) -> int:
        return len(self.keypoints_src)
        
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidence))
```

---

### 4. GCP Generator (`core/gcp_generator.py`)

**Responsibilities:**
- Transform match coordinates to geographic coordinates
- Create QGIS GCP objects
- Validate GCP quality
- Export GCP list for georeferencer

**Key Functions:**

```python
class GCPGenerator:
    def __init__(self):
        pass
        
    def matches_to_gcps(self, match_result: MatchResult,
                       ref_extent: QgsRectangle,
                       ref_crs: QgsCoordinateReferenceSystem,
                       ref_image_size: Tuple[int, int],
                       src_image_size: Tuple[int, int]) -> List[QgsGeorefGCP]:
        """
        Convert match results to QGIS Ground Control Points
        
        Args:
            match_result: MatchResult from Matcher
            ref_extent: Geographic extent of reference image
            ref_crs: CRS of reference extent
            ref_image_size: (width, height) of reference image in pixels
            src_image_size: (width, height) of source image in pixels
            
        Returns:
            List of QgsGeorefGCP objects
            
        Process:
        1. For each match pair:
           a. Source coords: pixel coords in ungeoreferenced image
           b. Ref coords: pixel coords in basemap image
           c. Transform ref pixel coords to geographic coords using extent
           d. Create GCP with source pixels → geographic coords
        2. Validate GCP distribution
        3. Return GCP list
        """
        
    def pixel_to_geo(self, pixel_x: float, pixel_y: float,
                    extent: QgsRectangle, image_width: int, image_height: int) -> QgsPointXY:
        """
        Convert pixel coordinates to geographic coordinates
        
        Assumes image is georeferenced to extent (standard geo transform)
        """
        
    def validate_gcp_distribution(self, gcps: List[QgsGeorefGCP],
                                 image_width: int, image_height: int) -> Tuple[bool, str]:
        """
        Validate that GCPs are well-distributed
        
        Returns:
            (is_valid, message)
            
        Checks:
        - Minimum number of GCPs (4+ for perspective, 6+ for polynomial)
        - Coverage across image quadrants
        - No excessive clustering
        - Distance from image edges
        """
        
    def export_gcp_file(self, gcps: List[QgsGeorefGCP], filepath: Path):
        """Export GCPs to QGIS .points file format"""
```

---

### 5. Georeferencer (`core/georeferencer.py`)

**Responsibilities:**
- Interface with QGIS georeferencer
- Apply transformations
- Generate georeferenced output
- Handle various output formats

**Key Functions:**

```python
class Georeferencer:
    def __init__(self, iface: QgisInterface):
        self.iface = iface
        
    def georeference_image(self, source_image_path: Path,
                          gcps: List[QgsGeorefGCP],
                          output_path: Path,
                          target_crs: QgsCoordinateReferenceSystem,
                          transform_type: str = 'polynomial_1',
                          resampling: str = 'cubic',
                          compression: str = 'LZW') -> bool:
        """
        Georeference image using QGIS georeferencer
        
        Args:
            source_image_path: Path to ungeoreferenced image
            gcps: List of ground control points
            output_path: Output georeferenced raster path
            target_crs: Target coordinate reference system
            transform_type: 'polynomial_1', 'polynomial_2', 'polynomial_3', 'thin_plate_spline', 'projective'
            resampling: 'nearest', 'bilinear', 'cubic', 'lanczos'
            compression: 'NONE', 'LZW', 'PACKBITS', 'DEFLATE'
            
        Returns:
            True on success, False on failure
            
        Process:
        1. Create temporary GCP file
        2. Use QGIS gdal:warpreproject or georeferencer API
        3. Apply transformation
        4. Generate georeferenced raster
        5. Load into QGIS (optional)
        """
        
    def calculate_transform_error(self, gcps: List[QgsGeorefGCP],
                                  transform_type: str) -> Tuple[float, List[float]]:
        """
        Calculate reprojection error for given transform
        
        Returns:
            (mean_error, per_gcp_errors) in map units
        """
        
    def suggest_transform_type(self, num_gcps: int, 
                             distribution_quality: float) -> str:
        """
        Suggest appropriate transformation type based on GCP count and distribution
        
        Rules:
        - 4-5 GCPs: polynomial_1 (affine)
        - 6-9 GCPs: polynomial_1 or polynomial_2
        - 10+ GCPs: polynomial_2 or thin_plate_spline
        - Poor distribution: polynomial_1 (more stable)
        - Good distribution + many GCPs: polynomial_3 or thin_plate_spline
        """
```

---

## User Interface Specification

### Main Dialog (`ui/main_dialog.py`)

**Layout:**

```
┌─────────────────────────────────────────────────────────────┐
│ Magic Georeferencer                                    [?][×]│
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ 1. Load Ungeoreferenced Image                                │
│    ┌─────────────────────────────────────────┐  [Browse...] │
│    │ /path/to/historical_map.jpg             │              │
│    └─────────────────────────────────────────┘              │
│    Image size: 4000 × 3000 px                                │
│                                                               │
│ 2. Position Map Canvas                                       │
│    ☑ Show image overlay preview                              │
│    ℹ Navigate the map to the approximate location of your   │
│      image. The visible map extent will be used for matching.│
│                                                               │
│ 3. Configure Matching                                        │
│    Basemap Source: [OSM Standard          ▼]                 │
│                    ⓘ Best for road maps and building features│
│                                                               │
│    Match Quality:  [Balanced              ▼]                 │
│                    ○ Strict   ● Balanced   ○ Permissive      │
│                                                               │
│    ☑ Progressive refinement (slower but more accurate)       │
│                                                               │
│ 4. Run Matching                                              │
│    [         Match & Generate GCPs         ]                 │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│ Status: Ready                                                │
│ [?] Help    [⚙] Settings                     [Close] [Match] │
└─────────────────────────────────────────────────────────────┘
```

**UI Elements:**

1. **Image Selection**
   - File browser for source image
   - Display image dimensions
   - Preview thumbnail (optional)

2. **Map Positioning**
   - Checkbox to overlay source image on canvas (semi-transparent)
   - Instructions to navigate to approximate location
   - Button to capture current extent

3. **Configuration**
   - Dropdown: Basemap source selection
   - Radio buttons / Dropdown: Quality preset (Strict/Balanced/Permissive)
   - Checkbox: Enable progressive refinement
   - Advanced: Confidence threshold slider (if not using preset)

4. **Action Buttons**
   - "Match & Generate GCPs" - primary action
   - "Close" - cancel/exit
   - "Help" - show documentation
   - "Settings" - plugin settings dialog

**State Management:**

```python
class MagicGeoreferencerDialog(QDialog):
    def __init__(self, iface: QgisInterface, parent=None):
        self.iface = iface
        self.source_image_path = None
        self.model_manager = ModelManager()
        
        # Check if first run (weights not downloaded)
        if not self.model_manager.weights_exist():
            self.show_first_run_dialog()
            
    def show_first_run_dialog(self):
        """
        First-run dialog for weight download
        
        "Magic Georeferencer uses an AI model to automatically match images.
        
        The model weights (~800 MB) need to be downloaded once.
        
        Checking for CUDA GPU... [Found/Not Found]
        Recommended version: [GPU/CPU]
        
        [Download Now]  [Download Later]  [Cancel]"
        """
        
    def validate_inputs(self) -> Tuple[bool, str]:
        """
        Validate dialog inputs before processing
        
        Checks:
        - Source image selected and exists
        - Map canvas has valid extent
        - Model weights downloaded
        - Model loaded successfully
        
        Returns: (is_valid, error_message)
        """
        
    def run_matching(self):
        """
        Execute matching workflow
        
        1. Validate inputs
        2. Show progress dialog
        3. Capture map canvas / fetch tiles
        4. Run matching (with progress updates)
        5. Filter results
        6. Show confidence viewer
        7. Generate GCPs
        8. Offer to georeference
        """
```

---

### Progress Dialog (`ui/progress_dialog.py`)

**Purpose:** Show progress during long operations (download, matching)

**Layout:**

```
┌─────────────────────────────────────────┐
│ Magic Georeferencer - Processing...    │
├─────────────────────────────────────────┤
│                                         │
│ Downloading model weights...           │
│ ████████████░░░░░░░░  65% (520/800 MB) │
│                                         │
│ Estimated time remaining: 2m 15s        │
│                                         │
│              [Cancel]                   │
│                                         │
└─────────────────────────────────────────┘
```

**Features:**
- Cancellable operations
- Progress bar with percentage
- Status text (current operation)
- Estimated time remaining
- Download speed (for downloads)

**States:**
1. "Downloading model weights..." (first run)
2. "Loading model into memory..."
3. "Capturing map canvas..."
4. "Running image matching (Scale 1/3)..."
5. "Filtering matches..."
6. "Generating ground control points..."

---

### Confidence Viewer (`ui/confidence_viewer.py`)

**Purpose:** Visualize match results before committing to georeferencing

**Layout:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ Match Results - Review Before Georeferencing                  [?][×]│
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐ ┌─────────────────────────┐             │
│ │                         │ │                         │             │
│ │   Source Image          │ │   Reference Map         │             │
│ │   (Ungeoreferenced)     │ │   (Basemap)             │             │
│ │                         │ │                         │             │
│ │   • • •                 │ │   • • •                 │             │
│ │  • •  •                 │ │  • •  •                 │             │
│ │   •  ••                 │ │   •  ••                 │             │
│ └─────────────────────────┘ └─────────────────────────┘             │
│                                                                       │
│ Match Statistics:                                                    │
│ ├─ Total Matches: 24                                                 │
│ ├─ Mean Confidence: 0.82                                             │
│ ├─ Geometric Score: 0.88                                             │
│ └─ Distribution Quality: 0.75 (Good)                                 │
│                                                                       │
│ Confidence Threshold: ──────●──────────  0.70                        │
│                       Low         High                               │
│ Showing: 18 matches (6 filtered out)                                 │
│                                                                       │
│ ☑ Show only high-confidence matches (>0.7)                           │
│ ☑ Show match connections                                             │
│ ☐ Color by confidence (red=low, green=high)                          │
│                                                                       │
│ Estimated Transformation Error: 2.3 meters                           │
│ Suggested Transform Type: Polynomial (2nd order)                     │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│              [Adjust Threshold]  [Cancel]  [Accept & Georeference]  │
└─────────────────────────────────────────────────────────────────────┘
```

**Features:**

1. **Side-by-side image display**
   - Source image on left
   - Reference map on right
   - Synchronized zoom/pan
   - Keypoint overlays

2. **Interactive filtering**
   - Confidence threshold slider
   - Real-time filtering of displayed matches
   - Statistics update dynamically

3. **Visual indicators**
   - Colored keypoints (by confidence)
   - Connection lines between matching points
   - Highlight on hover
   - Click to remove individual matches

4. **Quality metrics**
   - Total match count
   - Mean confidence score
   - Geometric consistency score
   - Spatial distribution quality
   - Estimated transformation error
   - Suggested transform type

5. **Actions**
   - "Adjust Threshold" - modify filtering
   - "Cancel" - discard results
   - "Accept & Georeference" - proceed to georeferencing

---

### Settings Dialog (`ui/settings_dialog.py`)

**Global plugin settings:**

```python
SETTINGS = {
    # Performance
    'use_gpu': True,  # Auto-detected but can override
    'cpu_threads': 4,  # CPU thread count
    
    # Matching
    'default_quality_preset': 'balanced',
    'enable_progressive_refinement': True,
    'progressive_scales': [512, 768, 1024],
    
    # Tile fetching
    'default_basemap_source': 'osm_standard',
    'tile_cache_size_mb': 500,
    'tile_cache_expiry_days': 30,
    
    # Georeferencing
    'default_transform_type': 'auto',  # or 'polynomial_1', etc.
    'default_resampling': 'cubic',
    'default_compression': 'LZW',
    'auto_load_result': True,  # Load georeferenced raster into QGIS
    
    # Output
    'default_output_directory': '~/georeferenced_outputs',
    'output_naming_scheme': '{source_name}_georef.tif',
    
    # UI
    'show_overlay_preview': True,
    'show_confidence_viewer': True,  # Can skip for batch processing
}
```

---

## MatchAnything Integration

### Wrapper Implementation (`matchanything/inference.py`)

**Purpose:** Simplify MatchAnything inference for plugin use

```python
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class MatchAnythingInference:
    """Simplified MatchAnything inference wrapper"""
    
    def __init__(self, weights_path: Path, device: str = 'auto'):
        """
        Initialize MatchAnything model
        
        Args:
            weights_path: Path to model weights directory
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
        """
        self.weights_path = weights_path
        
        # Auto-detect device if needed
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model (actual implementation depends on MatchAnything API)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model(self):
        """Load MatchAnything model from weights"""
        # TODO: Use actual MatchAnything loading code
        # This will depend on their model architecture
        pass
        
    def match_images(self, 
                    image1: np.ndarray, 
                    image2: np.ndarray,
                    max_keypoints: int = 2048) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match two images
        
        Args:
            image1: First image [H, W, 3] RGB numpy array
            image2: Second image [H, W, 3] RGB numpy array  
            max_keypoints: Maximum number of keypoints to detect
            
        Returns:
            (keypoints1, keypoints2, confidence)
            - keypoints1: [N, 2] coordinates in image1
            - keypoints2: [N, 2] coordinates in image2
            - confidence: [N] confidence score per match
        """
        
        # Preprocess images
        img1_tensor = self._preprocess_image(image1)
        img2_tensor = self._preprocess_image(image2)
        
        # Run inference
        with torch.no_grad():
            results = self.model(img1_tensor, img2_tensor, max_keypoints=max_keypoints)
            
        # Parse results (depends on MatchAnything output format)
        keypoints1 = results['keypoints1'].cpu().numpy()  # [N, 2]
        keypoints2 = results['keypoints2'].cpu().numpy()  # [N, 2]
        confidence = results['confidence'].cpu().numpy()  # [N]
        
        return keypoints1, keypoints2, confidence
        
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for MatchAnything
        
        - Convert to correct color space (RGB)
        - Normalize pixel values
        - Resize if needed
        - Convert to torch tensor
        - Add batch dimension
        """
        # TODO: Use actual MatchAnything preprocessing
        pass
        
    def estimate_homography(self, 
                           keypoints1: np.ndarray, 
                           keypoints2: np.ndarray,
                           confidence: np.ndarray,
                           ransac_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate homography matrix using RANSAC
        
        Returns:
            (H, inlier_mask)
            - H: [3, 3] homography matrix
            - inlier_mask: [N] boolean mask of inliers
        """
        import cv2
        
        # Use OpenCV's RANSAC homography estimation
        H, mask = cv2.findHomography(
            keypoints1, 
            keypoints2,
            cv2.RANSAC,
            ransac_threshold
        )
        
        return H, mask.ravel().astype(bool)
```

**Dependencies** (`matchanything/requirements.txt`):
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.8.0
pillow>=9.0.0
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

**Goals:**
- Plugin scaffolding
- Model weight download system
- Device detection (CUDA/CPU)
- Basic UI structure

**Tasks:**
1. Set up QGIS plugin structure
2. Implement `ModelManager` class
   - CUDA detection
   - Weight download with progress
   - Model loading
3. Create main dialog UI (Qt Designer)
4. Implement progress dialog
5. Test weight download flow

**Deliverables:**
- Plugin loads in QGIS
- First-run download works
- Model loads successfully on GPU/CPU

---

### Phase 2: MatchAnything Integration (Week 2-3)

**Goals:**
- Get MatchAnything working in plugin context
- Test inference on sample images
- Validate device handling (GPU/CPU)

**Tasks:**
1. Study MatchAnything codebase
2. Implement `MatchAnythingInference` wrapper
3. Create simple test script (outside QGIS)
4. Integrate into `Matcher` class
5. Test single-scale matching
6. Profile performance (GPU vs CPU)

**Deliverables:**
- MatchAnything runs successfully
- Keypoints and confidence scores returned
- Performance benchmarks documented

---

### Phase 3: Tile Fetching & Coordinate Transforms (Week 3-4)

**Goals:**
- Capture QGIS map canvas
- Fetch basemap tiles
- Coordinate system transformations

**Tasks:**
1. Implement `TileFetcher` class
2. Canvas capture functionality
3. Tile URL generation and fetching
4. Tile stitching
5. Coordinate transformation utilities
6. Test with various CRS and zoom levels

**Deliverables:**
- Map canvas captured as image
- Tiles fetched and stitched correctly
- Coordinate transforms validated

---

### Phase 4: GCP Generation & Georeferencing (Week 4-5)

**Goals:**
- Convert matches to GCPs
- Integrate with QGIS georeferencer
- Output georeferenced rasters

**Tasks:**
1. Implement `GCPGenerator` class
2. Pixel-to-geographic coordinate transformation
3. GCP validation and distribution checks
4. Implement `Georeferencer` class
5. QGIS georeferencer API integration
6. Test end-to-end workflow

**Deliverables:**
- GCPs generated correctly
- Georeferenced outputs produced
- Transformation accuracy validated

---

### Phase 5: Progressive Refinement (Week 5-6)

**Goals:**
- Multi-scale matching
- Iterative refinement of matches
- Performance optimization

**Tasks:**
1. Implement progressive matching in `Matcher`
2. Coarse-to-fine workflow
3. Match filtering and refinement
4. Optimize for speed and memory
5. Test on challenging images

**Deliverables:**
- Progressive refinement working
- Improved accuracy over single-scale
- Reasonable performance on CPU

---

### Phase 6: UI Polish & Visualization (Week 6-7)

**Goals:**
- Confidence viewer implementation
- Match visualization
- User feedback and controls

**Tasks:**
1. Implement `ConfidenceViewer` dialog
2. Side-by-side image display
3. Keypoint visualization
4. Interactive filtering
5. Quality metrics display
6. Settings dialog

**Deliverables:**
- Full UI implemented
- Match results reviewable
- User can adjust parameters

---

### Phase 7: Testing & Documentation (Week 7-8)

**Goals:**
- Comprehensive testing
- User documentation
- Bug fixes

**Tasks:**
1. Create test dataset (various image types)
2. Test all workflows
3. Edge case handling
4. Write user manual
5. Create tutorial video/GIFs
6. Code documentation

**Deliverables:**
- Stable, tested plugin
- Complete documentation
- Known limitations documented

---

## Testing Strategy

### Unit Tests

**Test coverage for core components:**

```python
# tests/test_model_manager.py
def test_cuda_detection()
def test_weight_download()
def test_model_loading()
def test_inference_config_gpu()
def test_inference_config_cpu()

# tests/test_tile_fetcher.py
def test_canvas_capture()
def test_tile_url_generation()
def test_extent_to_tile_coords()
def test_tile_fetching()
def test_tile_stitching()

# tests/test_matcher.py
def test_single_scale_matching()
def test_progressive_matching()
def test_match_filtering()
def test_spatial_distribution_quality()

# tests/test_gcp_generator.py
def test_pixel_to_geo_transform()
def test_match_to_gcp_conversion()
def test_gcp_distribution_validation()

# tests/test_georeferencer.py
def test_transformation_type_suggestion()
def test_transform_error_calculation()
def test_georeferencing_workflow()
```

### Integration Tests

**End-to-end workflows:**

1. **Simple road map georeferencing**
   - Input: Scanned road map
   - Basemap: OSM Standard
   - Expected: Accurate georeferencing with 10+ GCPs

2. **Aerial photo matching**
   - Input: Historic aerial photo
   - Basemap: ESRI World Imagery
   - Expected: Successful matching despite temporal differences

3. **Rotated and skewed map**
   - Input: Rotated topographic map
   - Basemap: OSM Standard
   - Expected: Correct orientation recovery

4. **Low quality scan**
   - Input: Noisy, low-contrast map
   - Basemap: OSM Humanitarian (high contrast)
   - Expected: Robust matching despite quality issues

5. **CPU fallback test**
   - Force CPU mode
   - Run all above tests
   - Expected: Same results (slower)

### Performance Benchmarks

**Target metrics:**

| Hardware | Image Size | Processing Time | Acceptable? |
|----------|-----------|-----------------|-------------|
| GPU (4070Ti) | 2048×2048 | < 5 seconds | ✓ |
| GPU (4070Ti) | 4096×4096 | < 15 seconds | ✓ |
| CPU (8-core) | 2048×2048 | < 60 seconds | ✓ |
| CPU (8-core) | 4096×4096 | < 180 seconds | ✓ |

**Memory usage:**
- GPU: < 8 GB VRAM
- CPU: < 16 GB RAM

### Accuracy Validation

**Compare against manual georeferencing:**

1. Create ground truth dataset:
   - 10 test images
   - Manually place 20+ GCPs per image
   - Record accurate reference coordinates

2. Run plugin on same images

3. Compare results:
   - GCP position error (RMSE)
   - Coverage similarity
   - Visual alignment quality

**Target accuracy:**
- Mean GCP error: < 5 pixels at display resolution
- 90th percentile: < 10 pixels
- Comparable to skilled human georeferencing

---

## Known Limitations & Future Enhancements

### Current Limitations

1. **Model size**: ~800MB download required
2. **GPU requirement**: CPU is very slow for large images
3. **Basemap dependency**: Requires features visible in both images
4. **Temporal changes**: Struggles if landscapes changed dramatically
5. **Text-heavy maps**: May work poorly on label-dense maps without visual features
6. **Extreme rotations**: May need multiple matching attempts

### Future Enhancements

**Phase 2 Features:**

1. **Batch processing**
   - Process multiple images at once
   - Folder-based workflows
   - Progress tracking for batches

2. **Advanced matching modes**
   - Multi-basemap matching (try both OSM and aerial)
   - Ensemble methods (combine multiple model runs)
   - Manual GCP refinement mode

3. **Smart preprocessing**
   - Auto-rotation detection
   - Contrast enhancement for faded maps
   - Automatic basemap selection (analyze input image)

4. **Quality assurance**
   - Automatic accuracy reporting
   - Comparison against manual GCPs
   - Export quality reports

5. **Template library**
   - Save successful configurations
   - Share matching templates
   - Auto-detect map series/publishers

6. **Cloud processing**
   - Optional cloud API for users without GPU
   - Faster processing for CPU-only systems
   - Usage tracking and quotas

7. **Fine-tuning support**
   - Train on user's specific map types
   - Improve accuracy for niche datasets
   - Export fine-tuned models

---

## Deployment Checklist

### Pre-release

- [ ] All unit tests passing
- [ ] Integration tests validated
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Tutorial materials created
- [ ] Code reviewed
- [ ] License compliance checked (MatchAnything, dependencies)

### Plugin Metadata

```ini
# metadata.txt
[general]
name=Magic Georeferencer
qgisMinimumVersion=3.22
description=AI-powered automatic image georeferencing using MatchAnything
version=1.0.0
author=Your Name
email=your.email@example.com

about=Automatically georeference maps, aerial photos, and sketches using 
      the MatchAnything deep learning model. Simply navigate to the approximate 
      location and let AI find the matching features.

tracker=https://github.com/yourusername/magic-georeferencer/issues
repository=https://github.com/yourusername/magic-georeferencer
tags=georeferencing, machine learning, AI, automation, image matching

homepage=https://github.com/yourusername/magic-georeferencer
category=Raster
icon=icon.png
experimental=False
deprecated=False

# Model attribution
credits=Built on MatchAnything by He et al. (2025)
        https://github.com/XingyiHe/MatchAnything
```

### QGIS Plugin Repository

1. **Packaging**
   - Create ZIP with correct structure
   - Exclude development files (.git, tests, etc.)
   - Include only necessary dependencies

2. **Upload**
   - Submit to QGIS Plugin Repository
   - Provide screenshots
   - Write clear description

3. **Documentation**
   - README with quick start
   - Installation instructions
   - Troubleshooting guide
   - Link to full documentation

### User Support

**Resources:**
- GitHub wiki for documentation
- Issue tracker for bugs
- Discussion forum for questions
- Video tutorials
- Example datasets

**Common issues:**
- CUDA not found (driver installation)
- Out of memory (reduce image size)
- No matches found (wrong location or basemap)
- Poor accuracy (need manual refinement)

---

## Development Environment Setup

### Prerequisites

```bash
# QGIS 3.22+ with Python 3.9+
# CUDA Toolkit 11.7+ (for GPU support)
# Git
```

### Initial Setup

```bash
# Clone MatchAnything
git clone https://github.com/XingyiHe/MatchAnything.git
cd MatchAnything

# Create conda environment
conda env create -f environment.yaml
conda activate env

# Download weights
# (Download link from MatchAnything repo)
# Place in weights/ directory

# Test MatchAnything standalone
python test_inference.py  # Create simple test script
```

### Plugin Development

```bash
# Create plugin directory
mkdir -p ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/magic_georeferencer
cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/magic_georeferencer

# Initialize plugin structure
# (Follow structure defined in Architecture section)

# Install development dependencies
pip install pytest pytest-qgis black flake8

# Enable plugin in QGIS
# Plugins -> Manage and Install Plugins -> Installed -> Enable Magic Georeferencer
```

### Development Workflow

1. Make code changes
2. Reload plugin in QGIS (Plugin Reloader plugin is helpful)
3. Test in QGIS
4. Run unit tests: `pytest tests/`
5. Format code: `black .`
6. Lint: `flake8 .`
7. Commit changes

---

## Technical Notes

### Coordinate System Considerations

**CRS Handling:**
- Input image: No CRS (ungeoreferenced)
- Map canvas: Project CRS
- Basemap tiles: Usually EPSG:3857 (Web Mercator)
- Output: User-selected target CRS

**Transformation pipeline:**
1. Basemap tile pixel coords → EPSG:3857 (or tile native CRS)
2. EPSG:3857 → Project CRS (if different)
3. Project CRS → Target output CRS
4. Apply to GCPs for georeferencing

### Memory Management

**Large image handling:**
- Max recommended size: 8192×8192 pixels
- For larger: downsample for matching, upsample GCPs
- Progressive refinement helps (start small)

**GPU memory:**
- Monitor VRAM usage
- Batch size = 1 (one image pair at a time)
- Clear cache between runs

### Error Handling

**Graceful degradation:**
- If GPU fails → fallback to CPU automatically
- If download fails → offer retry or manual download
- If matching fails → suggest different basemap or manual GCPs
- If too few GCPs → warn user, suggest lowering threshold

**User feedback:**
- Always show progress
- Estimate time remaining
- Explain failures in plain language
- Offer actionable next steps

---

## Appendix: MatchAnything Details

### Model Architecture

Based on the README, MatchAnything is a:
- **Cross-modality** image matching model
- Trained on large-scale diverse datasets
- Handles visible, thermal, SAR, vectorized maps, medical images
- Produces keypoint correspondences with confidence scores
- Pre-trained weights available

### Input/Output Format

**Expected Input:**
- Two images (numpy arrays or PIL Images)
- RGB format
- No specific size requirement (model handles resizing)

**Expected Output:**
```python
{
    'keypoints0': torch.Tensor,  # [N, 2] (x, y) in image 0
    'keypoints1': torch.Tensor,  # [N, 2] (x, y) in image 1
    'confidence': torch.Tensor,  # [N] match confidence
    'matching_scores': torch.Tensor  # Additional scoring info
}
```

### Evaluation Metrics (from paper)

- Position accuracy (RMSE)
- Match precision/recall
- Geometric consistency (post-RANSAC inlier ratio)

---

## Success Criteria

### Minimum Viable Product (MVP)

- ✓ Plugin installs in QGIS without errors
- ✓ Model weights download successfully
- ✓ Can load ungeoreferenced image
- ✓ Can capture map canvas
- ✓ Produces matches between images
- ✓ Generates valid GCPs
- ✓ Outputs georeferenced raster
- ✓ Works on both GPU and CPU

### Quality Thresholds

- ≥80% success rate on test dataset
- ≥10 GCPs per successful match
- Mean GCP error ≤5 pixels
- Processing time ≤60s on CPU (2048px image)
- User satisfaction score ≥4/5

### Documentation

- README with quick start guide
- Full user manual
- API documentation for developers
- Tutorial with screenshots/video
- Troubleshooting guide

---

## Questions for Clarification

### Before Starting Development

1. **MatchAnything specifics:**
   - Exact model architecture used (ELoFTR-based?)
   - Input preprocessing requirements
   - Output format details
   - Any undocumented limitations

2. **QGIS API:**
   - Best approach for georeferencer integration
   - Canvas capture best practices
   - GCP object format requirements

3. **User requirements:**
   - Target accuracy expectations (pixel-level)
   - Acceptable processing time limits
   - Must-have vs nice-to-have features

4. **Distribution:**
   - License compatibility (MatchAnything, dependencies)
   - Plugin repository guidelines
   - Weight hosting solution (HuggingFace is good?)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-14 | Initial specification |

---

## Contact & Contribution

- **Developer:** [Your Name]
- **Email:** [Your Email]
- **Repository:** https://github.com/yourusername/magic-georeferencer
- **Issues:** https://github.com/yourusername/magic-georeferencer/issues

**Contributions welcome!** Please see CONTRIBUTING.md for guidelines.

---

*This specification is a living document and will be updated as development progresses.*

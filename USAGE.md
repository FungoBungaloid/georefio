# Magic Georeferencer - Usage Guide

## Quick Start

### Basic Workflow

1. **Load an ungeoreferenced image** (map, aerial photo, sketch)
2. **Navigate QGIS map** to the approximate location
3. **Configure matching** (basemap source, quality settings)
4. **Click "Match & Generate GCPs"** and let AI do the work
5. **Review results** and save georeferenced output

That's it! The plugin automates the entire georeferencing process.

## Detailed Step-by-Step Guide

### Step 1: Open the Plugin

**Method 1:** Click the Magic Georeferencer icon in the toolbar
**Method 2:** Go to `Raster � Magic Georeferencer`

The main dialog will open.

### Step 2: Load Ungeoreferenced Image

1. Click the **"Browse..."** button
2. Select your image file:
   - Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP
   - Common uses:
     - Historical maps
     - Scanned paper maps
     - Aerial photographs
     - Hand-drawn sketches
     - Satellite imagery without coordinates

3. The image size will be displayed
4. Large images will work but may take longer to process

**Tips:**
- Images should be reasonably clear and have identifiable features
- Higher resolution images may provide better accuracy
- The image doesn't need to be oriented correctly - the AI will figure it out

### Step 3: Navigate to the Location

This is crucial for successful matching!

1. **In the QGIS map canvas**, navigate to where your image is located:
   - Use the Pan tool to move around
   - Use Zoom In/Out to adjust the view
   - Load basemap layers if needed (OSM, satellite imagery)

2. **Match the scale roughly**:
   - The visible map extent will be compared to your image
   - Try to zoom so the coverage is similar
   - Don't worry about being pixel-perfect

3. **Optional: Enable overlay preview** (if available)
   - Check "Show image overlay preview"
   - This helps you position the image correctly
   - Adjust transparency to see both map and image

**Tips:**
- Start with a wider view if unsure of exact location
- You can be off by several kilometers and still get matches
- The AI is quite forgiving with positioning

### Step 4: Select Basemap Source

Choose the basemap that best matches your image type:

#### OSM Standard (Default)
- **Best for:** Road maps, city maps, topographic maps
- **Features:** Roads, buildings, labels, administrative boundaries
- **When to use:** Your image shows streets, buildings, or infrastructure

#### ESRI World Imagery
- **Best for:** Aerial photos, satellite imagery
- **Features:** High-resolution aerial/satellite views
- **When to use:** Your image is an aerial photograph or satellite image

#### OSM Humanitarian
- **Best for:** High-contrast simplified maps
- **Features:** Simplified, high-contrast rendering
- **When to use:** Your image has strong features but poor quality

**Basemap Selection Tips:**
- Match the **type** not the **age** of imagery
- For historical maps showing roads/buildings, use OSM Standard
- For old aerial photos, use ESRI World Imagery (features often haven't changed)
- You can try multiple basemaps if the first doesn't work well

### Step 5: Configure Match Quality

Select a quality preset:

#### Strict (0.85 threshold)
- **Use when:** You want only the highest confidence matches
- **Pros:** Very accurate, minimal false matches
- **Cons:** May find fewer GCPs, may fail on difficult images
- **Recommended for:** High-quality images with clear features

#### Balanced (0.70 threshold) - DEFAULT
- **Use when:** Most situations
- **Pros:** Good balance of quantity and quality
- **Cons:** May include some lower-confidence matches
- **Recommended for:** General use, typical map georeferencing

#### Permissive (0.55 threshold)
- **Use when:** Image is poor quality or very different from basemap
- **Pros:** Finds more matches, works on difficult cases
- **Cons:** More false matches possible, requires review
- **Recommended for:** Sketches, heavily degraded images, cross-modality matching

**Progressive Refinement** (checkbox):
-  **Enabled (recommended):** Multi-scale matching for better accuracy
  - **How it works:** Starts matching at low resolution for speed, then refines at higher resolutions for accuracy
  - **Advantages:**
    - More accurate results - refines match locations progressively
    - Better handling of scale differences between your image and the map
    - More robust for challenging images (poor quality, unusual perspectives)
    - Helps filter out false matches early
  - **Tradeoffs:**
    - 2-3x slower than single-scale (still usually under 30 seconds with GPU)
    - Uses slightly more memory
  - **Best for:** Historical maps, degraded images, complex scenes, when accuracy matters more than speed

-  **Disabled:** Single-scale matching
  - **How it works:** Processes at one resolution (832px) only
  - **Advantages:**
    - Faster processing (5-10 seconds with GPU vs 15-30 seconds with progressive)
    - Lower memory usage
    - Sufficient for most simple georeferencing tasks
  - **Tradeoffs:**
    - May miss some matches or be less accurate
    - Less robust to scale mismatches
  - **Best for:** High-quality modern images, when you need quick results, batch processing, or working with limited hardware resources

### Step 6: Run Matching

1. Click **"Match & Generate GCPs"**
2. Watch the progress dialog:
   - Loading source image...
   - Capturing basemap...
   - Running AI matching...
   - Filtering matches...
   - Generating Ground Control Points...
   - Georeferencing...

**Processing Time:**
- **GPU (NVIDIA):** 5-15 seconds for typical images
- **CPU:** 1-3 minutes for typical images
- Larger images or progressive refinement take longer

### Step 7: Review Results (Confidence Viewer)

If enabled in settings, the Confidence Viewer will show:

**Statistics:**
- Total matches found
- Mean confidence score
- Geometric consistency score
- Spatial distribution quality

**Visual Display:**
- Side-by-side view of source image and basemap
- Matching keypoints highlighted
- Connection lines between matches
- Color-coded by confidence (red=low, green=high)

**Controls:**
- Adjust confidence threshold slider
- Remove individual bad matches
- Accept or reject the results

**What to Look For:**
-  Matches spread across the image (not clustered)
-  High confidence scores (>0.7)
-  Visually correct correspondences
-  All matches in one corner
-  Obviously wrong connections

### Step 8: Save Georeferenced Image

1. Choose output file location
2. Default name: `[original_name]_georef.tif`
3. Format: GeoTIFF (standard georeferenced format)

The plugin will:
- Automatically select the best transformation type
- Apply the transformation
- Create a georeferenced GeoTIFF
- Load it into your QGIS project

**Success Dialog Shows:**
- Output file path
- Number of matches used
- Mean confidence score
- Transform type applied
- Confirmation that layer was added to map

## Understanding Results

### Match Statistics

**Total Matches:**
- Minimum needed: 4-6 (depending on transform type)
- Typical: 10-50 matches
- Good: 20+ matches
- Excellent: 50+ matches

**Mean Confidence:**
- < 0.5: Poor - likely to fail
- 0.5-0.7: Moderate - may work
- 0.7-0.85: Good - reliable
- > 0.85: Excellent - very reliable

**Distribution Quality:**
- 0.0-0.3: Poor - all clustered
- 0.3-0.5: Fair - somewhat spread
- 0.5-0.7: Good - well distributed
- 0.7-1.0: Excellent - evenly distributed

### Transform Types

The plugin automatically selects the best transform:

**Polynomial 1st Order (Affine)**
- Requires: 4-6 GCPs
- Suitable for: Scanned maps, simple distortion
- Preserves: Parallel lines
- Use when: Image is relatively undistorted

**Polynomial 2nd Order**
- Requires: 6-10 GCPs
- Suitable for: Maps with some warping
- Preserves: General shape
- Use when: Moderate distortion present

**Polynomial 3rd Order**
- Requires: 10+ GCPs
- Suitable for: Complex distortion
- Preserves: Overall structure
- Use when: Significant warping

**Thin Plate Spline**
- Requires: 10+ GCPs
- Suitable for: Localized distortions
- Preserves: Local features
- Use when: Excellent GCP distribution and many points

## Advanced Usage

### Batch Processing

For multiple images:

1. Georeference one image to establish best settings
2. Note the successful basemap and quality settings
3. Process remaining images with same settings
4. Review each result in confidence viewer

### Manual Refinement

If automatic matching partially succeeds:

1. Let the plugin generate initial GCPs
2. Open QGIS Georeferencer manually
3. Load the generated .points file
4. Add or adjust GCPs manually
5. Complete georeferencing

### Different Map Projections

For maps in specific projections:

1. Complete automatic matching
2. Note the transform type used
3. Manually adjust output CRS if needed
4. Re-run georeferencer with correct target CRS

### Quality Assurance

Verify georeferencing quality:

1. **Visual inspection:**
   - Load OpenStreetMap or satellite basemap
   - Add your georeferenced image
   - Check alignment of features
   - Look for systematic errors

2. **Measure check:**
   - Use QGIS measurement tool
   - Compare distances to known values
   - Check angles and orientations

3. **Accuracy assessment:**
   - Create additional GCPs manually
   - Compare to automatic GCP positions
   - Calculate RMSE (Root Mean Square Error)

## Troubleshooting

### "Insufficient Matches" Warning

**Problem:** Plugin finds fewer than 6 matches

**Solutions:**
1. **Lower quality threshold:** Switch from Strict to Balanced or Permissive
2. **Change basemap:** Try a different basemap source
3. **Zoom adjustment:** Make sure map view matches image scale
4. **Location check:** Verify you're looking at the right place
5. **Enable progressive refinement:** More thorough matching
6. **Image quality:** Ensure image has identifiable features

### Poor Alignment Results

**Problem:** Georeferenced image doesn't align well

**Solutions:**
1. **Review confidence viewer:** Check if matches look correct
2. **More matches needed:** Try permissive mode
3. **Better distribution:** Ensure matches spread across image
4. **Different transform type:** Try polynomial 2 or thin plate spline
5. **Manual refinement:** Add/adjust GCPs manually

### Slow Processing

**Problem:** Matching takes a very long time

**Solutions:**
1. **Use GPU:** Install CUDA and PyTorch with GPU support
2. **Disable progressive refinement:** Faster single-scale matching
3. **Reduce image size:** Resize very large images before processing
4. **Close other applications:** Free up CPU/GPU resources

### Wrong Location Matches

**Problem:** Matches found but in wrong geographic area

**Solutions:**
1. **Check map position:** You may be looking at wrong location
2. **Verify image contents:** Ensure it's the location you think
3. **Unique features:** Image location must have distinguishable features
4. **Try different basemap:** Some areas map better with different sources

### Model Not Loading

**Problem:** "Model not loaded" error

**Solutions:**
1. **Check dependencies:** Ensure PyTorch and transformers installed
2. **Re-download model:** Delete weights folder, restart plugin
3. **Check disk space:** Model requires ~500 MB
4. **Internet connection:** Model downloads from HuggingFace
5. **Check error log:** QGIS Python Console � Show Errors

## Tips for Best Results

### Image Preparation

 **Do:**
- Use highest resolution available
- Ensure image is clear and readable
- Crop to area of interest
- Remove borders/legends if possible

 **Don't:**
- Over-compress JPEGs
- Use extremely low resolution
- Include large blank areas
- Use heavily degraded images without trying cleanup

### Map Selection

 **Do:**
- Match basemap type to image type
- Use current basemaps (features often unchanged)
- Try multiple basemaps if first fails
- Consider time period (but not critical)

 **Don't:**
- Assume old maps need old basemaps
- Give up after one basemap attempt
- Use basemaps from wrong region

### Location Positioning

 **Do:**
- Get approximate location right
- Match scale roughly
- Include some buffer area
- Use landmarks to orient

 **Don't:**
- Expect pixel-perfect positioning
- Zoom too far in or out
- Position map inverted from image

## Example Workflows

### Workflow 1: Historical Road Map

1. Load scanned 1950s city map
2. Navigate QGIS to modern city location
3. Select "OSM Standard" basemap
4. Use "Balanced" quality
5. Enable progressive refinement
6. Process and review
7. Roads match well - save result

### Workflow 2: Aerial Photograph

1. Load aerial photo from 1980s
2. Navigate to approximate area
3. Select "ESRI World Imagery"
4. Use "Strict" quality (aerial photos match well)
5. Process quickly with single-scale
6. Review shows excellent alignment
7. Save georeferenced image

### Workflow 3: Hand-drawn Sketch

1. Load field sketch map
2. Navigate to GPS-recorded location
3. Select "OSM Humanitarian" (high contrast)
4. Use "Permissive" quality
5. Enable progressive refinement
6. Review carefully, remove bad matches
7. Acceptable alignment for sketch - save

## Getting Help

### Resources

- **Documentation:** README.md, INSTALL.md, this file
- **QGIS docs:** https://qgis.org/en/docs/
- **HuggingFace model:** https://huggingface.co/zju-community/matchanything_eloftr

### Support

- **GitHub Issues:** Report bugs and request features
- **Discussions:** Ask questions and share experiences
- **QGIS Community:** General QGIS georeferencing questions

### Contributing

Found a way to improve the workflow? Submit a pull request or open an issue!

---

**Happy Georeferencing!** =�(

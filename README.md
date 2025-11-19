# Magic Georeferencer

![Logo](magic_georeferencer/icon.png)

**AI-Powered Automatic Image Georeferencing for QGIS**

Magic Georeferencer is a QGIS plugin that leverages the MatchAnything deep learning model to automatically georeference ungeoreferenced images (maps, aerial photos, sketches) by matching them against real-world basemap data.

## Features

- **Zero Manual GCP Placement** - Fully automated feature matching
- **Cross-Modality Support** - Handles maps to aerial, aerial to maps, sketches to maps, etc.
- **Progressive Refinement** - Multi-scale matching for improved accuracy
- **GPU Acceleration** - CUDA support with automatic CPU fallback
- **Confidence Visualization** - Review matches before committing
- **Flexible Basemap Sources** - OSM standard, aerial imagery, and more

## Quick Start

> **Full Documentation**: See [INSTALL.md](INSTALL.md) for detailed installation instructions and [USAGE.md](USAGE.md) for comprehensive usage guide.

### Installation

**Recommended: Install from QGIS Plugin Repository**

1. Open QGIS
2. Go to `Plugins > Manage and Install Plugins`
3. Search for "Magic Georeferencer"
4. Click "Install Plugin"
5. On first run, the plugin will prompt to download AI model weights (~400 MB)

**Alternative: Manual Installation from ZIP**

If the plugin is not yet available in the repository, or you need a development version:

1. Download the repository as a ZIP file from GitHub
2. Unzip the downloaded file
3. Inside the extracted folder, find the `magic_georeferencer` directory
4. Create a ZIP file containing **only** the `magic_georeferencer` folder:
   - Windows: Right-click `magic_georeferencer` > "Send to" > "Compressed (zipped) folder"
   - macOS: Right-click `magic_georeferencer` > "Compress"
   - Linux: `zip -r magic_georeferencer.zip magic_georeferencer/`
5. In QGIS: `Plugins > Manage and Install Plugins > Install from ZIP`
6. Select your `magic_georeferencer.zip` and click "Install Plugin"

**First Run Setup**

- The plugin will automatically download model weights (~400 MB) from HuggingFace on first use
- This is a one-time download with automatic caching
- GPU acceleration will be auto-detected if available

### Basic Usage

1. **Load Ungeoreferenced Image**
   - Click the Magic Georeferencer icon in the toolbar
   - Click "Browse" and select your image (map, aerial photo, etc.)

2. **Navigate Map Canvas**
   - In the QGIS map canvas, zoom to the approximate location of your image
   - The current map view will be used for matching

3. **Configure Matching**
   - Select appropriate basemap source:
     - **OSM Standard**: Best for road maps, building footprints
     - **ESRI World Imagery**: Best for aerial photos, satellite imagery
   - Choose quality preset (Strict/Balanced/Permissive)
   - Enable progressive refinement for better accuracy (slower)

4. **Run Matching**
   - Click "Match & Generate GCPs"
   - Review matches in the confidence viewer
   - Accept to georeference the image
   - Output will be in your project's CRS

## Requirements

### System Requirements

- **QGIS**: Version 3.22 or higher
- **Python**: 3.9+ (bundled with QGIS)
- **Operating System**:
  - Windows 10/11 (64-bit)
  - macOS 10.15+ (Catalina or later)
  - Linux (Ubuntu 20.04+, Fedora 33+, or equivalent)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk Space**: ~2 GB for model weights and cache

### Optional (for GPU acceleration)

- **NVIDIA GPU** with CUDA Compute Capability 3.5+
- **CUDA Toolkit**: 11.7 or higher
- **GPU RAM**: 4 GB minimum, 8 GB recommended
- **Note**: GPU acceleration currently only available on Windows and Linux with NVIDIA CUDA. macOS users (including M1/M2) will use CPU mode (fully functional, just slower).

### Python Dependencies

The plugin requires the following Python packages (automatically installed):

- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `transformers >= 4.30.0`
- `huggingface-hub >= 0.16.0`
- `scipy >= 1.9.0`
- `numpy >= 1.21.0`
- `opencv-python >= 4.8.0`
- `pillow >= 9.0.0`

## Known Limitations & When It May Struggle

While Magic Georeferencer works well for most standard georeferencing tasks, there are scenarios where it may struggle or fail. Understanding these limitations helps you know when to use the tool and when manual georeferencing might be more appropriate.

### Image Types That May Cause Issues

**Very Stylized or Artistic Maps:**
- Hand-drawn artistic maps with minimal realistic features
- Highly abstracted or schematic representations
- Fantasy/fictional maps with no real-world correspondence
- **Workaround:** Try using a permissive quality threshold, or add manual GCPs alongside AI-generated ones

**Low Contrast or Degraded Images:**
- Severely faded historical documents
- Poor quality scans with low dynamic range
- Overexposed or underexposed photographs
- **Workaround:** Pre-process images to enhance contrast; use OSM Humanitarian basemap for higher contrast matching

**Very Large Geographic Areas:**
- Continental or global-scale maps
- Small-scale maps where detail is minimal
- Maps covering thousands of kilometers
- **Why:** Feature matching requires sufficient local detail, which decreases at very small scales
- **Workaround:** Focus on a specific region of the map, or use manual GCPs for large-scale maps

**Unusual or Non-Standard Projections:**
- Polar projections (stereographic, azimuthal)
- Highly distorted artistic projections
- Historical projections not commonly used today
- **Why:** The basemap is in Web Mercator (EPSG:3857), which has high distortion at poles
- **Workaround:** Manually georeference or choose a region within reasonable latitudes (< 70 degrees)

**Polar and High-Latitude Regions:**
- Maps of Arctic or Antarctic regions
- Areas above 80 degrees latitude
- **Why:** Web Mercator projection (used by tile services) doesn't cover polar regions well
- **Workaround:** Use manual georeferencing or specialized polar basemaps

**Maps with Extreme Rotation or Perspective:**
- Severely rotated scans (>45 degrees off-axis can be challenging)
- Oblique aerial photographs
- Bird's-eye perspective drawings
- **Note:** The AI can handle moderate rotation, but extreme cases may need manual orientation first

### Other Challenging Scenarios

**Temporal Mismatches:**
- Historical maps showing areas that have completely changed (new development, natural disasters, etc.)
- **Example:** A 1950s map of a city that has been completely rebuilt
- **Workaround:** Look for unchanged features (rivers, coastlines, major roads)

**Text-Only or Label-Heavy Maps:**
- Maps with mostly text and minimal geographic features
- Cadastral maps with primarily boundary lines and labels
- **Why:** The model works best with visual features, not text recognition

**Nighttime or Specialized Imagery:**
- Nighttime satellite imagery (lights)
- Thermal imagery
- Radar/SAR imagery
- **Note:** Cross-modality matching can work but may be less reliable

### Success Tips

To maximize your chances of success:

1. **Start with clear, well-preserved images** when possible
2. **Match modality:** Use aerial basemaps for aerial images, road maps for road maps
3. **Try multiple basemaps** if the first doesn't work (OSM Standard, ESRI Imagery, OSM Humanitarian)
4. **Use progressive refinement** for challenging images
5. **Adjust quality thresholds:** Start with Balanced, try Permissive for difficult cases
6. **Pre-process images:** Enhance contrast, crop to relevant area, straighten severe rotation
7. **Focus on unchanged features:** Rivers, coastlines, mountain peaks often persist across time

### When to Use Manual Georeferencing Instead

Consider manual GCP placement when:
- Map is purely schematic or artistic
- Image quality is extremely poor
- Coverage is global/continental scale
- Projection is highly unusual
- Area has changed completely since map creation
- You need sub-meter accuracy for scientific applications

You can also use a **hybrid approach**: Let the AI generate initial GCPs, then manually add or refine them in QGIS's georeferencer.

## Troubleshooting

### Model weights won't download

- Check your internet connection
- Verify firewall/proxy settings aren't blocking HuggingFace
- Model downloads automatically from [HuggingFace Hub](https://huggingface.co/zju-community/matchanything_eloftr)
- Cached in `magic_georeferencer/weights/` directory
- Try clearing cache and restarting QGIS if download fails

### CUDA not detected

- Ensure NVIDIA drivers are installed
- Install CUDA Toolkit 11.7+
- Restart QGIS after installation
- CPU mode will be used automatically as fallback

### No matches found

- Verify you navigated to the correct location on the map
- Try a different basemap source (e.g., aerial vs. road map)
- Lower the quality threshold (use Permissive instead of Strict)
- Ensure your image has sufficient visual features

### Poor accuracy

- Enable progressive refinement
- Use more GCPs (lower confidence threshold)
- Try a different transform type
- Manually refine GCPs in QGIS georeferencer

## Use of Esri World Imagery

This tool can optionally access Esri World Imagery as a supporting basemap layer.
No Esri imagery, tiles, or data are included in this repository.

Use of Esri World Imagery is subject to the Esri ArcGIS Online Terms of Use and any Access and Use Constraints specified on the corresponding ArcGIS Online item page:

- Esri World Imagery (ArcGIS Online item): https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9
- Terms of Use: https://www.esri.com/legal/terms/full

Users of this tool are responsible for ensuring their own compliance with Esri's licensing and attribution requirements.

Required attribution for Esri World Imagery (from the item's "Credits" field):
"Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community."

## Credits

This plugin is built on top of:

- **MatchAnything** by He et al. (2025) - [GitHub](https://github.com/zju3dv/MatchAnything) - Licensed under Apache 2.0
- **QGIS** - Open Source Geographic Information System
- **PyTorch** - Deep learning framework
- **OpenStreetMap** - Map tile provider

## Development

This plugin was co-developed using [Claude Code](https://www.anthropic.com/claude-code), Anthropic's AI-powered software development tool.

## License

This project is released under the Apache License 2.0.

See [LICENSE](LICENSE) for details.

The underlying MatchAnything model is provisionally licensed under Apache 2.0 as per [this discussion](https://github.com/zju3dv/MatchAnything/issues/2).

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/FungoBungaloid/georefio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/FungoBungaloid/georefio/discussions)

---

**Made with care for the QGIS community**

# Plugin Icon

## SVG Icon
The plugin includes an SVG icon (`icon.svg`) that represents:
- A map/document (white rectangle with folded corner)
- Grid lines symbolizing georeferencing
- A magic wand with sparkles representing AI automation
- Neural network nodes symbolizing the AI model

## Using the Icon in QGIS
QGIS can use SVG icons directly. The `icon.svg` file will work as-is.

## Creating PNG Icon (Optional)
If you need a PNG version, you can convert the SVG using:

### Using ImageMagick:
```bash
convert -background none icon.svg -resize 64x64 icon.png
```

### Using Inkscape:
```bash
inkscape icon.svg --export-filename=icon.png --export-width=64 --export-height=64
```

### Using Python (cairosvg):
```python
import cairosvg
cairosvg.svg2png(url='icon.svg', write_to='icon.png', output_width=64, output_height=64)
```

## Icon Design
- Size: 64x64 pixels
- Format: SVG (scalable) or PNG
- Colors: Blue gradient (#4A90E2 to #357ABD) for professionalism
- Gold (#FFD700) accents for the "magic" elements
- Clean, modern design suitable for GIS software

# Bulk Georeferencing Guide

The **Batch Georeference** tool (`Raster > Magic Georeferencer > Batch
Georeference`) processes an entire library of images. Every image needs an
*approximate* location before the feature matcher can refine it into a
precise georeference. This document explains where those approximate
locations can come from and how to prepare your data.

## How location resolution works

For each image, the plugin tries these sources **in order** and uses the
first one that produces a result:

| Order | Source | Requires | Cost |
|-------|--------|----------|------|
| 1 | Capture-path / manifest file | A CSV, GeoJSON, or GPX file you supply | Free |
| 2 | Existing rough georeferencing | World file sidecar or embedded GeoTIFF tags | Free |
| 3 | EXIF GPS | GPS tags embedded in the image (typical for drone/aerial photos) | Free |
| 4 | Vision AI | An OpenAI or Anthropic API key | ~$0.001–0.03 per image |

The batch summary reports how many images were located by each source, so
you can see at a glance whether your metadata is being picked up.

**Why prefer metadata over the vision AI?** It's free, it's deterministic,
and for surveys it's usually far more accurate than asking a model to
recognize the location from pixels. The vision AI remains useful as a
fallback for scanned maps and images with no metadata at all, and its
accuracy will improve as vision models improve — the pipeline treats it as
just another location source, so nothing else changes when it gets better.

## Roughly georeferenced image libraries

Many large surveys (historical aerial photography, orthophoto archives,
scanned map series) are distributed as images that are already *roughly*
georeferenced. The plugin detects two common conventions automatically —
no configuration needed:

### GeoTIFF tags

If your `.tif` files carry an embedded geotransform, the plugin reads the
footprint directly (including the CRS, which is reprojected to WGS84
automatically). This is the most reliable input.

### World files

Sidecar text files next to each image containing six lines (pixel size,
rotation terms, and the origin coordinate):

```
photo_0431.jpg
photo_0431.jgw     <- world file (also accepted: .wld, .pgw, .tfw, .jpgw, ...)
photo_0431.prj     <- optional CRS definition
```

**Caveat:** world files carry no CRS. If a `.prj` sidecar exists it is
honored; otherwise the plugin assumes **EPSG:4326** (longitude/latitude).
If your survey uses a projected CRS (UTM is common), either add `.prj`
files or the coordinates will be rejected with a warning in the log. The
derived footprint is expanded by ~30% to tolerate rough georeferencing.

## Capture-path / manifest files

Surveys are also commonly stored as a folder of plain images plus a
separate record of the flight or capture path. Because there is a lot of
variance in how these are distributed, the plugin accepts three formats
and is deliberately liberal about naming. Select the file under
**Location Sources > Capture path / manifest** in the batch dialog.

Image filenames in manifests are matched **case-insensitively and without
extensions**, so `IMG_0042`, `img_0042.jpg`, and `IMG_0042.TIF` all refer
to the same record.

### CSV

One row per image. Comma, semicolon, and tab delimiters are auto-detected.
Column names are matched case-insensitively against these aliases:

| Field | Accepted column names |
|-------|----------------------|
| Filename | `filename`, `file`, `image`, `image_name`, `name`, `photo`, `frame` |
| Latitude | `lat`, `latitude`, `y`, `gps_lat`, `center_lat` |
| Longitude | `lon`, `lng`, `long`, `longitude`, `x`, `gps_lon`, `center_lon` |
| Extent (optional) | `extent_m`, `extent`, `footprint_m`, `width_m`, `coverage_m` |

```csv
filename,lat,lon,extent_m
IMG_0001.jpg,48.8584,2.2945,500
IMG_0002.jpg,48.8601,2.2903,500
IMG_0003.jpg,48.8617,2.2861,
```

- Coordinates must be WGS84 decimal degrees.
- `extent_m` is the approximate ground width covered by the frame, in
  meters. Rows without it use the **"Assumed footprint for point
  locations"** setting from the dialog (default 1000 m).

If your survey's log uses different column names, the fastest fix is to
rename the header row — no code changes needed.

### GeoJSON

A `FeatureCollection` of `Point` features with a filename-like property:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [2.2945, 48.8584]},
      "properties": {"filename": "IMG_0001.jpg", "extent_m": 500}
    }
  ]
}
```

The same property-name aliases as CSV apply. Note GeoJSON coordinate
order is `[longitude, latitude]`.

### GPX (flight tracks)

Two GPX mechanisms are supported, and they can be combined in one file:

1. **Waypoints** — a `<wpt>` whose `<name>` matches the image filename
   (with or without extension) pins that image directly.
2. **Track interpolation** — if the GPX contains a `<trk>` with
   timestamped track points, images *without* a waypoint are positioned by
   interpolating along the track using their EXIF `DateTimeOriginal`
   timestamp. This matches the common workflow of flying with a GPS logger
   and a separate camera.

**Timestamp caveats for track interpolation:**

- EXIF timestamps carry no timezone. The plugin uses the EXIF
  `OffsetTimeOriginal` tag when present; otherwise timestamps are treated
  as UTC. If your camera clock was set to local time, positions will be
  shifted along the track — either fix the timestamps (e.g. with
  `exiftool -AllDates+=...`) or convert the track to per-image waypoints.
- Images timestamped more than 10 minutes outside the recorded track are
  rejected rather than snapped to a track endpoint.

## Formats not (yet) directly supported

If your survey uses one of these, convert to CSV first — it's usually a
one-liner:

- **Pix4D / Agisoft / drone flight logs** — export the camera positions
  table as CSV and rename the header columns to the aliases above.
- **Shapefile footprints / flight indexes** — in QGIS: open the layer,
  add `lat`/`lon` fields with the centroid coordinates (`x(centroid($geometry))`
  in the field calculator, in EPSG:4326), then `Export > Save Features As... > CSV`.
- **KML/KMZ placemark indexes** — open in QGIS and export to CSV or
  GeoJSON as above.
- **ESRI world-file variants with projected CRS but no .prj** — create
  `.prj` files containing the CRS (e.g. the WKT for your UTM zone), or
  batch-convert the images to GeoTIFF with `gdal_translate -a_srs`.

The provider architecture (`core/location_providers.py`) is deliberately
pluggable: each source implements a single `get_hint(image_path)` method,
so adding a new format is a small, isolated change. Issues/PRs with sample
files of other real-world formats are welcome.

## Tuning tips for large batches

- **Assumed footprint**: set this close to the real ground coverage of a
  single frame. Too small and the matcher sees only the center of the
  frame; too large and features are too small to match. The matcher
  searches a range of scales around the estimate (±2 steps of √2 by
  default), so it tolerates being off by ~2x.
- **Quality backoff**: leave automatic mode on. Matching runs once per
  image; only the confidence filtering is retried at progressively relaxed
  thresholds, so backoff is cheap.
- **Basemap**: in automatic mode, images located via EXIF GPS default to
  aerial imagery (they're photographs); manifest/world-file images default
  to OSM. Use manual mode if your whole batch should use one basemap.
- **Vision AI cost**: the estimate shown in the dialog is a worst case
  assuming every image falls through to the API. Images resolved from
  metadata cost nothing.
- **Failures**: failed images are listed in the completion summary with
  the reason. Common causes: no location source found (add the image to
  the manifest or enable the vision fallback), or insufficient matches
  (landscape changed, wrong footprint estimate, or featureless terrain).

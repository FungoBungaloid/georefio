"""
Location hint providers for bulk georeferencing.

Large image surveys are commonly distributed as a library of roughly
georeferenced images plus some record of where each frame was captured
(a flight path, capture log, or embedded metadata). These providers turn
that existing information into per-image location hints (bounding box
estimates), which feed directly into the same match-and-refine pipeline
used by the interactive georeferencer.

Provider chain (cheapest and most reliable first):

1. ManifestHintProvider   - explicit capture-path / manifest file supplied
                            by the user (CSV, GeoJSON, or GPX)
2. SidecarGeorefProvider  - existing rough georeferencing: world files
                            (.jgw/.tfw/.wld/...) or embedded GeoTIFF tags
3. ExifGpsHintProvider    - GPS coordinates embedded in image EXIF
4. VisionAPIHintProvider  - AI location estimation (costs money; fallback)

Because survey formats vary widely, ManifestHintProvider is deliberately
liberal about column/property names. See docs/BULK_GEOREFERENCING.md for the
accepted formats and known caveats.
"""

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logging_utils import get_logger
from .vision_api import BoundingBoxEstimate, DEFAULT_BASEMAP

logger = get_logger('location_providers')

# Meters per degree of latitude (approximately constant)
METERS_PER_DEGREE_LAT = 111320.0

# Recognized manifest column/property aliases (lowercase)
FILENAME_KEYS = ('filename', 'file', 'image', 'image_name', 'name', 'photo', 'frame')
LAT_KEYS = ('lat', 'latitude', 'y', 'gps_lat', 'center_lat')
LON_KEYS = ('lon', 'lng', 'long', 'longitude', 'x', 'gps_lon', 'center_lon')
EXTENT_KEYS = ('extent_m', 'extent', 'footprint_m', 'width_m', 'coverage_m')
TIME_KEYS = ('time', 'timestamp', 'datetime', 'date_time', 'capture_time')

# World file extensions, both format-specific and generic
WORLD_FILE_SUFFIXES = ('.jgw', '.jpgw', '.pgw', '.pngw', '.tfw', '.tifw',
                       '.gfw', '.bpw', '.wld')


def point_to_bbox(
    lat: float,
    lon: float,
    extent_meters: float,
    reasoning: str,
    source: str,
    recommended_basemap: str = DEFAULT_BASEMAP,
) -> BoundingBoxEstimate:
    """Build a bounding box centered on a point with the given extent."""
    half_lat = (extent_meters / 2) / METERS_PER_DEGREE_LAT
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    half_lon = (extent_meters / 2) / (METERS_PER_DEGREE_LAT * cos_lat)

    return BoundingBoxEstimate(
        min_lon=max(-180.0, lon - half_lon),
        min_lat=max(-90.0, lat - half_lat),
        max_lon=min(180.0, lon + half_lon),
        max_lat=min(90.0, lat + half_lat),
        reasoning=reasoning,
        recommended_basemap=recommended_basemap,
        source=source,
    )


class LocationHintProvider(ABC):
    """A source of per-image location hints."""

    #: Short identifier used in logs and results
    name: str = 'unknown'

    @abstractmethod
    def get_hint(self, image_path: Path) -> Optional[BoundingBoxEstimate]:
        """Return a location hint for the image, or None if unavailable.

        Providers should return None (not raise) when they simply have no
        information for an image; exceptions are reserved for hard errors
        (e.g. an unreadable manifest supplied by the user).
        """


class ManifestHintProvider(LocationHintProvider):
    """Location hints from a user-supplied capture-path / manifest file.

    Supported formats (detected by extension, falling back to sniffing):

    - CSV: one row per image with filename + lat/lon columns and an optional
      per-image extent column. Column names are matched case-insensitively
      against common aliases (see FILENAME_KEYS etc.).
    - GeoJSON: FeatureCollection of Point features with a filename-like
      property; optional 'extent_m' property.
    - GPX: waypoints whose <name> matches the image filename (with or
      without extension). Track points are also loaded so images with EXIF
      timestamps can be interpolated along the recorded flight path.
    """

    name = 'manifest'

    def __init__(self, manifest_path: Path, default_extent_m: float = 1000.0):
        self.manifest_path = Path(manifest_path)
        self.default_extent_m = default_extent_m

        # filename (lowercase, no extension) -> (lat, lon, extent or None)
        self._by_name: Dict[str, Tuple[float, float, Optional[float]]] = {}
        # Sorted (timestamp, lat, lon) track points for interpolation
        self._track: List[Tuple[float, float, float]] = []

        self._load()

    # -- loading -----------------------------------------------------------

    def _load(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        suffix = self.manifest_path.suffix.lower()
        text = self.manifest_path.read_text(encoding='utf-8-sig', errors='replace')

        if suffix == '.gpx' or text.lstrip().startswith('<'):
            self._load_gpx(text)
        elif suffix in ('.geojson', '.json') or text.lstrip().startswith('{'):
            self._load_geojson(text)
        else:
            self._load_csv(text)

        logger.info(
            f"Manifest '{self.manifest_path.name}': {len(self._by_name)} image "
            f"records, {len(self._track)} track points"
        )

        if not self._by_name and not self._track:
            raise ValueError(
                f"No usable records found in manifest {self.manifest_path}. "
                "See docs/BULK_GEOREFERENCING.md for supported formats."
            )

    @staticmethod
    def _pick(record: Dict[str, str], keys) -> Optional[str]:
        for key in keys:
            if key in record and record[key] not in (None, ''):
                return record[key]
        return None

    @staticmethod
    def _normalize_name(filename: str) -> str:
        return Path(str(filename).strip()).stem.lower()

    def _add_record(self, filename, lat, lon, extent):
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            return
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return
        extent_val = None
        if extent not in (None, ''):
            try:
                extent_val = float(extent)
            except (TypeError, ValueError):
                extent_val = None
        self._by_name[self._normalize_name(filename)] = (lat, lon, extent_val)

    def _load_csv(self, text: str):
        # Sniff the delimiter (comma, semicolon, or tab)
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=',;\t')
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(text.splitlines(), dialect=dialect)
        if not reader.fieldnames:
            return

        for row in reader:
            record = {
                (k or '').strip().lower(): (v or '').strip()
                for k, v in row.items()
            }
            filename = self._pick(record, FILENAME_KEYS)
            lat = self._pick(record, LAT_KEYS)
            lon = self._pick(record, LON_KEYS)
            if filename and lat and lon:
                self._add_record(filename, lat, lon, self._pick(record, EXTENT_KEYS))

    def _load_geojson(self, text: str):
        data = json.loads(text)
        features = data.get('features', [data] if data.get('type') == 'Feature' else [])

        for feature in features:
            geometry = feature.get('geometry') or {}
            if geometry.get('type') != 'Point':
                continue
            coords = geometry.get('coordinates') or []
            if len(coords) < 2:
                continue
            lon, lat = coords[0], coords[1]

            props = {
                str(k).strip().lower(): v
                for k, v in (feature.get('properties') or {}).items()
            }
            filename = self._pick(props, FILENAME_KEYS)
            if filename:
                self._add_record(filename, lat, lon, self._pick(props, EXTENT_KEYS))

    def _load_gpx(self, text: str):
        root = ET.fromstring(text)
        # Handle GPX namespaces generically
        ns = {'gpx': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

        def findall(parent, tag):
            return parent.findall(f'gpx:{tag}', ns) if ns else parent.findall(tag)

        def find_text(parent, tag):
            el = parent.find(f'gpx:{tag}', ns) if ns else parent.find(tag)
            return el.text if el is not None else None

        # Waypoints keyed by name
        for wpt in findall(root, 'wpt'):
            name = find_text(wpt, 'name')
            if name:
                self._add_record(name, wpt.get('lat'), wpt.get('lon'), None)

        # Track points with timestamps for interpolation
        for trk in findall(root, 'trk'):
            for seg in findall(trk, 'trkseg'):
                for pt in findall(seg, 'trkpt'):
                    time_text = find_text(pt, 'time')
                    if not time_text:
                        continue
                    timestamp = _parse_iso_time(time_text)
                    if timestamp is None:
                        continue
                    try:
                        self._track.append(
                            (timestamp, float(pt.get('lat')), float(pt.get('lon')))
                        )
                    except (TypeError, ValueError):
                        continue

        self._track.sort(key=lambda p: p[0])

    # -- lookup ------------------------------------------------------------

    def get_hint(self, image_path: Path) -> Optional[BoundingBoxEstimate]:
        key = self._normalize_name(image_path.name)
        record = self._by_name.get(key)

        if record is not None:
            lat, lon, extent = record
            return point_to_bbox(
                lat, lon,
                extent or self.default_extent_m,
                reasoning=f"Capture position from manifest {self.manifest_path.name}",
                source=self.name,
            )

        # No direct record; try interpolating along the GPX track using the
        # image's EXIF capture timestamp
        if self._track:
            timestamp = _read_exif_timestamp(image_path)
            if timestamp is not None:
                position = self._interpolate_track(timestamp)
                if position is not None:
                    lat, lon = position
                    return point_to_bbox(
                        lat, lon, self.default_extent_m,
                        reasoning=(
                            f"Interpolated along GPX track from "
                            f"{self.manifest_path.name} using EXIF timestamp"
                        ),
                        source=f'{self.name}_track',
                    )

        return None

    def _interpolate_track(self, timestamp: float) -> Optional[Tuple[float, float]]:
        """Linearly interpolate a position along the track for a timestamp."""
        track = self._track
        if not track:
            return None

        first_t, last_t = track[0][0], track[-1][0]

        # Reject timestamps far outside the recorded track (> 10 minutes);
        # clock offsets between camera and GPS logger are common, so a small
        # margin is tolerated but wild mismatches are not.
        margin = 600.0
        if timestamp < first_t - margin or timestamp > last_t + margin:
            logger.debug(
                f"EXIF timestamp {timestamp} outside GPX track range "
                f"[{first_t}, {last_t}] (+/-{margin}s)"
            )
            return None

        if timestamp <= first_t:
            return track[0][1], track[0][2]
        if timestamp >= last_t:
            return track[-1][1], track[-1][2]

        # Binary search for the surrounding pair
        lo, hi = 0, len(track) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if track[mid][0] <= timestamp:
                lo = mid
            else:
                hi = mid

        t0, lat0, lon0 = track[lo]
        t1, lat1, lon1 = track[hi]
        if t1 == t0:
            return lat0, lon0

        fraction = (timestamp - t0) / (t1 - t0)
        return (
            lat0 + (lat1 - lat0) * fraction,
            lon0 + (lon1 - lon0) * fraction,
        )


class SidecarGeorefProvider(LocationHintProvider):
    """Location hints from existing rough georeferencing.

    Checks, in order:
    1. Embedded geotransform (e.g. GeoTIFF tags), via GDAL when available
    2. Sidecar world files (.jgw, .tfw, .wld, ...)

    World files carry no CRS, so a CRS must be assumed for them (a .prj
    sidecar is honored when present). The computed footprint is expanded
    slightly since the source is only *roughly* georeferenced.
    """

    name = 'sidecar_georef'

    # Expand the derived footprint to tolerate rough georeferencing
    FOOTPRINT_MARGIN = 1.3

    def __init__(self, world_file_crs: str = 'EPSG:4326'):
        self.world_file_crs = world_file_crs

    def get_hint(self, image_path: Path) -> Optional[BoundingBoxEstimate]:
        hint = self._from_embedded_georef(image_path)
        if hint is not None:
            return hint
        return self._from_world_file(image_path)

    def _from_embedded_georef(self, image_path: Path) -> Optional[BoundingBoxEstimate]:
        """Read an embedded geotransform (GeoTIFF and friends) via GDAL."""
        try:
            from osgeo import gdal, osr
        except ImportError:
            return None

        try:
            gdal.UseExceptions()
            ds = gdal.Open(str(image_path))
            if ds is None:
                return None

            gt = ds.GetGeoTransform(can_return_null=True)
            if gt is None:
                return None

            width, height = ds.RasterXSize, ds.RasterYSize
            projection = ds.GetProjection()
            ds = None

            corners = _geotransform_corners(gt, width, height)
            if projection:
                corners = _project_to_wgs84_gdal(corners, projection)
                crs_label = 'embedded CRS'
            elif self.world_file_crs.upper() not in ('EPSG:4326', 'WGS84'):
                corners = _project_to_wgs84_gdal(corners, self.world_file_crs)
                crs_label = self.world_file_crs
            else:
                crs_label = 'assumed EPSG:4326'

            if corners is None:
                return None

            return self._corners_to_hint(
                corners,
                reasoning=f"Embedded georeferencing in {image_path.name} ({crs_label})"
            )

        except Exception as e:
            logger.debug(f"Embedded georef read failed for {image_path.name}: {e}")
            return None

    def _from_world_file(self, image_path: Path) -> Optional[BoundingBoxEstimate]:
        world_path = self._find_world_file(image_path)
        if world_path is None:
            return None

        try:
            values = [
                float(line.strip())
                for line in world_path.read_text().splitlines()
                if line.strip()
            ][:6]
        except (ValueError, OSError) as e:
            logger.warning(f"Could not parse world file {world_path.name}: {e}")
            return None

        if len(values) != 6:
            logger.warning(f"World file {world_path.name} has {len(values)} values, expected 6")
            return None

        # World file order: A (x scale), D, B, E (y scale), C (x origin), F (y origin)
        a, d, b, e, c, f = values
        gt = (c, a, b, f, d, e)  # GDAL geotransform order

        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as ex:
            logger.warning(f"Could not read image size for {image_path.name}: {ex}")
            return None

        corners = _geotransform_corners(gt, width, height)

        # Honor a .prj sidecar; otherwise use the configured assumption
        crs = self.world_file_crs
        prj_path = image_path.with_suffix('.prj')
        if prj_path.exists():
            try:
                crs = prj_path.read_text().strip()
            except OSError:
                pass

        if crs.upper() not in ('EPSG:4326', 'WGS84'):
            corners = _project_to_wgs84_gdal(corners, crs)
            if corners is None:
                logger.warning(
                    f"Could not reproject world file coordinates for "
                    f"{image_path.name} (CRS: {crs[:60]})"
                )
                return None

        # Sanity check: coordinates should look like lon/lat now
        if not all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in corners):
            logger.warning(
                f"World file for {image_path.name} produced non-geographic "
                f"coordinates; set the correct CRS for world files "
                f"(currently {self.world_file_crs})"
            )
            return None

        return self._corners_to_hint(
            corners,
            reasoning=f"World file {world_path.name} ({crs if len(crs) < 40 else 'projected CRS'})"
        )

    @staticmethod
    def _find_world_file(image_path: Path) -> Optional[Path]:
        # Try format-specific extensions first, then generic .wld, and the
        # "append w to original extension" convention (.jpg -> .jpgw)
        candidates = [image_path.with_suffix(s) for s in WORLD_FILE_SUFFIXES]
        candidates.append(image_path.with_suffix(image_path.suffix + 'w'))

        for candidate in candidates:
            if candidate.exists():
                return candidate
            # Case-insensitive check for case-sensitive filesystems
            if candidate.parent.exists():
                lower_name = candidate.name.lower()
                for sibling in candidate.parent.iterdir():
                    if sibling.name.lower() == lower_name:
                        return sibling
        return None

    def _corners_to_hint(self, corners, reasoning: str) -> BoundingBoxEstimate:
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]

        center_lon = (min(lons) + max(lons)) / 2
        center_lat = (min(lats) + max(lats)) / 2
        half_lon = (max(lons) - min(lons)) / 2 * self.FOOTPRINT_MARGIN
        half_lat = (max(lats) - min(lats)) / 2 * self.FOOTPRINT_MARGIN

        return BoundingBoxEstimate(
            min_lon=max(-180.0, center_lon - half_lon),
            min_lat=max(-90.0, center_lat - half_lat),
            max_lon=min(180.0, center_lon + half_lon),
            max_lat=min(90.0, center_lat + half_lat),
            reasoning=reasoning,
            recommended_basemap=DEFAULT_BASEMAP,
            source=self.name,
        )


class ExifGpsHintProvider(LocationHintProvider):
    """Location hints from GPS coordinates embedded in image EXIF."""

    name = 'exif_gps'

    def __init__(self, default_extent_m: float = 1000.0):
        self.default_extent_m = default_extent_m

    def get_hint(self, image_path: Path) -> Optional[BoundingBoxEstimate]:
        position = _read_exif_gps(image_path)
        if position is None:
            return None

        lat, lon = position
        return point_to_bbox(
            lat, lon, self.default_extent_m,
            reasoning=f"EXIF GPS position in {image_path.name}",
            source=self.name,
            # EXIF GPS almost always means a photograph, so prefer imagery
            recommended_basemap='esri_world_imagery',
        )


class VisionAPIHintProvider(LocationHintProvider):
    """Location hints from a vision AI model (costs money per image)."""

    name = 'vision_api'

    def __init__(self, vision_client, progress_callback=None):
        """
        Args:
            vision_client: VisionAPIClient instance
            progress_callback: Optional callback(message) for progress updates
        """
        self.vision_client = vision_client
        self.progress_callback = progress_callback

    def get_hint(self, image_path: Path) -> Optional[BoundingBoxEstimate]:
        # API errors propagate: they usually mean a config problem (bad key,
        # rate limit) the user needs to hear about, not "no data available".
        return self.vision_client.estimate_bounding_box(
            image_path,
            progress_callback=self.progress_callback
        )


class CompositeHintProvider(LocationHintProvider):
    """Try a chain of providers in order and return the first hint."""

    name = 'composite'

    def __init__(self, providers: List[LocationHintProvider]):
        self.providers = providers

    def get_hint(self, image_path: Path) -> Optional[BoundingBoxEstimate]:
        for provider in self.providers:
            try:
                hint = provider.get_hint(image_path)
            except Exception:
                # Hard errors from the last provider in the chain (usually
                # the vision API) should surface to the caller
                if provider is self.providers[-1]:
                    raise
                logger.warning(
                    f"Location provider '{provider.name}' failed for "
                    f"{image_path.name}; trying next provider"
                )
                continue

            if hint is not None:
                is_valid, message = hint.is_valid()
                if is_valid:
                    logger.info(
                        f"{image_path.name}: location from '{provider.name}' "
                        f"({hint.center_lat:.5f}, {hint.center_lon:.5f})"
                    )
                    return hint
                logger.warning(
                    f"{image_path.name}: '{provider.name}' returned invalid "
                    f"bounding box ({message}); trying next provider"
                )

        return None


def build_hint_chain(
    use_metadata_hints: bool = True,
    manifest_path: Optional[Path] = None,
    default_extent_m: float = 1000.0,
    world_file_crs: str = 'EPSG:4326',
    vision_client=None,
    vision_progress_callback=None,
) -> CompositeHintProvider:
    """Assemble the standard provider chain for batch processing.

    Order: manifest -> sidecar georeferencing -> EXIF GPS -> vision API.

    Raises:
        ValueError: if the resulting chain would be empty
    """
    providers: List[LocationHintProvider] = []

    if manifest_path:
        providers.append(ManifestHintProvider(manifest_path, default_extent_m))

    if use_metadata_hints:
        providers.append(SidecarGeorefProvider(world_file_crs))
        providers.append(ExifGpsHintProvider(default_extent_m))

    if vision_client is not None:
        providers.append(VisionAPIHintProvider(vision_client, vision_progress_callback))

    if not providers:
        raise ValueError(
            "No location hint sources configured. Enable metadata hints, "
            "supply a manifest file, or configure a vision API."
        )

    return CompositeHintProvider(providers)


# -- helpers ---------------------------------------------------------------

def _geotransform_corners(gt, width: int, height: int):
    """Compute the four corner coordinates from a GDAL geotransform."""
    def apply(px, py):
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        return x, y

    return [apply(0, 0), apply(width, 0), apply(0, height), apply(width, height)]


def _project_to_wgs84_gdal(corners, crs_definition: str):
    """Reproject (x, y) corner coordinates to WGS84 lon/lat using GDAL.

    Returns None if GDAL is unavailable or the CRS can't be parsed.
    """
    try:
        from osgeo import osr
    except ImportError:
        return None

    try:
        source = osr.SpatialReference()
        if source.SetFromUserInput(crs_definition) != 0:
            return None

        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)

        # Force x=lon, y=lat ordering regardless of CRS axis definitions
        if hasattr(osr, 'OAMS_TRADITIONAL_GIS_ORDER'):
            source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        transform = osr.CoordinateTransformation(source, target)
        return [transform.TransformPoint(x, y)[:2] for x, y in corners]
    except Exception as e:
        logger.debug(f"CRS reprojection failed: {e}")
        return None


def _parse_iso_time(text: str) -> Optional[float]:
    """Parse an ISO-8601 timestamp to a UTC epoch value."""
    text = text.strip()
    try:
        # fromisoformat in older Pythons doesn't accept the 'Z' suffix
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def _read_exif_gps(image_path: Path) -> Optional[Tuple[float, float]]:
    """Read (lat, lon) from image EXIF GPS tags, or None."""
    try:
        from PIL import Image
    except ImportError:
        return None

    GPS_IFD = 0x8825
    # GPS IFD tag ids
    LAT_REF, LAT, LON_REF, LON = 1, 2, 3, 4

    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None
            gps = exif.get_ifd(GPS_IFD)
            if not gps or LAT not in gps or LON not in gps:
                return None

            lat = _dms_to_degrees(gps[LAT])
            lon = _dms_to_degrees(gps[LON])
            if lat is None or lon is None:
                return None

            if str(gps.get(LAT_REF, 'N')).upper().startswith('S'):
                lat = -lat
            if str(gps.get(LON_REF, 'E')).upper().startswith('W'):
                lon = -lon

            if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat, lon) != (0.0, 0.0):
                return lat, lon
            return None

    except Exception as e:
        logger.debug(f"EXIF GPS read failed for {image_path.name}: {e}")
        return None


def _dms_to_degrees(value) -> Optional[float]:
    """Convert EXIF degrees/minutes/seconds rationals to decimal degrees."""
    try:
        parts = [float(v) for v in value]
        while len(parts) < 3:
            parts.append(0.0)
        return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _read_exif_timestamp(image_path: Path) -> Optional[float]:
    """Read the EXIF capture timestamp as a UTC epoch value, or None.

    EXIF DateTimeOriginal has no timezone; it is interpreted as UTC unless an
    OffsetTimeOriginal tag is present. When matching against a GPX track the
    small tolerance in the interpolator absorbs modest clock offsets.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import IFD
    except ImportError:
        return None

    DATETIME_ORIGINAL = 36867
    OFFSET_TIME_ORIGINAL = 36881
    DATETIME = 306

    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None

            try:
                exif_ifd = exif.get_ifd(IFD.Exif)
            except Exception:
                exif_ifd = {}

            raw = exif_ifd.get(DATETIME_ORIGINAL) or exif.get(DATETIME)
            if not raw:
                return None

            # EXIF format: "YYYY:MM:DD HH:MM:SS"
            match = re.match(r'(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})', str(raw))
            if not match:
                return None

            parsed = datetime(*(int(g) for g in match.groups()), tzinfo=timezone.utc)

            offset = exif_ifd.get(OFFSET_TIME_ORIGINAL)
            if offset:
                offset_match = re.match(r'([+-])(\d{2}):(\d{2})', str(offset))
                if offset_match:
                    sign = -1 if offset_match.group(1) == '+' else 1
                    seconds = int(offset_match.group(2)) * 3600 + int(offset_match.group(3)) * 60
                    return parsed.timestamp() + sign * seconds

            return parsed.timestamp()

    except Exception as e:
        logger.debug(f"EXIF timestamp read failed for {image_path.name}: {e}")
        return None

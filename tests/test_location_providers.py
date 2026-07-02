"""
Tests for magic_georeferencer.core.location_providers.

These cover the pure-Python provider paths (manifest parsing, EXIF GPS,
world files in EPSG:4326, composite chaining) and run without QGIS, torch,
or GDAL. Requires: pillow.

Run from the repository root:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from magic_georeferencer.core.location_providers import (  # noqa: E402
    ManifestHintProvider,
    SidecarGeorefProvider,
    ExifGpsHintProvider,
    build_hint_chain,
    point_to_bbox,
)


@pytest.fixture
def gps_image(tmp_path):
    """JPEG with EXIF GPS (51.5, -0.125) and a capture timestamp."""
    path = tmp_path / 'IMG_0042.jpg'
    img = Image.new('RGB', (100, 80), (10, 20, 30))
    exif = Image.Exif()
    exif[0x8825] = {  # GPS IFD
        1: 'N', 2: (51.0, 30.0, 0.0),   # 51 deg 30' N
        3: 'W', 4: (0.0, 7.0, 30.0),    # 0 deg 7' 30" W
    }
    exif[0x8769] = {36867: '2024:06:01 12:00:30'}  # DateTimeOriginal
    img.save(path, exif=exif)
    return path


@pytest.fixture
def plain_image(tmp_path):
    path = tmp_path / 'plain.jpg'
    Image.new('RGB', (100, 80), (120, 130, 140)).save(path)
    return path


class TestPointToBbox:
    def test_roundtrip(self):
        bbox = point_to_bbox(51.5, -0.125, 1000, 'test', 'test')
        assert abs(bbox.center_lat - 51.5) < 1e-9
        assert abs(bbox.center_lon - -0.125) < 1e-9
        assert abs(bbox.to_extent_meters() - 1000) < 20
        assert bbox.is_valid()[0]


class TestCsvManifest:
    def test_basic_and_default_extent(self, tmp_path):
        manifest = tmp_path / 'manifest.csv'
        manifest.write_text(
            "Filename,Latitude,Longitude,extent_m\n"
            "IMG_0001.jpg,48.8584,2.2945,500\n"
            "IMG_0002.jpg,48.8600,2.2900,\n"
        )
        p = ManifestHintProvider(manifest, default_extent_m=1200)

        hint = p.get_hint(Path('IMG_0001.jpg'))
        assert hint is not None
        assert abs(hint.center_lat - 48.8584) < 1e-6
        assert abs(hint.to_extent_meters() - 500) < 10

        # Case-insensitive, extension-independent lookup; default extent
        hint2 = p.get_hint(Path('img_0002.TIF'))
        assert hint2 is not None
        assert abs(hint2.to_extent_meters() - 1200) < 25

        assert p.get_hint(Path('unknown.jpg')) is None

    def test_semicolon_delimiter(self, tmp_path):
        manifest = tmp_path / 'manifest.csv'
        manifest.write_text("file;lat;lon\nphoto_a.tif;-12.05;-77.04\n")
        p = ManifestHintProvider(manifest)
        hint = p.get_hint(Path('photo_a.tif'))
        assert hint is not None
        assert abs(hint.center_lon - -77.04) < 1e-6

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ManifestHintProvider(tmp_path / 'missing.csv')

    def test_empty_manifest_raises(self, tmp_path):
        manifest = tmp_path / 'empty.csv'
        manifest.write_text("a,b,c\n1,2,3\n")  # no recognizable columns
        with pytest.raises(ValueError):
            ManifestHintProvider(manifest)


class TestGeojsonManifest:
    def test_point_features(self, tmp_path):
        manifest = tmp_path / 'manifest.geojson'
        manifest.write_text('''{
          "type": "FeatureCollection",
          "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [-70.66, -33.45]},
             "properties": {"image": "IMG_0003.jpg", "extent_m": 800}}
          ]
        }''')
        p = ManifestHintProvider(manifest)
        hint = p.get_hint(Path('IMG_0003.jpg'))
        assert hint is not None
        assert abs(hint.center_lat - -33.45) < 1e-6
        assert abs(hint.to_extent_meters() - 800) < 15


class TestGpxManifest:
    GPX = '''<?xml version="1.0"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="40.7128" lon="-74.0060"><name>IMG_0004</name></wpt>
  <trk><trkseg>
    <trkpt lat="40.0" lon="-75.0"><time>2024-06-01T12:00:00Z</time></trkpt>
    <trkpt lat="41.0" lon="-74.0"><time>2024-06-01T12:01:00Z</time></trkpt>
  </trkseg></trk>
</gpx>'''

    def test_waypoint_by_name(self, tmp_path):
        manifest = tmp_path / 'flight.gpx'
        manifest.write_text(self.GPX)
        p = ManifestHintProvider(manifest)
        hint = p.get_hint(Path('IMG_0004.jpg'))
        assert hint is not None
        assert abs(hint.center_lat - 40.7128) < 1e-6

    def test_track_interpolation_from_exif_timestamp(self, tmp_path):
        manifest = tmp_path / 'flight.gpx'
        manifest.write_text(self.GPX)

        # Image timestamped halfway along the 60s track
        image = tmp_path / 'frame_100.jpg'
        img = Image.new('RGB', (100, 80), (5, 5, 5))
        exif = Image.Exif()
        exif[0x8769] = {36867: '2024:06:01 12:00:30'}
        img.save(image, exif=exif)

        p = ManifestHintProvider(manifest)
        hint = p.get_hint(image)
        assert hint is not None
        assert abs(hint.center_lat - 40.5) < 1e-6
        assert abs(hint.center_lon - -74.5) < 1e-6
        assert hint.source == 'manifest_track'


class TestExifGps:
    def test_decode(self, gps_image):
        p = ExifGpsHintProvider(default_extent_m=900)
        hint = p.get_hint(gps_image)
        assert hint is not None
        assert abs(hint.center_lat - 51.5) < 1e-6
        assert abs(hint.center_lon - -0.125) < 1e-6
        assert hint.recommended_basemap == 'esri_world_imagery'

    def test_no_gps(self, plain_image):
        assert ExifGpsHintProvider().get_hint(plain_image) is None


class TestWorldFile:
    def test_wgs84_world_file(self, tmp_path):
        image = tmp_path / 'ortho.png'
        Image.new('RGB', (200, 100), (50, 60, 70)).save(image)
        # 0.0001 deg/px, origin lon=10 lat=45
        (tmp_path / 'ortho.pngw').write_text("0.0001\n0\n0\n-0.0001\n10.0\n45.0\n")

        p = SidecarGeorefProvider(world_file_crs='EPSG:4326')
        hint = p.get_hint(image)
        assert hint is not None
        # 200x100 px -> 0.02 x 0.01 deg footprint centered at (10.01, 44.995)
        assert abs(hint.center_lon - 10.01) < 1e-6
        assert abs(hint.center_lat - 44.995) < 1e-6
        assert hint.width_degrees > 0.02  # safety margin applied

    def test_no_world_file(self, plain_image):
        assert SidecarGeorefProvider().get_hint(plain_image) is None


class TestCompositeChain:
    def test_priority_and_fallthrough(self, tmp_path, gps_image):
        manifest = tmp_path / 'manifest.csv'
        manifest.write_text(
            "filename,lat,lon\nIMG_0001.jpg,48.8584,2.2945\n"
        )
        chain = build_hint_chain(
            use_metadata_hints=True,
            manifest_path=manifest,
            default_extent_m=1000,
        )

        # Manifest entry wins
        listed = tmp_path / 'IMG_0001.jpg'
        Image.new('RGB', (10, 10)).save(listed)
        hint = chain.get_hint(listed)
        assert hint is not None and hint.source == 'manifest'

        # No manifest entry -> falls through to EXIF GPS
        hint = chain.get_hint(gps_image)
        assert hint is not None and hint.source == 'exif_gps'

    def test_nothing_found(self, plain_image):
        chain = build_hint_chain(use_metadata_hints=True)
        assert chain.get_hint(plain_image) is None

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError):
            build_hint_chain(use_metadata_hints=False)

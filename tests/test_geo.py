"""Unit tests for pure-Python ESRI world-file geospatial transforms."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_zoom.geo import GeoBox, GeoTransform
from vlm_zoom.tiler import PixelBox


def test_geo_transform_from_world_file_and_roundtrip(tmp_path: Path):
    wld = tmp_path / "geology.pgw"
    # North-up 2.5 m/pixel UTM map
    wld.write_text("2.5\n0.0\n0.0\n-2.5\n500000.0\n3500000.0\n", encoding="utf-8")

    tf = GeoTransform.from_world_file(wld, crs="EPSG:32650")
    assert tf.crs == "EPSG:32650"

    mx, my = tf.pixel_to_map(100.0, 200.0)
    assert mx == pytest.approx(500250.0)
    assert my == pytest.approx(3499500.0)

    px, py = tf.map_to_pixel(mx, my)
    assert px == pytest.approx(100.0)
    assert py == pytest.approx(200.0)

    gbox = tf.pixel_box_to_geo(PixelBox(100.0, 200.0, 300.0, 400.0))
    assert gbox == GeoBox(
        min_x=500250.0,
        min_y=3499000.0,
        max_x=500750.0,
        max_y=3499500.0,
        crs="EPSG:32650",
    )


def test_geo_transform_rotated_affine_roundtrip():
    tf = GeoTransform(
        pixel_width=1.5,
        col_rotation=0.25,
        row_rotation=-0.35,
        pixel_height=-1.5,
        origin_x=118.75,
        origin_y=32.05,
        crs="EPSG:4326",
    )
    mx, my = tf.pixel_to_map(420.5, 198.25)
    px, py = tf.map_to_pixel(mx, my)
    assert px == pytest.approx(420.5)
    assert py == pytest.approx(198.25)


def test_geo_transform_invalid_world_file_and_singular_matrix(tmp_path: Path):
    bad = tmp_path / "short.wld"
    bad.write_text("1.0\n0.0\n0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="6 numeric lines"):
        GeoTransform.from_world_file(bad)

    singular = GeoTransform(1.0, 2.0, 1.0, 2.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="Singular"):
        singular.map_to_pixel(10.0, 10.0)

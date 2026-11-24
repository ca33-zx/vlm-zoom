"""Unit tests for AdaptiveTiler, coordinate mapping, and PinnedContext."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vlm_zoom.tiler import (
    AdaptiveTiler,
    PixelBox,
    TileSpec,
    build_pinned_context,
    load_image,
    normalize_scientific_image,
    read_image_size,
)


def test_pixel_box_clamp_and_pad():
    box = PixelBox(10.0, 20.0, 110.0, 70.0)
    assert box.width == 100.0
    assert box.height == 50.0
    assert box.area == 5000.0

    padded = box.pad(0.2, max_w=120, max_h=100)
    assert padded.x1 == 0.0
    assert padded.y1 == 10.0
    assert padded.x2 == 120.0
    assert padded.y2 == 80.0


def test_tiler_validation():
    with pytest.raises(ValueError, match="overlap"):
        AdaptiveTiler(overlap=0.5)
    with pytest.raises(ValueError, match="max_tokens_per_tile"):
        AdaptiveTiler(max_tokens_per_tile=0)
    with pytest.raises(ValueError, match="min_detail_scale"):
        AdaptiveTiler(min_detail_scale=1.2)
    with pytest.raises(ValueError, match="max_tiles"):
        AdaptiveTiler(max_tiles=0)


def test_tiler_plan_full_coverage_and_overlap():
    tiler = AdaptiveTiler(
        "qwen3_vl",
        max_tokens_per_tile=1024,
        overlap=0.2,
        min_detail_scale=0.85,
    )
    plan = tiler.plan(4800, 3000)

    assert plan.rows >= 2
    assert plan.cols >= 2
    assert len(plan.tiles) == plan.rows * plan.cols
    assert plan.min_effective_scale >= 0.75

    # Every tile stays strictly within its per-tile visual token cap.
    for tile in plan.tiles:
        assert tile.estimated_tokens <= 1024
        assert tile.box.x1 >= 0.0
        assert tile.box.y1 >= 0.0
        assert tile.box.x2 <= 4800.0
        assert tile.box.y2 <= 3000.0

    # First tile starts at (0, 0) and last tile reaches (4800, 3000).
    assert plan.tiles[0].box.x1 == 0.0
    assert plan.tiles[0].box.y1 == 0.0
    assert plan.tiles[-1].box.x2 == 4800.0
    assert plan.tiles[-1].box.y2 == 3000.0

    # Adjacent horizontal tiles must actually overlap (x2 of col c > x1 of col c+1).
    t00 = plan.tiles[0]
    t01 = plan.tiles[1]
    assert t00.row == t01.row == 0
    assert t00.box.x2 > t01.box.x1
    overlap_ratio = (t00.box.x2 - t01.box.x1) / t00.box.width
    assert abs(overlap_ratio - 0.2) < 1e-6


def test_tiler_respects_max_tiles_and_explicit_roi():
    tiler = AdaptiveTiler("qwen3_vl", max_tokens_per_tile=256, max_tiles=6)
    plan = tiler.plan(8000, 6000)
    assert len(plan.tiles) <= 6

    roi = PixelBox(1000.0, 500.0, 3000.0, 1500.0)
    roi_plan = tiler.plan(8000, 6000, roi=roi)
    assert roi_plan.tiles[0].box.x1 == 1000.0
    assert roi_plan.tiles[0].box.y1 == 500.0
    assert roi_plan.tiles[-1].box.x2 == 3000.0
    assert roi_plan.tiles[-1].box.y2 == 1500.0


def test_tile_materialization_and_header_only_size(tmp_path: Path):
    img_path = tmp_path / "map.png"
    src = Image.new("RGB", (2400, 1600), (200, 210, 220))
    src.save(img_path)

    assert read_image_size(img_path) == (2400, 1600)
    assert read_image_size(src) == (2400, 1600)

    tiler = AdaptiveTiler("qwen3_vl", max_tokens_per_tile=512, overlap=0.1)
    plan, tiles = tiler.tile(img_path)
    assert len(tiles) == len(plan.tiles)
    for tile in tiles:
        assert tile.image.width <= tile.spec.sent_width
        assert tile.image.height <= tile.spec.sent_height
        url = tile.to_data_url(fmt="JPEG", quality=85)
        assert url.startswith("data:image/jpeg;base64,")


def test_tile_spec_local_to_global_box_all_coord_spaces():
    spec = TileSpec(
        tile_id="r1_c2",
        row=1,
        col=2,
        box=PixelBox(1000.0, 2000.0, 2000.0, 2500.0),  # width=1000, height=500
        sent_width=500,
        sent_height=250,
        estimated_tokens=200,
    )

    g_norm1000 = spec.local_to_global_box((100.0, 200.0, 500.0, 800.0), coord_space="norm1000")
    assert g_norm1000 == PixelBox(1100.0, 2100.0, 1500.0, 2400.0)

    g_norm1 = spec.local_to_global_box((0.1, 0.2, 0.5, 0.8), coord_space="norm1")
    assert g_norm1 == PixelBox(1100.0, 2100.0, 1500.0, 2400.0)

    g_pixel = spec.local_to_global_box((50.0, 50.0, 250.0, 200.0), coord_space="pixel")
    assert g_pixel == PixelBox(1100.0, 2100.0, 1500.0, 2400.0)

    # Inverted coordinates are normalized cleanly.
    g_swapped = spec.local_to_global_box((500.0, 800.0, 100.0, 200.0), coord_space="norm1000")
    assert g_swapped == PixelBox(1100.0, 2100.0, 1500.0, 2400.0)

    with pytest.raises(ValueError, match="Unsupported coord_space"):
        spec.local_to_global_box((0, 0, 1, 1), coord_space="invalid")  # type: ignore[arg-type]


def test_build_pinned_context_crops_legend_losslessly():
    img = Image.new("RGB", (2000, 1200), (240, 240, 240))
    ctx = build_pinned_context(
        img,
        label="Geological Legend",
        box=(1600, 100, 1950, 1100),
        model_or_family="qwen3_vl",
        max_tokens=1024,
        fmt="PNG",
    )
    assert ctx.label == "Geological Legend"
    assert ctx.source_box == PixelBox(1600.0, 100.0, 1950.0, 1100.0)
    assert ctx.estimated_tokens <= 1024
    assert ctx.data_url.startswith("data:image/png;base64,")


def test_normalize_scientific_16bit_tiff_and_float_array(tmp_path: Path):
    # 16-bit single-band remote sensing TIFF (values in 1000..4000 range)
    arr16 = np.linspace(1000, 4000, 64 * 96, dtype=np.uint16).reshape(64, 96)
    tiff_path = tmp_path / "sentinel_band.tif"
    Image.fromarray(arr16).save(tiff_path)

    loaded = load_image(tiff_path)
    assert loaded.mode == "RGB"
    assert loaded.size == (96, 64)
    rgb = np.asarray(loaded)
    assert rgb.min() == 0
    assert rgb.max() == 255

    # Float32 multi-band array with NaNs (4 bands -> takes first 3)
    f_arr = np.linspace(-50.0, 250.0, 40 * 80 * 4, dtype=np.float32).reshape(40, 80, 4)
    f_arr[0, 0, :] = np.nan
    assert read_image_size(f_arr) == (80, 40)
    stretched = normalize_scientific_image(f_arr)
    assert stretched.mode == "RGB"
    assert stretched.size == (80, 40)
    assert np.asarray(stretched).max() == 255

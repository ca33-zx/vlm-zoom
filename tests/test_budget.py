"""Unit tests for exact visual token budgeting and smart_resize geometry."""

from __future__ import annotations

import pytest

from vlm_zoom.budget import (
    MODEL_SPECS,
    VisionModelSpec,
    estimate_visual_tokens,
    resolve_model_spec,
    smart_resize,
)


def test_smart_resize_divisibility_and_bounds():
    w, h = smart_resize(4800, 3000, factor=32, min_pixels=4 * 32 * 32, max_pixels=1024 * 32 * 32)
    assert w % 32 == 0
    assert h % 32 == 0
    assert 4 * 32 * 32 <= w * h <= 1024 * 32 * 32
    # Aspect ratio 4800 / 3000 = 1.6 should be closely preserved.
    assert abs((w / h) - (4800 / 3000)) < 0.1


def test_smart_resize_upscales_tiny_images_to_min_pixels():
    min_px = 16 * 28 * 28
    w, h = smart_resize(20, 20, factor=28, min_pixels=min_px, max_pixels=1024 * 28 * 28)
    assert w % 28 == 0
    assert h % 28 == 0
    assert w * h >= min_px


def test_smart_resize_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        smart_resize(0, 100)
    with pytest.raises(ValueError, match="positive"):
        smart_resize(100, -5)
    with pytest.raises(ValueError, match="max_pixels"):
        smart_resize(500, 500, min_pixels=2000, max_pixels=1000)
    with pytest.raises(ValueError, match="aspect ratio"):
        smart_resize(50000, 100, max_aspect_ratio=200.0)


@pytest.mark.parametrize(
    ("model_name", "expected_family", "expected_factor"),
    [
        ("Qwen/Qwen2-VL-7B-Instruct", "qwen2_vl", 28),
        ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen2_5_vl", 28),
        ("Qwen/Qwen3-VL-4B-Instruct", "qwen3_vl", 32),
        ("Qwen/Qwen3.5-9B", "qwen3_5", 32),
        ("OpenGVLab/InternVL3-8B", "internvl", 28),
        ("some-custom-vlm", "generic", 28),
    ],
)
def test_resolve_model_spec_aliases(model_name: str, expected_family: str, expected_factor: int):
    spec = resolve_model_spec(model_name)
    assert spec.family == expected_family
    assert spec.factor == expected_factor


def test_resolve_model_spec_overrides_token_bounds():
    base = MODEL_SPECS["qwen3_vl"]
    custom = resolve_model_spec(base, min_tokens=16, max_tokens=512)
    assert custom.family == "qwen3_vl"
    assert custom.min_tokens == 16
    assert custom.max_tokens == 512
    assert custom.max_pixels == 512 * (32**2)


def test_estimate_visual_tokens_qwen25_vs_qwen3():
    # Exact multiple of 28*32 = 896 so both families keep native size.
    est_q25 = estimate_visual_tokens(896, 896, "qwen2_5_vl")
    est_q3 = estimate_visual_tokens(896, 896, "qwen3_vl")

    assert est_q25.resized_width == 896
    assert est_q25.resized_height == 896
    assert est_q25.visual_tokens == (896 // 28) * (896 // 28)  # 32 * 32 = 1024

    assert est_q3.resized_width == 896
    assert est_q3.resized_height == 896
    assert est_q3.visual_tokens == (896 // 32) * (896 // 32)  # 28 * 28 = 784
    assert est_q3.visual_tokens < est_q25.visual_tokens


def test_estimate_visual_tokens_respects_max_tokens_budget():
    est = estimate_visual_tokens(8000, 6000, "qwen3_vl", max_tokens=1024)
    assert est.visual_tokens <= 1024
    assert est.min_scale < 0.25


def test_estimate_visual_tokens_internvl_dynamic_tiles_and_thumbnail():
    # Large 2:1 image should use multiple 448x448 tiles + 1 thumbnail (256 tokens each).
    est_large = estimate_visual_tokens(1792, 896, "internvl")
    assert est_large.model_family == "internvl"
    assert est_large.visual_tokens % 256 == 0
    # 4x2=8 tiles + 1 thumbnail = 9 tiles -> 2304 tokens (or similar multi-tile + thumbnail)
    assert est_large.visual_tokens >= 3 * 256

    # Small square image -> 1x1 tile, no extra thumbnail needed.
    est_small = estimate_visual_tokens(448, 448, "internvl")
    assert est_small.visual_tokens == 256

    # Verify custom VisionModelSpec pass-through works
    no_thumb = VisionModelSpec(
        family="internvl",
        patch_size=14,
        merge_size=2,
        min_tokens=256,
        max_tokens=256 * 4,
        tile_size=448,
        tokens_per_tile=256,
        max_dynamic_tiles=4,
        use_thumbnail=False,
    )
    est_nt = estimate_visual_tokens(896, 896, no_thumb)
    assert est_nt.visual_tokens == 4 * 256

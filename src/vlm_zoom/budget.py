"""Exact visual token estimators and budget planners for major VLM families.

Supports exact `smart_resize` token accounting for Qwen2-VL, Qwen2.5-VL, Qwen3-VL,
and Qwen3.5, dynamic tile counting for InternVL, and generic patch-grid models.
Zero PyTorch/CUDA dependency — operates strictly on image `(width, height)` headers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ModelFamily = Literal[
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_5",
    "internvl",
    "generic",
]


@dataclass(frozen=True)
class VisionModelSpec:
    """Visual tokenization geometry for a VLM family."""

    family: ModelFamily
    patch_size: int
    merge_size: int
    min_tokens: int
    max_tokens: int
    max_aspect_ratio: float = 200.0
    # InternVL dynamic tiling parameters
    tile_size: int | None = None
    tokens_per_tile: int | None = None
    max_dynamic_tiles: int = 12
    use_thumbnail: bool = True

    @property
    def factor(self) -> int:
        """Spatial divisibility factor (`patch_size * merge_size`)."""
        return self.patch_size * self.merge_size

    @property
    def min_pixels(self) -> int:
        return self.min_tokens * (self.factor**2)

    @property
    def max_pixels(self) -> int:
        return self.max_tokens * (self.factor**2)


MODEL_SPECS: dict[ModelFamily, VisionModelSpec] = {
    "qwen2_vl": VisionModelSpec(
        family="qwen2_vl",
        patch_size=14,
        merge_size=2,
        min_tokens=4,
        max_tokens=16384,
    ),
    "qwen2_5_vl": VisionModelSpec(
        family="qwen2_5_vl",
        patch_size=14,
        merge_size=2,
        min_tokens=4,
        max_tokens=16384,
    ),
    "qwen3_vl": VisionModelSpec(
        family="qwen3_vl",
        patch_size=16,
        merge_size=2,
        min_tokens=4,
        max_tokens=16384,
    ),
    "qwen3_5": VisionModelSpec(
        family="qwen3_5",
        patch_size=16,
        merge_size=2,
        min_tokens=4,
        max_tokens=16384,
    ),
    "internvl": VisionModelSpec(
        family="internvl",
        patch_size=14,
        merge_size=2,
        min_tokens=256,
        max_tokens=256 * 13,
        tile_size=448,
        tokens_per_tile=256,
        max_dynamic_tiles=12,
        use_thumbnail=True,
    ),
    "generic": VisionModelSpec(
        family="generic",
        patch_size=14,
        merge_size=2,
        min_tokens=4,
        max_tokens=16384,
    ),
}


def resolve_model_spec(
    model_or_family: str | VisionModelSpec,
    *,
    max_tokens: int | None = None,
    min_tokens: int | None = None,
) -> VisionModelSpec:
    """Resolve a model name or family key into a `VisionModelSpec`."""
    if isinstance(model_or_family, VisionModelSpec):
        spec = model_or_family
    else:
        key = model_or_family.lower().strip()
        if key in MODEL_SPECS:
            spec = MODEL_SPECS[key]  # type: ignore[index]
        elif "qwen3.5" in key or "qwen3_5" in key or "qwen3-5" in key:
            spec = MODEL_SPECS["qwen3_5"]
        elif "qwen3" in key:
            spec = MODEL_SPECS["qwen3_vl"]
        elif "qwen2.5" in key or "qwen2_5" in key or "qwen2-5" in key:
            spec = MODEL_SPECS["qwen2_5_vl"]
        elif "qwen2" in key:
            spec = MODEL_SPECS["qwen2_vl"]
        elif "internvl" in key:
            spec = MODEL_SPECS["internvl"]
        else:
            spec = MODEL_SPECS["generic"]

    if max_tokens is None and min_tokens is None:
        return spec
    return VisionModelSpec(
        family=spec.family,
        patch_size=spec.patch_size,
        merge_size=spec.merge_size,
        min_tokens=min_tokens if min_tokens is not None else spec.min_tokens,
        max_tokens=max_tokens if max_tokens is not None else spec.max_tokens,
        max_aspect_ratio=spec.max_aspect_ratio,
        tile_size=spec.tile_size,
        tokens_per_tile=spec.tokens_per_tile,
        max_dynamic_tiles=spec.max_dynamic_tiles,
        use_thumbnail=spec.use_thumbnail,
    )


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    width: int,
    height: int,
    *,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
    max_aspect_ratio: float = 200.0,
) -> tuple[int, int]:
    """Exact Qwen-VL `smart_resize` returning `(resized_width, resized_height)`.

    Guarantees both output dimensions are multiples of `factor` and their product
    lies within `[min_pixels, max_pixels]` while closely preserving aspect ratio.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({width}, {height}).")
    if max_pixels < min_pixels:
        raise ValueError(f"max_pixels ({max_pixels}) must be >= min_pixels ({min_pixels}).")
    aspect = max(height, width) / min(height, width)
    if aspect > max_aspect_ratio:
        raise ValueError(f"Absolute aspect ratio {aspect:.2f} exceeds maximum {max_aspect_ratio}.")

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return w_bar, h_bar


@dataclass(frozen=True)
class TokenEstimate:
    """Exact visual token and resized-geometry estimate for an image."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    visual_tokens: int
    scale_x: float
    scale_y: float
    model_family: ModelFamily

    @property
    def min_scale(self) -> float:
        """Smallest linear scale factor (< 1.0 means detail was downscaled away)."""
        return min(self.scale_x, self.scale_y)


def _internvl_best_grid(width: int, height: int, max_tiles: int, tile_size: int) -> tuple[int, int]:
    aspect_ratio = width / height
    area = width * height
    best_diff = float("inf")
    best_grid = (1, 1)
    for n in range(1, max_tiles + 1):
        for cols in range(1, n + 1):
            if n % cols != 0:
                continue
            rows = n // cols
            target_ratio = cols / rows
            diff = abs(aspect_ratio - target_ratio)
            if diff < best_diff or (
                math.isclose(diff, best_diff) and area > 0.5 * tile_size * tile_size * cols * rows
            ):
                best_diff = diff
                best_grid = (cols, rows)
    return best_grid


def estimate_visual_tokens(
    width: int,
    height: int,
    model_or_family: str | VisionModelSpec = "qwen3_vl",
    *,
    max_tokens: int | None = None,
    min_tokens: int | None = None,
) -> TokenEstimate:
    """Compute exact visual token count and effective scale for `(width, height)`."""
    spec = resolve_model_spec(model_or_family, max_tokens=max_tokens, min_tokens=min_tokens)
    if spec.family == "internvl" and spec.tile_size and spec.tokens_per_tile:
        cols, rows = _internvl_best_grid(width, height, spec.max_dynamic_tiles, spec.tile_size)
        n_tiles = cols * rows
        total_tiles = n_tiles + (1 if spec.use_thumbnail and n_tiles > 1 else 0)
        rw, rh = cols * spec.tile_size, rows * spec.tile_size
        return TokenEstimate(
            original_width=width,
            original_height=height,
            resized_width=rw,
            resized_height=rh,
            visual_tokens=total_tiles * spec.tokens_per_tile,
            scale_x=rw / width,
            scale_y=rh / height,
            model_family=spec.family,
        )

    rw, rh = smart_resize(
        width,
        height,
        factor=spec.factor,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
        max_aspect_ratio=spec.max_aspect_ratio,
    )
    grid_w = rw // spec.patch_size
    grid_h = rh // spec.patch_size
    tokens = (grid_w * grid_h) // (spec.merge_size**2)
    return TokenEstimate(
        original_width=width,
        original_height=height,
        resized_width=rw,
        resized_height=rh,
        visual_tokens=tokens,
        scale_x=rw / width,
        scale_y=rh / height,
        model_family=spec.family,
    )

"""Adaptive high-resolution tiling, ROI cropping, and pinned-context builders.

Designed for ultra-high-resolution imagery (remote sensing, geological maps,
engineering CAD blueprints, 4K/8K GUI screenshots, dense PDF pages) where direct
downscaling destroys small text/symbols while raw full-res inference explodes
visual tokens and KV cache memory.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image

from .budget import TokenEstimate, VisionModelSpec, estimate_visual_tokens, resolve_model_spec


@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned box `[x1, y1, x2, y2]` in pixel coordinates of the original image."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamp(self, max_w: int, max_h: int) -> PixelBox:
        x1 = max(0.0, min(float(max_w), self.x1))
        y1 = max(0.0, min(float(max_h), self.y1))
        x2 = max(x1, min(float(max_w), self.x2))
        y2 = max(y1, min(float(max_h), self.y2))
        return PixelBox(x1, y1, x2, y2)

    def pad(self, ratio: float, max_w: int, max_h: int) -> PixelBox:
        """Expand the box symmetrically by `ratio` of its width/height and clamp."""
        dx = self.width * ratio
        dy = self.height * ratio
        return PixelBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy).clamp(max_w, max_h)

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))


@dataclass(frozen=True)
class TileSpec:
    """Metadata describing one cropped tile and its mapping back to the full image."""

    tile_id: str
    row: int
    col: int
    box: PixelBox  # region in original image coordinates
    sent_width: int  # width actually sent to VLM (after per-tile budget resize)
    sent_height: int  # height actually sent to VLM
    estimated_tokens: int

    def local_to_global_box(
        self,
        box: tuple[float, float, float, float],
        *,
        coord_space: Literal["norm1000", "norm1", "pixel"] = "norm1000",
    ) -> PixelBox:
        """Map a detection box produced on this tile back into original-image pixels."""
        lx1, ly1, lx2, ly2 = box
        if coord_space == "norm1000":
            sx, sy = self.box.width / 1000.0, self.box.height / 1000.0
        elif coord_space == "norm1":
            sx, sy = self.box.width, self.box.height
        elif coord_space == "pixel":
            sx = self.box.width / max(self.sent_width, 1)
            sy = self.box.height / max(self.sent_height, 1)
        else:
            raise ValueError(f"Unsupported coord_space: {coord_space!r}")

        gx1 = self.box.x1 + lx1 * sx
        gy1 = self.box.y1 + ly1 * sy
        gx2 = self.box.x1 + lx2 * sx
        gy2 = self.box.y1 + ly2 * sy
        if gx1 > gx2:
            gx1, gx2 = gx2, gx1
        if gy1 > gy2:
            gy1, gy2 = gy2, gy1
        return PixelBox(gx1, gy1, gx2, gy2)


@dataclass
class Tile:
    """Materialized tile image paired with its coordinate transform metadata."""

    spec: TileSpec
    image: Image.Image

    def to_data_url(
        self, *, fmt: Literal["PNG", "JPEG", "WEBP"] = "JPEG", quality: int = 90
    ) -> str:
        return image_to_data_url(self.image, fmt=fmt, quality=quality)


@dataclass(frozen=True)
class PinnedContext:
    """Shared visual context pinned as a prefix across all tile requests.

    Typical uses:
    - Geological maps / CAD drawings: pin the map **legend** so every zoomed tile
      can cross-reference stratigraphic codes, lithology colors, and fault symbols.
    - Remote sensing / 4K GUI: pin a low-res **global overview** thumbnail so every
      tile retains global spatial awareness.

    Because the exact same base64 payload sits at the start of every tile's message
    history, vLLM Automatic Prefix Caching (`--enable-prefix-caching`) computes the
    visual KV cache for the pinned context **once** and reuses it across all tiles.
    """

    label: str
    image: Image.Image
    source_box: PixelBox
    estimated_tokens: int
    data_url: str


@dataclass(frozen=True)
class TilingPlan:
    """Dry-run tiling plan computed purely from image dimensions (zero pixel decode)."""

    image_width: int
    image_height: int
    roi: PixelBox
    rows: int
    cols: int
    overlap: float
    tiles: tuple[TileSpec, ...]
    direct_estimate: TokenEstimate
    max_tokens_per_tile: int

    @property
    def total_tile_tokens(self) -> int:
        return sum(t.estimated_tokens for t in self.tiles)

    @property
    def min_effective_scale(self) -> float:
        """Minimum linear preservation ratio across all tiles (1.0 = lossless native pixels)."""
        if not self.tiles:
            return 1.0
        return min(
            min(t.sent_width / max(t.box.width, 1.0), t.sent_height / max(t.box.height, 1.0))
            for t in self.tiles
        )


def load_image(source: str | Path | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(source) as img:
        return img.convert("RGB")


def read_image_size(source: str | Path | Image.Image) -> tuple[int, int]:
    """Read `(width, height)` lazily from image headers without decoding pixels."""
    if isinstance(source, Image.Image):
        return source.size
    with Image.open(source) as img:
        return img.size


def image_to_data_url(
    image: Image.Image,
    *,
    fmt: Literal["PNG", "JPEG", "WEBP"] = "JPEG",
    quality: int = 90,
) -> str:
    buf = BytesIO()
    save_kwargs = {"quality": quality} if fmt in ("JPEG", "WEBP") else {}
    img = image.convert("RGB") if fmt == "JPEG" and image.mode != "RGB" else image
    img.save(buf, format=fmt, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def resize_to_token_budget(
    image: Image.Image,
    model_or_family: str | VisionModelSpec = "qwen3_vl",
    *,
    max_tokens: int = 1024,
) -> tuple[Image.Image, TokenEstimate]:
    """Downscale an image so it fits cleanly inside `max_tokens` visual tokens."""
    est = estimate_visual_tokens(image.width, image.height, model_or_family, max_tokens=max_tokens)
    if est.resized_width < image.width or est.resized_height < image.height:
        resized = image.resize((est.resized_width, est.resized_height), Image.Resampling.LANCZOS)
    else:
        resized = image
    return resized, est


def build_pinned_context(
    image: Image.Image,
    *,
    label: str = "Map Legend / Global Context",
    box: PixelBox | tuple[float, float, float, float] | None = None,
    model_or_family: str | VisionModelSpec = "qwen3_vl",
    max_tokens: int = 1024,
    fmt: Literal["PNG", "JPEG", "WEBP"] = "PNG",
    quality: int = 92,
) -> PinnedContext:
    """Create a prefix-cacheable `PinnedContext` from a legend box or full-image overview."""
    if box is None:
        pbox = PixelBox(0.0, 0.0, float(image.width), float(image.height))
        cropped = image
    else:
        pbox = box if isinstance(box, PixelBox) else PixelBox(*box)
        pbox = pbox.clamp(image.width, image.height)
        cropped = image.crop(pbox.to_int_tuple())

    resized, est = resize_to_token_budget(cropped, model_or_family, max_tokens=max_tokens)
    data_url = image_to_data_url(resized, fmt=fmt, quality=quality)
    return PinnedContext(
        label=label,
        image=resized,
        source_box=pbox,
        estimated_tokens=est.visual_tokens,
        data_url=data_url,
    )


class AdaptiveTiler:
    """Token-budget-aware high-resolution tiler.

    Given an image (or ROI sub-box), chooses the smallest `(rows, cols)` grid such
    that each tile stays within `max_tokens_per_tile` while keeping linear detail
    scale >= `min_detail_scale` (default 0.85, i.e. nearly native resolution).
    """

    def __init__(
        self,
        model_or_family: str | VisionModelSpec = "qwen3_vl",
        *,
        max_tokens_per_tile: int = 1024,
        overlap: float = 0.15,
        min_detail_scale: float = 0.85,
        max_tiles: int = 36,
    ) -> None:
        if not 0.0 <= overlap < 0.5:
            raise ValueError(f"overlap must be in [0.0, 0.5), got {overlap}.")
        if max_tokens_per_tile <= 0:
            raise ValueError(f"max_tokens_per_tile must be positive, got {max_tokens_per_tile}.")
        if not 0.1 <= min_detail_scale <= 1.0:
            raise ValueError(f"min_detail_scale must be in [0.1, 1.0], got {min_detail_scale}.")
        if max_tiles < 1:
            raise ValueError(f"max_tiles must be >= 1, got {max_tiles}.")

        self.spec = resolve_model_spec(model_or_family)
        self.max_tokens_per_tile = max_tokens_per_tile
        self.overlap = overlap
        self.min_detail_scale = min_detail_scale
        self.max_tiles = max_tiles

    def _choose_grid(
        self,
        roi_w: float,
        roi_h: float,
        *,
        grid: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        if grid is not None:
            rows, cols = grid
            if rows < 1 or cols < 1:
                raise ValueError(f"grid dimensions must be >= 1, got {grid}.")
            return rows, cols

        # Maximum native pixel area one tile can carry at `min_detail_scale`.
        tile_pixel_cap = self.max_tokens_per_tile * (self.spec.factor**2)
        effective_tile_area = tile_pixel_cap / (self.min_detail_scale**2)
        roi_area = roi_w * roi_h
        if roi_area <= effective_tile_area:
            return 1, 1

        # Compute aspect-matched grid size accounting for overlap expansion.
        stride_ratio = 1.0 - self.overlap
        target_side = math.sqrt(effective_tile_area)
        cols = max(
            1, math.ceil((roi_w - target_side * self.overlap) / (target_side * stride_ratio))
        )
        rows = max(
            1, math.ceil((roi_h - target_side * self.overlap) / (target_side * stride_ratio))
        )

        # Respect max_tiles cap while keeping the grid aspect ratio balanced.
        while rows * cols > self.max_tiles:
            if cols >= rows and cols > 1:
                cols -= 1
            elif rows > 1:
                rows -= 1
            else:
                break
        return rows, cols

    def plan(
        self,
        width: int,
        height: int,
        *,
        roi: PixelBox | tuple[float, float, float, float] | None = None,
        grid: tuple[int, int] | None = None,
    ) -> TilingPlan:
        """Build a `TilingPlan` strictly from dimensions (no pixel decoding)."""
        direct_est = estimate_visual_tokens(width, height, self.spec)
        if roi is None:
            roi_box = PixelBox(0.0, 0.0, float(width), float(height))
        else:
            roi_box = roi if isinstance(roi, PixelBox) else PixelBox(*roi)
            roi_box = roi_box.clamp(width, height)

        rows, cols = self._choose_grid(roi_box.width, roi_box.height, grid=grid)

        # Compute tile spans with exact overlap:
        # For N tiles with overlap `o`, tile_w * (1 + (N - 1) * (1 - o)) = roi_w.
        denom_x = 1.0 + (cols - 1) * (1.0 - self.overlap)
        denom_y = 1.0 + (rows - 1) * (1.0 - self.overlap)
        tile_w = roi_box.width / denom_x
        tile_h = roi_box.height / denom_y
        step_x = tile_w * (1.0 - self.overlap)
        step_y = tile_h * (1.0 - self.overlap)

        tiles: list[TileSpec] = []
        for r in range(rows):
            y1 = roi_box.y1 + r * step_y
            y2 = roi_box.y2 if r == rows - 1 else min(roi_box.y2, y1 + tile_h)
            for c in range(cols):
                x1 = roi_box.x1 + c * step_x
                x2 = roi_box.x2 if c == cols - 1 else min(roi_box.x2, x1 + tile_w)
                tbox = PixelBox(x1, y1, x2, y2).clamp(width, height)
                tw_int = max(1, int(round(tbox.width)))
                th_int = max(1, int(round(tbox.height)))
                est = estimate_visual_tokens(
                    tw_int,
                    th_int,
                    self.spec,
                    max_tokens=self.max_tokens_per_tile,
                )
                sent_w = min(tw_int, est.resized_width)
                sent_h = min(th_int, est.resized_height)
                tiles.append(
                    TileSpec(
                        tile_id=f"r{r}_c{c}",
                        row=r,
                        col=c,
                        box=tbox,
                        sent_width=sent_w,
                        sent_height=sent_h,
                        estimated_tokens=est.visual_tokens,
                    )
                )

        return TilingPlan(
            image_width=width,
            image_height=height,
            roi=roi_box,
            rows=rows,
            cols=cols,
            overlap=self.overlap,
            tiles=tuple(tiles),
            direct_estimate=direct_est,
            max_tokens_per_tile=self.max_tokens_per_tile,
        )

    def tile(
        self,
        source: str | Path | Image.Image,
        *,
        roi: PixelBox | tuple[float, float, float, float] | None = None,
        grid: tuple[int, int] | None = None,
    ) -> tuple[TilingPlan, list[Tile]]:
        """Crop and budget-resize all tiles from an image."""
        image = load_image(source)
        plan = self.plan(image.width, image.height, roi=roi, grid=grid)
        materialized: list[Tile] = []
        for spec in plan.tiles:
            crop = image.crop(spec.box.to_int_tuple())
            if crop.width > spec.sent_width or crop.height > spec.sent_height:
                crop = crop.resize((spec.sent_width, spec.sent_height), Image.Resampling.LANCZOS)
            materialized.append(Tile(spec=spec, image=crop))
        return plan, materialized

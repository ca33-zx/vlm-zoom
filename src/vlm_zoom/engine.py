"""High-level Glance-and-Zoom & Adaptive Grid inference orchestrator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from .budget import VisionModelSpec, resolve_model_spec
from .client import VLMBackend, VLMResponse, build_tile_messages
from .fusion import Detection, fuse_detections_nms, fuse_detections_wbf, parse_raw_boxes
from .geo import GeoBox, GeoTransform
from .tiler import (
    AdaptiveTiler,
    PinnedContext,
    PixelBox,
    Tile,
    TilingPlan,
    build_pinned_context,
    image_to_data_url,
    load_image,
    resize_to_token_budget,
)

DEFAULT_GLANCE_PROMPT = (
    "You are inspecting a low-resolution overview of a high-resolution image. "
    "Identify the rectangular region(s) of interest needed to answer the user's question. "
    'Return ONLY a JSON array of objects with `"label"` and `"bbox_2d": [x1, y1, x2, y2]` '
    "in normalized [0, 1000] coordinates."
)


@dataclass(frozen=True)
class TileResult:
    """Output of running the VLM on a single high-resolution tile."""

    tile_id: str
    box: PixelBox
    text: str
    detections: tuple[Detection, ...]
    response: VLMResponse


@dataclass(frozen=True)
class TokenUsageSummary:
    """Visual token budget and prefix-cache savings summary."""

    direct_full_res_tokens: int
    glance_tokens: int
    pinned_prefix_tokens: int
    total_tile_tokens: int
    num_tiles: int
    # With vLLM prefix caching, pinned context is encoded once instead of `num_tiles` times.
    estimated_prefix_tokens_saved: int
    server_prompt_tokens: int = 0
    server_cached_tokens: int = 0

    @property
    def effective_visual_tokens(self) -> int:
        """Total uncached visual tokens processed across Stage 1 + Stage 2."""
        return self.glance_tokens + self.pinned_prefix_tokens + self.total_tile_tokens


@dataclass
class ZoomResult:
    """End-to-end result of `ZoomEngine.run()`."""

    mode: Literal["grid", "glance_and_zoom"]
    image_size: tuple[int, int]
    plans: list[TilingPlan]
    tile_results: list[TileResult]
    detections: list[Detection]
    token_usage: TokenUsageSummary
    glance_rois: list[PixelBox] = field(default_factory=list)
    geo_detections: list[tuple[Detection, GeoBox]] = field(default_factory=list)


class ZoomEngine:
    """Orchestrates adaptive tiling, pinned-context prefix caching, and Glance-and-Zoom.

    Supports two execution modes:
      - ``mode="grid"``: tiles the entire image (or explicit ROI) within exact per-tile
        visual token budgets, runs all tiles concurrently on vLLM, and fuses detections.
      - ``mode="glance_and_zoom"``: Stage 1 sends a compact overview image to locate
        relevant ROI bounding boxes; Stage 2 zooms ONLY into those padded ROIs at
        high resolution (saving massive tokens on 8K remote sensing / geological maps).
    """

    def __init__(
        self,
        backend: VLMBackend,
        *,
        model_family: str | VisionModelSpec = "qwen3_vl",
        max_tokens_per_tile: int = 1024,
        overlap: float = 0.15,
        min_detail_scale: float = 0.85,
        max_tiles: int = 36,
        coord_space: Literal["norm1000", "norm1", "pixel"] = "norm1000",
        fusion_method: Literal["wbf", "nms"] = "wbf",
        iou_threshold: float = 0.5,
        image_format: Literal["JPEG", "PNG", "WEBP"] = "JPEG",
        image_quality: int = 90,
    ) -> None:
        self.backend = backend
        self.spec = resolve_model_spec(model_family)
        self.tiler = AdaptiveTiler(
            self.spec,
            max_tokens_per_tile=max_tokens_per_tile,
            overlap=overlap,
            min_detail_scale=min_detail_scale,
            max_tiles=max_tiles,
        )
        self.coord_space = coord_space
        self.fusion_method = fusion_method
        self.iou_threshold = iou_threshold
        self.image_format = image_format
        self.image_quality = image_quality

    async def _run_tile(
        self,
        tile: Tile,
        *,
        prompt: str,
        pinned_contexts: list[PinnedContext],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> TileResult:
        messages = build_tile_messages(
            prompt=prompt,
            tile=tile,
            pinned_contexts=pinned_contexts,
            system_prompt=system_prompt,
            image_format=self.image_format,
            image_quality=self.image_quality,
        )
        resp = await self.backend.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw_boxes = parse_raw_boxes(resp.text)
        dets = tuple(
            Detection(
                label=label,
                box=tile.spec.local_to_global_box(box, coord_space=self.coord_space),
                score=score,
                source_tile=tile.spec.tile_id,
            )
            for label, box, score in raw_boxes
        )
        return TileResult(
            tile_id=tile.spec.tile_id,
            box=tile.spec.box,
            text=resp.text,
            detections=dets,
            response=resp,
        )

    async def _run_glance(
        self,
        image: Image.Image,
        *,
        query: str,
        glance_max_tokens: int,
        roi_pad_ratio: float,
        max_rois: int,
    ) -> tuple[list[PixelBox], int, VLMResponse]:
        overview, est = resize_to_token_budget(image, self.spec, max_tokens=glance_max_tokens)
        data_url = image_to_data_url(overview, fmt=self.image_format, quality=self.image_quality)
        messages = [
            {"role": "system", "content": DEFAULT_GLANCE_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": f"Question: {query}"},
                ],
            },
        ]
        resp = await self.backend.complete(messages=messages, max_tokens=256, temperature=0.0)
        raw_boxes = parse_raw_boxes(resp.text)
        rois: list[PixelBox] = []
        for _, (x1, y1, x2, y2), _ in raw_boxes[:max_rois]:
            gbox = PixelBox(
                x1 * image.width / 1000.0,
                y1 * image.height / 1000.0,
                x2 * image.width / 1000.0,
                y2 * image.height / 1000.0,
            ).clamp(image.width, image.height)
            if gbox.area > 0:
                rois.append(gbox.pad(roi_pad_ratio, image.width, image.height))
        return rois, est.visual_tokens, resp

    async def run_async(
        self,
        source: str | Path | Image.Image,
        prompt: str,
        *,
        mode: Literal["grid", "glance_and_zoom"] = "grid",
        roi: PixelBox | tuple[float, float, float, float] | None = None,
        grid: tuple[int, int] | None = None,
        legend_box: PixelBox | tuple[float, float, float, float] | None = None,
        pin_overview: bool = False,
        pinned_max_tokens: int = 1024,
        glance_max_tokens: int = 1024,
        roi_pad_ratio: float = 0.1,
        max_rois: int = 4,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        geo_transform: GeoTransform | None = None,
    ) -> ZoomResult:
        """Run high-resolution inference asynchronously across all planned tiles."""
        image = load_image(source)

        pinned_contexts: list[PinnedContext] = []
        if pin_overview:
            pinned_contexts.append(
                build_pinned_context(
                    image,
                    label="Global Overview",
                    box=None,
                    model_or_family=self.spec,
                    max_tokens=pinned_max_tokens,
                    fmt=self.image_format,
                    quality=self.image_quality,
                )
            )
        if legend_box is not None:
            pinned_contexts.append(
                build_pinned_context(
                    image,
                    label="Map Legend",
                    box=legend_box,
                    model_or_family=self.spec,
                    max_tokens=pinned_max_tokens,
                    fmt="PNG",  # lossless for crisp tiny legend text & color swatches
                )
            )

        glance_rois: list[PixelBox] = []
        glance_tokens = 0
        server_prompt = 0
        server_cached = 0

        target_rois: list[PixelBox | None]
        if mode == "glance_and_zoom" and roi is None:
            found_rois, glance_tokens, g_resp = await self._run_glance(
                image,
                query=prompt,
                glance_max_tokens=glance_max_tokens,
                roi_pad_ratio=roi_pad_ratio,
                max_rois=max_rois,
            )
            server_prompt += g_resp.prompt_tokens
            server_cached += g_resp.cached_tokens
            glance_rois = found_rois
            target_rois = list(found_rois) if found_rois else [None]
        else:
            target_rois = [roi if isinstance(roi, PixelBox) else (PixelBox(*roi) if roi else None)]

        plans: list[TilingPlan] = []
        all_tiles: list[Tile] = []
        for idx, rbox in enumerate(target_rois):
            plan, tiles = self.tiler.tile(image, roi=rbox, grid=grid)
            plans.append(plan)
            if len(target_rois) > 1:
                for t in tiles:
                    t.spec = type(t.spec)(
                        tile_id=f"roi{idx}_{t.spec.tile_id}",
                        row=t.spec.row,
                        col=t.spec.col,
                        box=t.spec.box,
                        sent_width=t.spec.sent_width,
                        sent_height=t.spec.sent_height,
                        estimated_tokens=t.spec.estimated_tokens,
                    )
            all_tiles.extend(tiles)

        tile_results = list(
            await asyncio.gather(
                *(
                    self._run_tile(
                        tile,
                        prompt=prompt,
                        pinned_contexts=pinned_contexts,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    for tile in all_tiles
                )
            )
        )

        raw_dets: list[Detection] = []
        for tr in tile_results:
            raw_dets.extend(tr.detections)
            server_prompt += tr.response.prompt_tokens
            server_cached += tr.response.cached_tokens

        if self.fusion_method == "wbf":
            fused = fuse_detections_wbf(raw_dets, iou_threshold=self.iou_threshold)
        else:
            fused = fuse_detections_nms(raw_dets, iou_threshold=self.iou_threshold)

        pinned_tokens = sum(c.estimated_tokens for c in pinned_contexts)
        num_tiles = len(all_tiles)
        token_usage = TokenUsageSummary(
            direct_full_res_tokens=plans[0].direct_estimate.visual_tokens if plans else 0,
            glance_tokens=glance_tokens,
            pinned_prefix_tokens=pinned_tokens,
            total_tile_tokens=sum(p.total_tile_tokens for p in plans),
            num_tiles=num_tiles,
            estimated_prefix_tokens_saved=pinned_tokens * max(0, num_tiles - 1),
            server_prompt_tokens=server_prompt,
            server_cached_tokens=server_cached,
        )

        geo_dets = (
            [(d, geo_transform.pixel_box_to_geo(d.box)) for d in fused]
            if geo_transform is not None
            else []
        )

        return ZoomResult(
            mode=mode,
            image_size=image.size,
            plans=plans,
            tile_results=tile_results,
            detections=fused,
            token_usage=token_usage,
            glance_rois=glance_rois,
            geo_detections=geo_dets,
        )

    def run(self, source: str | Path | Image.Image, prompt: str, **kwargs) -> ZoomResult:
        """Synchronous convenience wrapper around `run_async()`."""
        return asyncio.run(self.run_async(source, prompt, **kwargs))

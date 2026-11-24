"""Async tests for ZoomEngine, pinned-context prefix caching, Glance-and-Zoom, and WBF."""

from __future__ import annotations

import json
from typing import Any

import pytest
from PIL import Image

from vlm_zoom.client import VLMResponse
from vlm_zoom.engine import ZoomEngine
from vlm_zoom.geo import GeoTransform
from vlm_zoom.tiler import PixelBox


class MockVLMBackend:
    """Deterministic async mock VLM recording every request message list."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> VLMResponse:
        self.calls.append(messages)
        return self.responder(messages)


@pytest.mark.asyncio
async def test_grid_mode_pins_byte_identical_prefix_and_fuses_cross_tile_detections():
    img = Image.new("RGB", (3200, 2000), (245, 245, 240))
    legend_box = PixelBox(2600.0, 100.0, 3150.0, 1900.0)
    map_roi = PixelBox(0.0, 0.0, 2500.0, 2000.0)

    # Return a detection that lands around global (1250, 1000) in any tile covering that point.
    def responder(messages: list[dict[str, Any]]) -> VLMResponse:
        last_text = messages[-1]["content"][1]["text"]
        # Parse tile bounds from header: x=[x1, x2], y=[y1, y2]
        import re

        m = re.search(r"x=\[(\d+), (\d+)\], y=\[(\d+), (\d+)\]", last_text)
        assert m is not None
        x1, x2, y1, y2 = (int(m.group(i)) for i in range(1, 5))
        # Check whether target point (1250, 1000) lies inside this tile
        if x1 <= 1250 <= x2 and y1 <= 1000 <= y2:
            lx1 = (1200 - x1) / (x2 - x1) * 1000.0
            ly1 = (950 - y1) / (y2 - y1) * 1000.0
            lx2 = (1300 - x1) / (x2 - x1) * 1000.0
            ly2 = (1050 - y1) / (y2 - y1) * 1000.0
            body = json.dumps(
                [{"label": "Fault F1", "bbox_2d": [lx1, ly1, lx2, ly2], "score": 0.9}]
            )
            return VLMResponse(
                text=body, prompt_tokens=1500, completion_tokens=30, cached_tokens=700
            )
        return VLMResponse(text="[]", prompt_tokens=1500, completion_tokens=5, cached_tokens=700)

    backend = MockVLMBackend(responder)
    engine = ZoomEngine(
        backend,
        model_family="qwen3_vl",
        max_tokens_per_tile=512,
        overlap=0.2,
        fusion_method="wbf",
    )
    geo_tf = GeoTransform(2.0, 0.0, 0.0, -2.0, 500000.0, 3500000.0, crs="EPSG:32650")

    result = await engine.run_async(
        img,
        "Locate major fault lines and classify lithology using the map legend.",
        mode="grid",
        roi=map_roi,
        grid=(2, 2),
        legend_box=legend_box,
        pin_overview=True,
        system_prompt="You are an expert geological cartographer.",
        geo_transform=geo_tf,
    )

    assert result.mode == "grid"
    assert len(result.tile_results) == 4
    assert len(backend.calls) == 4

    # Critical vLLM Automatic Prefix Caching invariant:
    # Messages [0:-1] (system + pinned overview + pinned legend + assistant ack)
    # MUST be 100% byte-identical across every single tile request!
    first_prefix = json.dumps(backend.calls[0][:-1], sort_keys=True)
    for call_msgs in backend.calls[1:]:
        assert json.dumps(call_msgs[:-1], sort_keys=True) == first_prefix

    # All 4 overlapping 2x2 tiles cover center (1250, 1000) -> WBF merges into 1 clean detection!
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.label == "Fault F1"
    assert det.metadata["merged_count"] == 4
    assert det.box.x1 == pytest.approx(1200.0, abs=2.0)
    assert det.box.y1 == pytest.approx(950.0, abs=2.0)
    assert det.box.x2 == pytest.approx(1300.0, abs=2.0)
    assert det.box.y2 == pytest.approx(1050.0, abs=2.0)

    # Geospatial projection populated for every fused detection
    assert len(result.geo_detections) == 1
    _, gbox = result.geo_detections[0]
    assert gbox.crs == "EPSG:32650"
    assert gbox.min_x == pytest.approx(502400.0, abs=4.0)

    # Token usage tracks prefix-cache savings across (num_tiles - 1) tiles
    usage = result.token_usage
    assert usage.num_tiles == 4
    assert usage.pinned_prefix_tokens > 0
    assert usage.estimated_prefix_tokens_saved == usage.pinned_prefix_tokens * 3
    assert usage.server_cached_tokens == 4 * 700


@pytest.mark.asyncio
async def test_glance_and_zoom_only_tiles_stage1_roi():
    img = Image.new("RGB", (4800, 3000), (240, 240, 240))

    def responder(messages: list[dict[str, Any]]) -> VLMResponse:
        # Stage 1 Glance call has the glance system prompt
        if messages[0]["role"] == "system" and "overview" in messages[0]["content"].lower():
            # Return a compact ROI in top-left quadrant: [100, 100, 300, 300] (norm1000)
            return VLMResponse(
                text='[{"label": "fault_zone", "bbox_2d": [100, 100, 300, 300]}]',
                prompt_tokens=900,
                completion_tokens=20,
            )
        # Stage 2 Zoom call
        return VLMResponse(
            text='[{"label": "Ore Vein #3", "bbox_2d": [200, 200, 600, 600], "score": 0.95}]',
            prompt_tokens=800,
            completion_tokens=25,
        )

    backend = MockVLMBackend(responder)
    engine = ZoomEngine(
        backend,
        model_family="qwen3_vl",
        max_tokens_per_tile=1024,
        fusion_method="nms",
    )

    result = await engine.run_async(
        img,
        "Find Ore Vein #3 and report its coordinates.",
        mode="glance_and_zoom",
        roi_pad_ratio=0.1,
    )

    assert result.mode == "glance_and_zoom"
    assert len(result.glance_rois) == 1
    assert result.token_usage.glance_tokens > 0
    # Small ROI (~20% of width & height) -> Stage 2 needs only 1-2 tiles instead of full 4.8K grid!
    assert result.token_usage.num_tiles <= 2
    assert len(result.detections) == 1
    assert result.detections[0].label == "Ore Vein #3"
    # Detection must map inside the Stage-1 ROI in global pixel coordinates
    roi = result.glance_rois[0]
    det_box = result.detections[0].box
    assert det_box.x1 >= roi.x1 - 1.0
    assert det_box.y1 >= roi.y1 - 1.0
    assert det_box.x2 <= roi.x2 + 1.0
    assert det_box.y2 <= roi.y2 + 1.0


@pytest.mark.asyncio
async def test_glance_and_zoom_falls_back_to_full_image_when_no_roi_returned():
    img = Image.new("RGB", (2000, 1500), (240, 240, 240))

    def responder(messages: list[dict[str, Any]]) -> VLMResponse:
        if messages[0]["role"] == "system" and "overview" in messages[0]["content"].lower():
            return VLMResponse(text="No clear region identified.")
        return VLMResponse(text="[]")

    backend = MockVLMBackend(responder)
    engine = ZoomEngine(backend, model_family="qwen3_vl", max_tokens_per_tile=512)
    result = await engine.run_async(img, "Inspect all details.", mode="glance_and_zoom")

    assert result.glance_rois == []
    assert len(result.plans) == 1
    assert result.plans[0].roi == PixelBox(0.0, 0.0, 2000.0, 1500.0)
    assert result.token_usage.num_tiles >= 1


@pytest.mark.asyncio
async def test_multiple_named_pinned_boxes_share_byte_identical_prefix():
    img = Image.new("RGB", (3840, 2160), (20, 25, 35))

    def responder(messages: list[dict[str, Any]]) -> VLMResponse:
        return VLMResponse(text="[]", prompt_tokens=1200, completion_tokens=4, cached_tokens=600)

    backend = MockVLMBackend(responder)
    engine = ZoomEngine(backend, model_family="qwen3_vl", max_tokens_per_tile=512)
    result = await engine.run_async(
        img,
        "Find all rows with ANOMALY status.",
        mode="grid",
        grid=(2, 2),
        pinned_boxes={
            "Top Toolbar": (0, 0, 3840, 74),
            "Table Header": (412, 104, 3800, 210),
        },
    )

    assert len(backend.calls) == 4
    prefix_0 = json.dumps(backend.calls[0][:-1], sort_keys=True)
    assert "[Top Toolbar]" in prefix_0
    assert "[Table Header]" in prefix_0
    for call_msgs in backend.calls[1:]:
        assert json.dumps(call_msgs[:-1], sort_keys=True) == prefix_0
    assert result.token_usage.pinned_prefix_tokens > 0
    assert (
        result.token_usage.estimated_prefix_tokens_saved
        == result.token_usage.pinned_prefix_tokens * 3
    )

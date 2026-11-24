"""Unit tests for cross-tile NMS, WBF fusion, and VLM box output parsers."""

from __future__ import annotations

import pytest

from vlm_zoom.fusion import (
    Detection,
    box_containment,
    box_iou,
    fuse_detections_nms,
    fuse_detections_wbf,
    parse_raw_boxes,
)
from vlm_zoom.tiler import PixelBox


def test_box_iou_and_containment():
    a = PixelBox(0.0, 0.0, 100.0, 100.0)
    b = PixelBox(50.0, 0.0, 150.0, 100.0)
    # Intersection = 50*100 = 5000, union = 15000
    assert box_iou(a, b) == pytest.approx(1.0 / 3.0)
    assert box_iou(a, PixelBox(200.0, 200.0, 300.0, 300.0)) == 0.0

    inner = PixelBox(20.0, 20.0, 60.0, 60.0)
    assert box_containment(inner, a) == pytest.approx(1.0)
    assert box_containment(a, inner) == pytest.approx(0.16)


def test_fuse_detections_nms_cross_tile_and_containment():
    dets = [
        Detection("Fault F1", PixelBox(100, 100, 300, 200), score=0.95, source_tile="r0_c0"),
        # Overlapping duplicate from neighbor tile
        Detection("Fault F1", PixelBox(105, 102, 295, 198), score=0.80, source_tile="r0_c1"),
        # Clipped fragment inside the full box from neighbor tile
        Detection("Fault F1", PixelBox(210, 110, 295, 190), score=0.70, source_tile="r0_c1"),
        # Different class at same position must survive when class_aware=True
        Detection("Ore Vein", PixelBox(100, 100, 300, 200), score=0.85, source_tile="r0_c0"),
    ]
    kept = fuse_detections_nms(
        dets, iou_threshold=0.5, containment_threshold=0.85, class_aware=True
    )
    assert len(kept) == 2
    assert {d.label for d in kept} == {"Fault F1", "Ore Vein"}


def test_fuse_detections_wbf_averages_coordinates_by_score():
    dets = [
        Detection("J3", PixelBox(100, 100, 200, 200), score=0.9, source_tile="r0_c0"),
        Detection("J3", PixelBox(110, 100, 210, 200), score=0.3, source_tile="r0_c1"),
        Detection("K1", PixelBox(500, 500, 600, 600), score=0.8, source_tile="r1_c1"),
    ]
    fused = fuse_detections_wbf(dets, iou_threshold=0.5)
    assert len(fused) == 2

    k1 = next(d for d in fused if d.label == "K1")
    j3 = next(d for d in fused if d.label == "J3")
    assert k1.box == PixelBox(500, 500, 600, 600)
    # Weighted x1 = (100 * 0.9 + 110 * 0.3) / 1.2 = 102.5
    assert j3.box.x1 == pytest.approx(102.5)
    assert j3.box.x2 == pytest.approx(202.5)
    assert j3.score == pytest.approx(0.6)
    assert j3.metadata["merged_count"] == 2
    assert j3.metadata["tiles"] == ["r0_c0", "r0_c1"]


def test_parse_raw_boxes_json_fenced_and_bare():
    qwen_json = """Here are the detected geological features:
```json
[
  {"label": "Fault F1", "bbox_2d": [120, 340, 560, 480], "score": 0.92},
  {"name": "J3 Shale", "bbox": [600, 200, 750, 310], "confidence": 0.87}
]
```
"""
    parsed = parse_raw_boxes(qwen_json)
    assert parsed == [
        ("Fault F1", (120.0, 340.0, 560.0, 480.0), 0.92),
        ("J3 Shale", (600.0, 200.0, 750.0, 310.0), 0.87),
    ]

    bare_obj = '{"label": "Granite", "box": [10, 20, 30, 40]}'
    assert parse_raw_boxes(bare_obj) == [("Granite", (10.0, 20.0, 30.0, 40.0), 1.0)]


def test_parse_raw_boxes_legacy_qwen_tags_and_plain_brackets():
    legacy = (
        "<|object_ref_start|>Strike Symbol<|object_ref_end|><|box_start|>(12,34),(56,78)<|box_end|>"
    )
    assert parse_raw_boxes(legacy) == [("Strike Symbol", (12.0, 34.0, 56.0, 78.0), 1.0)]

    plain = "Target located at [100, 200, 350, 420] and also [500, 600, 700, 800]."
    assert parse_raw_boxes(plain, default_label="roi") == [
        ("roi", (100.0, 200.0, 350.0, 420.0), 1.0),
        ("roi", (500.0, 600.0, 700.0, 800.0), 1.0),
    ]

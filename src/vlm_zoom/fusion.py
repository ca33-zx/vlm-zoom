"""Cross-tile bounding-box parsing, NMS, and Weighted Boxes Fusion (WBF).

Handles duplicate detections across overlapping tile margins and parses common
VLM grounding outputs (JSON lists, Qwen2.5-VL `bbox_2d`, Qwen3-VL `[x1, y1, x2, y2]`,
and `<|box_start|>(x1,y1),(x2,y2)<|box_end|>` legacy tags).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .tiler import PixelBox


@dataclass(frozen=True)
class Detection:
    """Single detected object/annotation in original-image pixel coordinates."""

    label: str
    box: PixelBox
    score: float = 1.0
    source_tile: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def box_iou(a: PixelBox, b: PixelBox) -> float:
    """Compute Intersection-over-Union between two `PixelBox` instances."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


def box_containment(inner: PixelBox, outer: PixelBox) -> float:
    """Fraction of `inner`'s area covered by `outer` (useful for clipped tile fragments)."""
    if inner.area <= 0.0:
        return 0.0
    ix1 = max(inner.x1, outer.x1)
    iy1 = max(inner.y1, outer.y1)
    ix2 = min(inner.x2, outer.x2)
    iy2 = min(inner.y2, outer.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / inner.area


def fuse_detections_nms(
    detections: list[Detection],
    *,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.85,
    class_aware: bool = True,
) -> list[Detection]:
    """Suppress duplicate detections across overlapping tiles via NMS + containment."""
    ordered = sorted(detections, key=lambda d: (d.score, d.box.area), reverse=True)
    kept: list[Detection] = []
    for cand in ordered:
        duplicate = False
        for prev in kept:
            if class_aware and prev.label.lower() != cand.label.lower():
                continue
            if box_iou(cand.box, prev.box) >= iou_threshold or (
                box_containment(cand.box, prev.box) >= containment_threshold
                and cand.source_tile != prev.source_tile
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def fuse_detections_wbf(
    detections: list[Detection],
    *,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.85,
    class_aware: bool = True,
) -> list[Detection]:
    """Weighted Boxes Fusion (WBF) for cross-tile detection merging.

    Clusters overlapping detections from multiple tiles and computes a score-weighted
    average box per cluster, tightening localization across tile seams.
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: (d.score, d.box.area), reverse=True)
    clusters: list[list[Detection]] = []
    cluster_boxes: list[PixelBox] = []

    def _weighted_box(members: list[Detection]) -> PixelBox:
        total_w = sum(max(m.score, 1e-6) for m in members)
        return PixelBox(
            x1=sum(m.box.x1 * max(m.score, 1e-6) for m in members) / total_w,
            y1=sum(m.box.y1 * max(m.score, 1e-6) for m in members) / total_w,
            x2=sum(m.box.x2 * max(m.score, 1e-6) for m in members) / total_w,
            y2=sum(m.box.y2 * max(m.score, 1e-6) for m in members) / total_w,
        )

    for det in ordered:
        best_idx = -1
        best_metric = -1.0
        for idx, rep_box in enumerate(cluster_boxes):
            if class_aware and clusters[idx][0].label.lower() != det.label.lower():
                continue
            iou = box_iou(det.box, rep_box)
            cross_tile = all(m.source_tile != det.source_tile for m in clusters[idx])
            cont = max(box_containment(det.box, rep_box), box_containment(rep_box, det.box))
            if iou >= iou_threshold and iou > best_metric:
                best_idx = idx
                best_metric = iou
            elif cross_tile and cont >= containment_threshold and cont > best_metric:
                best_idx = idx
                best_metric = cont

        if best_idx >= 0:
            clusters[best_idx].append(det)
            cluster_boxes[best_idx] = _weighted_box(clusters[best_idx])
        else:
            clusters.append([det])
            cluster_boxes.append(det.box)

    fused: list[Detection] = []
    for members, fbox in zip(clusters, cluster_boxes, strict=True):
        best_member = members[0]
        avg_score = sum(m.score for m in members) / len(members)
        tiles = sorted({m.source_tile for m in members if m.source_tile is not None})
        fused.append(
            Detection(
                label=best_member.label,
                box=fbox,
                score=avg_score,
                source_tile=tiles[0] if len(tiles) == 1 else ",".join(tiles) or None,
                metadata={"merged_count": len(members), "tiles": tiles},
            )
        )
    return sorted(fused, key=lambda d: d.score, reverse=True)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_LEGACY_BOX_RE = re.compile(
    r"(?:<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*)?"
    r"<\|box_start\|>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*<\|box_end\|>"
)
_BRACKET_BOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def parse_raw_boxes(
    text: str,
    *,
    default_label: str = "object",
) -> list[tuple[str, tuple[float, float, float, float], float]]:
    """Extract `(label, (x1, y1, x2, y2), score)` tuples from common VLM text outputs."""
    results: list[tuple[str, tuple[float, float, float, float], float]] = []

    # 1. Try structured JSON (fenced or bare JSON array/object).
    candidates = [m.group(1).strip() for m in _JSON_BLOCK_RE.finditer(text)]
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        candidates.append(stripped)

    for raw in candidates:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                continue
            coords = item.get("bbox_2d") or item.get("bbox") or item.get("box")
            if isinstance(coords, (list, tuple)) and len(coords) == 4:
                try:
                    box = (
                        float(coords[0]),
                        float(coords[1]),
                        float(coords[2]),
                        float(coords[3]),
                    )
                except (TypeError, ValueError):
                    continue
                label = str(item.get("label") or item.get("name") or default_label)
                score = float(item.get("score", item.get("confidence", 1.0)))
                results.append((label, box, score))
        if results:
            return results

    # 2. Try Qwen legacy <|box_start|>(x1,y1),(x2,y2)<|box_end|> syntax.
    for m in _LEGACY_BOX_RE.finditer(text):
        label = (m.group(1) or default_label).strip()
        box = (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
        results.append((label, box, 1.0))
    if results:
        return results

    # 3. Fallback: plain `[x1, y1, x2, y2]` coordinates in text.
    for m in _BRACKET_BOX_RE.finditer(text):
        box = (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
        results.append((default_label, box, 1.0))
    return results

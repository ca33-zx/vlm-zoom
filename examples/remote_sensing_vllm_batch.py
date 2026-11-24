"""Remote-sensing / 4K satellite Glance-and-Zoom batch example with vLLM.

Demonstrates Stage-1 Glance (compact 1024-token overview) -> Stage-2 Zoom
(high-resolution tiles ONLY inside candidate ROIs) -> cross-tile WBF fusion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_zoom import AdaptiveTiler, AsyncVLLMClient, GeoTransform, ZoomEngine, read_image_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote sensing Glance-and-Zoom on vLLM.")
    parser.add_argument("image", type=Path, help="High-res satellite / aerial image.")
    parser.add_argument(
        "--prompt",
        default=(
            "Detect all storage tanks, bridges, and solar arrays. "
            'Return JSON [{"label": str, "bbox_2d": [x1, y1, x2, y2], "score": float}].'
        ),
    )
    parser.add_argument("--world-file", type=Path, default=None, help="Optional .tfw/.jgw.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()

    # Step 1: Zero-decode header inspection before hitting the GPU server.
    w, h = read_image_size(args.image)
    tiler = AdaptiveTiler(args.model, max_tokens_per_tile=1024, overlap=0.15)
    plan = tiler.plan(w, h)
    print(
        f"Header inspection: {w}x{h} | direct={plan.direct_estimate.visual_tokens} tok "
        f"| adaptive grid={plan.rows}x{plan.cols} ({plan.total_tile_tokens} tok total)"
    )

    # Step 2: Run Glance-and-Zoom asynchronously against vLLM.
    client = AsyncVLLMClient(base_url=args.base_url, model=args.model, max_concurrency=8)
    engine = ZoomEngine(
        client,
        model_family=args.model,
        max_tokens_per_tile=1024,
        overlap=0.15,
        fusion_method="wbf",
    )
    geo_tf = GeoTransform.from_world_file(args.world_file) if args.world_file else None
    result = engine.run(
        args.image,
        args.prompt,
        mode="glance_and_zoom",
        pin_overview=True,
        geo_transform=geo_tf,
    )

    print(
        f"Glance found {len(result.glance_rois)} ROI(s), zoomed into "
        f"{result.token_usage.num_tiles} tile(s), fused {len(result.detections)} detections."
    )
    for det in result.detections:
        print(f"  [{det.score:.2f}] {det.label}: {det.box.to_int_tuple()}")


if __name__ == "__main__":
    main()

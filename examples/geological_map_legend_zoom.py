"""End-to-end geological map interpretation with Pinned Legend + vLLM Prefix Caching.

Launch vLLM first with Automatic Prefix Caching enabled:

    vllm serve Qwen/Qwen3-VL-4B-Instruct \
        --enable-prefix-caching \
        --max-model-len 8192 \
        --limit-mm-per-prompt '{"image": 4}'

Then run this script against either a real scanned geological map + `.pgw`/`.tfw`
world file, or run without arguments to generate a synthetic 4.8K geological map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_zoom import (
    AsyncVLLMClient,
    GeoTransform,
    PixelBox,
    ZoomEngine,
)
from vlm_zoom.viz import make_synthetic_geological_map, render_plan_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Geological map Glance-and-Zoom demo.")
    parser.add_argument("--image", type=Path, default=None, help="Path to geological map image.")
    parser.add_argument("--world-file", type=Path, default=None, help="Optional .tfw/.pgw/.jgw.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--mode", choices=["grid", "glance_and_zoom"], default="grid")
    parser.add_argument("--out-overlay", type=Path, default=Path("out/geology_overlay.png"))
    args = parser.parse_args()

    if args.image is None:
        image, legend_box, _ = make_synthetic_geological_map(width=4800, height=3000)
        map_roi = PixelBox(80.0, 120.0, legend_box.x1 - 40.0, 2880.0)
        geo_tf = GeoTransform(
            pixel_width=5.0,
            col_rotation=0.0,
            row_rotation=0.0,
            pixel_height=-5.0,
            origin_x=500000.0,
            origin_y=3500000.0,
            crs="EPSG:32650",
        )
    else:
        image = args.image
        legend_box = None
        map_roi = None
        geo_tf = GeoTransform.from_world_file(args.world_file) if args.world_file else None

    client = AsyncVLLMClient(
        base_url=args.base_url,
        model=args.model,
        max_concurrency=8,
    )
    engine = ZoomEngine(
        client,
        model_family=args.model,
        max_tokens_per_tile=1024,
        overlap=0.15,
        fusion_method="wbf",
    )

    prompt = (
        "Using the pinned Map Legend, detect all geological fault labels, ore veins, "
        "and stratigraphic unit annotations visible in this tile. "
        'Return a JSON list of {"label": str, "bbox_2d": [x1, y1, x2, y2], "score": float} '
        "in [0, 1000] normalized coordinates."
    )

    result = engine.run(
        image,
        prompt,
        mode=args.mode,
        roi=map_roi,
        legend_box=legend_box,
        pin_overview=True,
        geo_transform=geo_tf,
    )

    print(f"Tiles processed           : {result.token_usage.num_tiles}")
    print(f"Direct full-res tokens    : {result.token_usage.direct_full_res_tokens}")
    print(f"Pinned prefix tokens      : {result.token_usage.pinned_prefix_tokens}")
    print(f"Est. prefix tokens saved  : {result.token_usage.estimated_prefix_tokens_saved}")
    print(f"Server cached tokens      : {result.token_usage.server_cached_tokens}")
    print(f"Fused global detections   : {len(result.detections)}")

    for det, gbox in result.geo_detections:
        print(
            f"  - {det.label:22s} px={det.box.to_int_tuple()} "
            f"geo=({gbox.min_x:.1f}, {gbox.min_y:.1f})..({gbox.max_x:.1f}, {gbox.max_y:.1f})"
        )

    if result.plans:
        args.out_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay = render_plan_overlay(
            image,
            result.plans[0],
            detections=result.detections,
            legend_box=legend_box,
            glance_rois=result.glance_rois,
        )
        overlay.save(args.out_overlay)
        print(f"Saved overlay visualization to {args.out_overlay}")


if __name__ == "__main__":
    main()

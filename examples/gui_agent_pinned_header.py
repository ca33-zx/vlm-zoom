"""4K Desktop GUI Agent & Dense Financial Table extraction with Pinned Table Header.

Demonstrates pinning a 4K table schema / toolbar header in the vLLM prefix cache
so every zoomed tile retains full column context while resolving crisp 11px numbers
and exact click coordinates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_zoom import AsyncVLLMClient, PixelBox, ZoomEngine
from vlm_zoom.viz import make_synthetic_4k_gui_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="4K GUI & Dense Table Pinned-Header demo.")
    parser.add_argument("--image", type=Path, default=None, help="Optional 4K GUI screenshot.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    if args.image is None:
        image, header_box, _ = make_synthetic_4k_gui_dashboard()
        body_roi = PixelBox(header_box.x1, header_box.y2, header_box.x2, float(image.height - 20))
        pinned = {
            "Top Toolbar": PixelBox(0, 0, float(image.width), 74.0),
            "Table Column Header": header_box,
        }
    else:
        image = args.image
        body_roi = None
        pinned = {"Top Header": PixelBox(0, 0, 3840, 180)}

    client = AsyncVLLMClient(base_url=args.base_url, model=args.model, max_concurrency=8)
    engine = ZoomEngine(
        client,
        model_family=args.model,
        max_tokens_per_tile=1024,
        overlap=0.12,
        fusion_method="wbf",
    )

    result = engine.run(
        image,
        prompt=(
            "Using the pinned Table Column Header, locate every row marked ANOMALY "
            "and return the exact bounding box of its 'Review Flag' button as JSON "
            '[{"label": "<JOB_ID> Review Flag", "bbox_2d": [x1, y1, x2, y2], "score": float}].'
        ),
        mode="grid",
        roi=body_roi,
        pinned_boxes=pinned,
    )

    print(f"Tiles processed          : {result.token_usage.num_tiles}")
    print(f"Pinned header tokens     : {result.token_usage.pinned_prefix_tokens}")
    print(f"Prefix KV tokens saved   : {result.token_usage.estimated_prefix_tokens_saved}")
    for det in result.detections:
        cx = int((det.box.x1 + det.box.x2) / 2)
        cy = int((det.box.y1 + det.box.y2) / 2)
        print(f"  CLICK ({cx:4d}, {cy:4d})  {det.label}  box={det.box.to_int_tuple()}")


if __name__ == "__main__":
    main()

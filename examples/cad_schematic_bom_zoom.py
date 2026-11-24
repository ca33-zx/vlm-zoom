"""Engineering CAD / PCB Schematic inspection with Pinned Title Block & BOM Legend.

Pins the right-bottom Title Block & Bill of Materials (BOM) as a lossless PNG prefix
so vLLM computes its KV cache once and reuses it across every high-res schematic tile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_zoom import AsyncVLLMClient, PixelBox, ZoomEngine
from vlm_zoom.viz import make_synthetic_cad_blueprint


def main() -> None:
    parser = argparse.ArgumentParser(description="CAD / PCB schematic Pinned BOM demo.")
    parser.add_argument("--image", type=Path, default=None, help="Optional schematic PNG/TIFF.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    if args.image is None:
        image, title_block, _ = make_synthetic_cad_blueprint()
        pinned = {"Title Block & BOM Legend": title_block}
    else:
        image = args.image
        pinned = {"Title Block": PixelBox(3000, 1700, 4100, 2500)}

    client = AsyncVLLMClient(base_url=args.base_url, model=args.model, max_concurrency=8)
    engine = ZoomEngine(
        client,
        model_family=args.model,
        max_tokens_per_tile=1024,
        overlap=0.15,
        fusion_method="wbf",
    )

    result = engine.run(
        image,
        prompt=(
            "Cross-reference the pinned Title Block & BOM Legend and detect all critical "
            "component designators (resistors, capacitors, test points) in this tile. "
            'Return JSON [{"label": str, "bbox_2d": [x1, y1, x2, y2], "score": float}].'
        ),
        mode="grid",
        pinned_boxes=pinned,
    )

    print(f"Tiles processed          : {result.token_usage.num_tiles}")
    print(f"Prefix KV tokens saved   : {result.token_usage.estimated_prefix_tokens_saved}")
    for det in result.detections:
        print(f"  - {det.label:16s} box={det.box.to_int_tuple()} score={det.score:.2f}")


if __name__ == "__main__":
    main()

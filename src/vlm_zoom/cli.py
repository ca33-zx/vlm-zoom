"""Command-line interface for `vlm-zoom`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import AsyncVLLMClient
from .engine import ZoomEngine
from .geo import GeoTransform
from .tiler import AdaptiveTiler, PixelBox, build_pinned_context, read_image_size
from .viz import (
    make_synthetic_4k_gui_dashboard,
    make_synthetic_cad_blueprint,
    make_synthetic_geological_map,
    render_hero_comparison,
)


def _parse_box(raw: str | None) -> PixelBox | None:
    if not raw:
        return None
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 comma-separated coordinates x1,y1,x2,y2, got {raw!r}.")
    return PixelBox(*parts)


def _parse_pinned_boxes(items: list[str] | None) -> dict[str, PixelBox]:
    parsed: dict[str, PixelBox] = {}
    for raw in items or ():
        if "=" not in raw:
            raise ValueError(f"Expected LABEL=x1,y1,x2,y2 for --pinned-box, got {raw!r}.")
        label, coords = raw.split("=", 1)
        box = _parse_box(coords)
        assert box is not None
        parsed[label.strip()] = box
    return parsed


def cmd_inspect(args: argparse.Namespace) -> int:
    w, h = read_image_size(args.image)
    tiler = AdaptiveTiler(
        args.model,
        max_tokens_per_tile=args.max_tokens_per_tile,
        overlap=args.overlap,
        min_detail_scale=args.min_detail_scale,
        max_tiles=args.max_tiles,
    )
    roi = _parse_box(args.roi)
    plan = tiler.plan(w, h, roi=roi)
    report = {
        "image": str(args.image),
        "dimensions": [w, h],
        "model_family": plan.direct_estimate.model_family,
        "direct_full_res_tokens": plan.direct_estimate.visual_tokens,
        "direct_resized": [plan.direct_estimate.resized_width, plan.direct_estimate.resized_height],
        "direct_min_scale": round(plan.direct_estimate.min_scale, 3),
        "recommended_grid": [plan.rows, plan.cols],
        "num_tiles": len(plan.tiles),
        "overlap": plan.overlap,
        "max_tokens_per_tile": plan.max_tokens_per_tile,
        "total_tile_tokens": plan.total_tile_tokens,
        "min_effective_scale": round(plan.min_effective_scale, 3),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Image           : {args.image} ({w}x{h})")
        print(f"Model family    : {plan.direct_estimate.model_family}")
        print(
            f"Direct input    : {plan.direct_estimate.visual_tokens} visual tokens "
            f"({plan.direct_estimate.resized_width}x{plan.direct_estimate.resized_height}, "
            f"scale={plan.direct_estimate.min_scale:.2f}x)"
        )
        print(
            f"Adaptive grid   : {plan.rows}x{plan.cols} ({len(plan.tiles)} tiles, "
            f"overlap={plan.overlap:.0%}, max {plan.max_tokens_per_tile} tok/tile)"
        )
        print(
            f"Tiled budget    : {plan.total_tile_tokens} total tokens "
            f"(min detail scale={plan.min_effective_scale:.2f}x)"
        )
    return 0


def cmd_tile(args: argparse.Namespace) -> int:
    tiler = AdaptiveTiler(
        args.model,
        max_tokens_per_tile=args.max_tokens_per_tile,
        overlap=args.overlap,
        min_detail_scale=args.min_detail_scale,
        max_tiles=args.max_tiles,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan, tiles = tiler.tile(args.image, roi=_parse_box(args.roi))
    manifest: dict[str, object] = {
        "image": str(args.image),
        "width": plan.image_width,
        "height": plan.image_height,
        "grid": [plan.rows, plan.cols],
        "tiles": [],
    }
    tile_entries: list[dict[str, object]] = []
    for t in tiles:
        fname = f"{t.spec.tile_id}.{args.format.lower()}"
        t.image.save(out_dir / fname)
        tile_entries.append(
            {
                "tile_id": t.spec.tile_id,
                "file": fname,
                "box": list(t.spec.box.to_int_tuple()),
                "sent_size": [t.spec.sent_width, t.spec.sent_height],
                "estimated_tokens": t.spec.estimated_tokens,
            }
        )
    manifest["tiles"] = tile_entries

    legend_box = _parse_box(args.legend_box)
    if legend_box is not None:
        from .tiler import load_image

        ctx = build_pinned_context(
            load_image(args.image),
            box=legend_box,
            model_or_family=args.model,
            max_tokens=args.max_tokens_per_tile,
        )
        ctx.image.save(out_dir / "pinned_legend.png")
        manifest["pinned_legend"] = {
            "file": "pinned_legend.png",
            "box": list(ctx.source_box.to_int_tuple()),
            "estimated_tokens": ctx.estimated_tokens,
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(tiles)} tiles and manifest.json to {out_dir}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    scenario = args.scenario
    tiler = AdaptiveTiler(
        args.model,
        max_tokens_per_tile=args.max_tokens_per_tile,
        overlap=0.15,
        min_detail_scale=0.85,
    )
    if scenario == "gui":
        img, pinned_box, dets = make_synthetic_4k_gui_dashboard(
            width=args.width or 3840, height=args.height or 2160
        )
        body_roi = PixelBox(pinned_box.x1, pinned_box.y2, pinned_box.x2, float(img.height - 20))
        plan = tiler.plan(img.width, img.height, roi=body_roi)
        render_hero_comparison(
            img,
            plan,
            detections=dets,
            legend_box=pinned_box,
            pinned_label="PINNED TABLE HEADER (KV)",
            zoom_callout_box=PixelBox(1620.0, 1110.0, 2460.0, 1438.0),
            scenario_title=(
                "vlm-zoom (4K GUI & Dense Table): Direct Input vs. "
                "Adaptive Tiling + Pinned Table Header"
            ),
            after_banner=(
                f"AFTER: vlm-zoom ({plan.rows}x{plan.cols} tiles, "
                f"<= {plan.max_tokens_per_tile} tok/tile, Pinned Header KV Cache)"
            ),
            left_bullets=[
                "• 11px numbers & anomaly flags turn illegible",
                "• GUI Agent clicks miss small action buttons",
                "• Naive slicing loses table column headers",
                f"• Raw 4K direct input = {plan.direct_estimate.visual_tokens:,} visual tok",
            ],
            right_bullets=[
                "• Pinned Table Header cached ONCE in vLLM KV",
                f"• Detail retention preserved at {plan.min_effective_scale:.2f}x",
                "• Exact pixel coordinates mapped for GUI clicks",
                "• Works on Excel, web tables, IDEs & dashboards",
            ],
            output_path=args.out,
        )
    elif scenario == "cad":
        img, pinned_box, dets = make_synthetic_cad_blueprint(
            width=args.width or 4200, height=args.height or 2600
        )
        plan = tiler.plan(img.width, img.height)
        render_hero_comparison(
            img,
            plan,
            detections=dets,
            legend_box=pinned_box,
            pinned_label="PINNED TITLE BLOCK / BOM (KV)",
            zoom_callout_box=PixelBox(960.0, 580.0, 1520.0, 910.0),
            scenario_title=(
                "vlm-zoom (Engineering CAD & PCB Schematic): Direct Input vs. "
                "Adaptive Tiling + Pinned BOM"
            ),
            after_banner=(
                f"AFTER: vlm-zoom ({plan.rows}x{plan.cols} tiles, "
                f"<= {plan.max_tokens_per_tile} tok/tile, Pinned BOM KV Cache)"
            ),
            left_bullets=[
                "• Resistor/cap values (R14 4.7k, C22) blur out",
                "• Pin labels & net names become unreadable",
                "• Cropping tiles loses BOM & tolerance block",
                f"• Raw 4.2K direct input = {plan.direct_estimate.visual_tokens:,} visual tok",
            ],
            right_bullets=[
                "• Pinned Title Block & BOM reused across tiles",
                f"• Native detail retained at {plan.min_effective_scale:.2f}x",
                "• Lossless PNG prefix keeps crisp schematics",
                "• Cross-tile WBF deduplicates bus annotations",
            ],
            output_path=args.out,
        )
    else:
        w = args.width or 4800
        h = args.height or 3000
        img, legend_box, dets = make_synthetic_geological_map(width=w, height=h)
        map_roi = PixelBox(80.0, 120.0, legend_box.x1 - 40.0, float(h - 120))
        plan = tiler.plan(img.width, img.height, roi=map_roi)
        render_hero_comparison(
            img,
            plan,
            detections=dets,
            legend_box=legend_box,
            zoom_callout_box=dets[0].box.pad(0.8, img.width, img.height),
            direct_max_tokens=1024,
            output_path=args.out,
        )
    print(f"Saved {scenario} visual comparison demo to {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    client = AsyncVLLMClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_concurrency=args.max_concurrency,
    )
    engine = ZoomEngine(
        client,
        model_family=args.model_family or args.model,
        max_tokens_per_tile=args.max_tokens_per_tile,
        overlap=args.overlap,
        coord_space=args.coord_space,
        fusion_method=args.fusion,
    )
    geo_tf = GeoTransform.from_world_file(args.world_file) if args.world_file else None
    result = engine.run(
        args.image,
        args.prompt,
        mode=args.mode,
        roi=_parse_box(args.roi),
        legend_box=_parse_box(args.legend_box),
        pinned_boxes=_parse_pinned_boxes(args.pinned_box) or None,
        pin_overview=args.pin_overview,
        geo_transform=geo_tf,
    )
    payload = {
        "mode": result.mode,
        "image_size": list(result.image_size),
        "num_tiles": result.token_usage.num_tiles,
        "token_usage": {
            "direct_full_res_tokens": result.token_usage.direct_full_res_tokens,
            "glance_tokens": result.token_usage.glance_tokens,
            "pinned_prefix_tokens": result.token_usage.pinned_prefix_tokens,
            "total_tile_tokens": result.token_usage.total_tile_tokens,
            "estimated_prefix_tokens_saved": result.token_usage.estimated_prefix_tokens_saved,
            "server_prompt_tokens": result.token_usage.server_prompt_tokens,
            "server_cached_tokens": result.token_usage.server_cached_tokens,
        },
        "detections": [
            {
                "label": d.label,
                "box": [
                    round(d.box.x1, 1),
                    round(d.box.y1, 1),
                    round(d.box.x2, 1),
                    round(d.box.y2, 1),
                ],
                "score": round(d.score, 4),
                "source_tile": d.source_tile,
            }
            for d in result.detections
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm-zoom",
        description="High-resolution adaptive tiling & Glance-and-Zoom inference for VLMs on vLLM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_insp = sub.add_parser("inspect", help="Header-only visual token & tiling budget inspection.")
    p_insp.add_argument("image", type=str)
    p_insp.add_argument("--model", default="qwen3_vl")
    p_insp.add_argument("--max-tokens-per-tile", type=int, default=1024)
    p_insp.add_argument("--overlap", type=float, default=0.15)
    p_insp.add_argument("--min-detail-scale", type=float, default=0.85)
    p_insp.add_argument("--max-tiles", type=int, default=36)
    p_insp.add_argument("--roi", type=str, default=None, help="Optional x1,y1,x2,y2 ROI box.")
    p_insp.add_argument("--json", action="store_true")
    p_insp.set_defaults(func=cmd_inspect)

    p_tile = sub.add_parser("tile", help="Export budget-sized tiles and manifest.json.")
    p_tile.add_argument("image", type=str)
    p_tile.add_argument("--out-dir", required=True)
    p_tile.add_argument("--model", default="qwen3_vl")
    p_tile.add_argument("--max-tokens-per-tile", type=int, default=1024)
    p_tile.add_argument("--overlap", type=float, default=0.15)
    p_tile.add_argument("--min-detail-scale", type=float, default=0.85)
    p_tile.add_argument("--max-tiles", type=int, default=36)
    p_tile.add_argument("--roi", type=str, default=None)
    p_tile.add_argument("--legend-box", type=str, default=None)
    p_tile.add_argument("--format", choices=["PNG", "JPEG"], default="PNG")
    p_tile.set_defaults(func=cmd_tile)

    p_demo = sub.add_parser("demo", help="Generate a visual before/after comparison PNG.")
    p_demo.add_argument("--scenario", choices=["geology", "gui", "cad"], default="geology")
    p_demo.add_argument("--out", default="assets/geological_map_demo.png")
    p_demo.add_argument("--model", default="qwen3_vl")
    p_demo.add_argument("--max-tokens-per-tile", type=int, default=1024)
    p_demo.add_argument("--width", type=int, default=0)
    p_demo.add_argument("--height", type=int, default=0)
    p_demo.set_defaults(func=cmd_demo)

    p_run = sub.add_parser("run", help="Run concurrent Glance-and-Zoom or Grid inference on vLLM.")
    p_run.add_argument("image", type=str)
    p_run.add_argument("--prompt", required=True)
    p_run.add_argument("--base-url", default="http://localhost:8000/v1")
    p_run.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p_run.add_argument("--model-family", default=None)
    p_run.add_argument("--api-key", default="EMPTY")
    p_run.add_argument("--mode", choices=["grid", "glance_and_zoom"], default="glance_and_zoom")
    p_run.add_argument("--max-tokens-per-tile", type=int, default=1024)
    p_run.add_argument("--overlap", type=float, default=0.15)
    p_run.add_argument("--max-concurrency", type=int, default=8)
    p_run.add_argument("--roi", type=str, default=None)
    p_run.add_argument("--legend-box", type=str, default=None)
    p_run.add_argument(
        "--pinned-box",
        action="append",
        default=None,
        help="Repeatable LABEL=x1,y1,x2,y2 region to pin & KV-cache across tiles.",
    )
    p_run.add_argument("--pin-overview", action="store_true")
    p_run.add_argument("--coord-space", choices=["norm1000", "norm1", "pixel"], default="norm1000")
    p_run.add_argument("--fusion", choices=["wbf", "nms"], default="wbf")
    p_run.add_argument("--world-file", type=str, default=None)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

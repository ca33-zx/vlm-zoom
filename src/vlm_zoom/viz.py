"""Visualization utilities for tile grids, fused detections, and side-by-side comparisons."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .budget import estimate_visual_tokens
from .fusion import Detection
from .tiler import PixelBox, TilingPlan, load_image


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    bg: tuple[int, ...],
    fg: tuple[int, ...],
    max_x: int,
    pad_x: int = 6,
    pad_y: int = 3,
) -> None:
    tw, th = _text_size(draw, text, font)
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    px = max(2, min(x, max_x - box_w - 2))
    py = max(2, y)
    draw.rectangle((px, py, px + box_w, py + box_h), fill=bg)
    draw.text((px + pad_x, py + pad_y - 1), text, fill=fg, font=font)


def render_plan_overlay(
    source: str | Path | Image.Image,
    plan: TilingPlan,
    *,
    detections: list[Detection] | None = None,
    legend_box: PixelBox | None = None,
    glance_rois: list[PixelBox] | None = None,
    max_preview_width: int = 1600,
) -> Image.Image:
    """Render tile boundaries, pinned legend box, Glance ROIs, and fused detections."""
    img = load_image(source)
    scale = min(1.0, max_preview_width / max(img.width, 1))
    pw, ph = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    canvas = img.resize((pw, ph), Image.Resampling.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    tile_font = _load_font(max(11, int(pw / 82)))
    det_font = _load_font(max(11, int(pw / 78)))

    def _s(box: PixelBox) -> tuple[int, int, int, int]:
        return (
            int(box.x1 * scale),
            int(box.y1 * scale),
            int(box.x2 * scale),
            int(box.y2 * scale),
        )

    # 1. Tile grid (cyan translucent borders)
    for t in plan.tiles:
        coords = _s(t.box)
        draw.rectangle(coords, outline=(0, 195, 245, 225), width=2)
        _draw_pill(
            draw,
            coords[0] + 3,
            coords[1] + 3,
            t.tile_id,
            font=tile_font,
            bg=(12, 28, 48, 195),
            fg=(145, 240, 255, 255),
            max_x=pw,
            pad_x=5,
            pad_y=2,
        )

    # 2. Pinned legend / context box (gold)
    if legend_box is not None:
        lcoords = _s(legend_box)
        draw.rectangle(lcoords, fill=(255, 195, 0, 28), outline=(245, 180, 0, 255), width=3)
        _draw_pill(
            draw,
            lcoords[0],
            max(2, lcoords[1] - 20),
            "PINNED LEGEND (KV)",
            font=tile_font,
            bg=(245, 185, 0, 245),
            fg=(18, 22, 30, 255),
            max_x=pw,
            pad_x=6,
            pad_y=2,
        )

    # 3. Stage-1 Glance ROIs (magenta)
    for idx, rbox in enumerate(glance_rois or ()):
        rcoords = _s(rbox)
        draw.rectangle(rcoords, outline=(255, 70, 180, 240), width=3)
        _draw_pill(
            draw,
            rcoords[0] + 4,
            rcoords[1] + 4,
            f"ROI #{idx}",
            font=tile_font,
            bg=(195, 30, 130, 225),
            fg=(255, 255, 255, 255),
            max_x=pw,
        )

    # 4. Fused global detections (bright emerald green)
    for det in detections or ():
        dcoords = _s(det.box)
        draw.rectangle(dcoords, fill=(16, 215, 120, 42), outline=(14, 195, 105, 255), width=2)
        tag = f"{det.label} ({det.score:.2f})"
        _draw_pill(
            draw,
            dcoords[0],
            max(2, dcoords[1] - 20),
            tag,
            font=det_font,
            bg=(12, 140, 78, 235),
            fg=(255, 255, 255, 255),
            max_x=pw,
            pad_x=5,
            pad_y=2,
        )

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def make_synthetic_geological_map(
    width: int = 4800,
    height: int = 3000,
) -> tuple[Image.Image, PixelBox, list[Detection]]:
    """Generate a synthetic 4.8K regional geological map with tiny stratigraphic codes,
    fault lines, strike/dip symbols, and a right-hand Legend panel for demos & tests.

    Returns `(image, legend_box, ground_truth_detections)`.
    """
    img = Image.new("RGB", (width, height), (247, 245, 238))
    draw = ImageDraw.Draw(img)

    map_w = int(width * 0.78)
    legend_x1 = map_w + 40
    legend_box = PixelBox(float(legend_x1), 120.0, float(width - 50), float(height - 120))

    # Realistic colored geological formations (polygons across map body)
    strata = [
        (
            "Q4",
            "Quaternary Alluvium",
            (245, 236, 178),
            [(80, 120), (map_w, 120), (map_w, 720), (2200, 920), (80, 620)],
        ),
        (
            "K1",
            "Lower Cretaceous Sandstone",
            (184, 224, 152),
            [(80, 620), (2200, 920), (map_w, 720), (map_w, 1520), (1950, 1720), (80, 1380)],
        ),
        (
            "J3",
            "Upper Jurassic Shale",
            (152, 198, 236),
            [(80, 1380), (1950, 1720), (map_w, 1520), (map_w, 2260), (1600, 2400), (80, 2120)],
        ),
        (
            "T2",
            "Middle Triassic Limestone",
            (214, 172, 224),
            [
                (80, 2120),
                (1600, 2400),
                (map_w, 2260),
                (map_w, height - 120),
                (80, height - 120),
            ],
        ),
        (
            "gamma5",
            "Yanshanian Granite Intrusion",
            (242, 148, 148),
            [(2350, 1250), (3050, 1180), (3280, 1650), (2950, 2060), (2320, 1920)],
        ),
    ]
    for _, _, color, poly in strata:
        draw.polygon(poly, fill=color, outline=(70, 70, 70))

    # Subtle coordinate grid lines inside map frame
    for gx in range(280, map_w, 400):
        draw.line([(gx, 120), (gx, height - 120)], fill=(140, 140, 130), width=1)
    for gy in range(320, height - 120, 400):
        draw.line([(80, gy), (map_w, gy)], fill=(140, 140, 130), width=1)

    # Major fault lines F1 and F2 (crimson)
    draw.line([(450, 2650), (2600, 1400), (3550, 320)], fill=(195, 25, 35), width=8)
    draw.line([(1050, 320), (2100, 1580), (3350, 2600)], fill=(195, 25, 35), width=6)

    # Map border & title
    draw.rectangle((80, 120, map_w, height - 120), outline=(30, 30, 30), width=5)
    title_font = _load_font(42)
    small_font = _load_font(22)
    tiny_font = _load_font(18)
    draw.text(
        (90, 45),
        "REGIONAL GEOLOGICAL MAP (1:50,000)  |  4800 x 3000 px",
        fill=(20, 25, 35),
        font=title_font,
    )

    # High-value tiny geological annotations (well spaced so labels never collide)
    annotations = [
        ("Fault F1 (NE 42 deg)", (2220, 1490), (175, 15, 25)),
        ("J3 (Shale, 28 deg / 135)", (1200, 1880), (15, 45, 95)),
        ("K1 (Sandstone, 19 deg / 110)", (1080, 990), (20, 80, 20)),
        ("gamma5 (Granite)", (2580, 1810), (130, 20, 20)),
        ("T2 (Limestone, 34 deg / 152)", (2100, 2540), (75, 25, 95)),
        ("Q4 (Alluvium)", (2860, 450), (95, 75, 10)),
        ("Ore Vein #3 (Cu-Au)", (2840, 1290), (170, 20, 20)),
    ]
    gt_detections: list[Detection] = []
    for text, (cx, cy), color in annotations:
        tw, th = _text_size(draw, text, small_font)
        box = PixelBox(cx - 18, cy - 10, cx + tw + 30, cy + th + 20)
        # Subtle pale halo behind tiny text so it remains crisp against fault lines
        draw.rectangle(
            (cx + 12, cy - 2, cx + tw + 24, cy + th + 10),
            fill=(255, 252, 245),
            outline=(190, 185, 175),
            width=1,
        )
        draw.ellipse(
            (cx - 8, cy + 4, cx + 8, cy + 20), fill=color, outline=(255, 255, 255), width=2
        )
        draw.text((cx + 18, cy + 2), text, fill=color, font=small_font)
        gt_detections.append(
            Detection(label=text.split(" (")[0], box=box, score=0.96, source_tile="gt")
        )

    # Crisp Legend panel on the right
    lx1, ly1, lx2, ly2 = legend_box.to_int_tuple()
    draw.rectangle((lx1, ly1, lx2, ly2), fill=(255, 255, 255), outline=(35, 40, 50), width=4)
    draw.text((lx1 + 35, ly1 + 35), "LEGEND / STRATIGRAPHY", fill=(20, 25, 35), font=_load_font(30))
    draw.line([(lx1 + 30, ly1 + 85), (lx2 - 30, ly1 + 85)], fill=(80, 80, 80), width=2)

    cur_y = ly1 + 120
    for code, desc, color, _ in strata:
        draw.rectangle(
            (lx1 + 35, cur_y, lx1 + 145, cur_y + 62),
            fill=color,
            outline=(40, 40, 40),
            width=2,
        )
        draw.text((lx1 + 58, cur_y + 18), code, fill=(20, 20, 20), font=small_font)
        draw.text((lx1 + 165, cur_y + 18), desc, fill=(30, 35, 45), font=small_font)
        cur_y += 105

    draw.line([(lx1 + 35, cur_y + 25), (lx1 + 145, cur_y + 25)], fill=(195, 25, 35), width=8)
    draw.text((lx1 + 165, cur_y + 12), "Major Fault (F1 / F2)", fill=(160, 20, 30), font=small_font)
    cur_y += 95
    draw.text(
        (lx1 + 35, cur_y),
        "Scale 1:50,000  |  CRS: WGS84 / UTM",
        fill=(80, 85, 95),
        font=tiny_font,
    )

    return img, legend_box, gt_detections


def render_hero_comparison(
    source: str | Path | Image.Image,
    plan: TilingPlan,
    *,
    detections: list[Detection],
    legend_box: PixelBox | None = None,
    zoom_callout_box: PixelBox | None = None,
    direct_max_tokens: int = 1024,
    output_path: str | Path | None = None,
) -> Image.Image:
    """Render a publication-ready side-by-side comparison image for README & demos.

    Left panel: what a VLM actually sees when forced to downscale a 4.8K/8K map
    directly into `direct_max_tokens` (tiny annotations blurred beyond recognition).
    Right panel: `vlm-zoom` adaptive grid + pinned legend (vLLM prefix-cached) +
    fused global detections + crisp 1:1 zoomed crop inset.
    """
    orig = load_image(source)
    direct_est = estimate_visual_tokens(
        orig.width,
        orig.height,
        plan.direct_estimate.model_family,
        max_tokens=direct_max_tokens,
    )
    # Simulate exact visual degradation of direct VLM downscaling
    degraded = orig.resize(
        (direct_est.resized_width, direct_est.resized_height),
        Image.Resampling.BILINEAR,
    )

    panel_w = 900
    panel_h = int(panel_w * orig.height / orig.width)
    banner_h = 36
    header_h = 84 + banner_h + 10
    footer_h = 212
    gap = 28
    pad = 28
    total_w = pad * 2 + panel_w * 2 + gap
    total_h = header_h + panel_h + footer_h + pad

    canvas = Image.new("RGB", (total_w, total_h), (15, 20, 30))
    draw = ImageDraw.Draw(canvas)
    h_font = _load_font(25)
    sub_font = _load_font(17)
    tag_font = _load_font(16)

    # Top title banner
    draw.text(
        (pad, 16),
        (
            "vlm-zoom: Direct Downscaled VLM Input vs. "
            "Adaptive Tiling + Pinned Legend (vLLM Prefix Caching)"
        ),
        fill=(240, 246, 255),
        font=h_font,
    )
    draw.text(
        (pad, 50),
        f"Source: {orig.width}x{orig.height} px  |  Model: {plan.direct_estimate.model_family}  |  "
        f"Direct scale: {direct_est.min_scale:.2f}x (blurred)  vs.  "
        f"vlm-zoom scale: {plan.min_effective_scale:.2f}x (native)",
        fill=(145, 170, 200),
        font=sub_font,
    )

    rx = pad + panel_w + gap
    by = 84

    # Left column banner (above map frame so nothing covers the map content!)
    draw.rectangle((pad, by, pad + panel_w, by + banner_h), fill=(175, 35, 45))
    draw.text(
        (pad + 14, by + 9),
        f"BEFORE: Direct Input ({direct_est.resized_width}x{direct_est.resized_height}, "
        f"{direct_est.visual_tokens} visual tok)  ->  Small Symbols Unreadable",
        fill=(255, 255, 255),
        font=tag_font,
    )

    # Right column banner
    draw.rectangle((rx, by, rx + panel_w, by + banner_h), fill=(16, 138, 82))
    draw.text(
        (rx + 14, by + 9),
        f"AFTER: vlm-zoom ({plan.rows}x{plan.cols} tiles, <= {plan.max_tokens_per_tile} tok/tile, "
        f"Pinned Legend KV Cache + WBF)",
        fill=(255, 255, 255),
        font=tag_font,
    )

    # Left panel: Direct downscaled VLM view (NEAREST upscale reveals actual pixelation)
    left_img = degraded.resize((panel_w, panel_h), Image.Resampling.NEAREST)
    canvas.paste(left_img, (pad, header_h))
    draw.rectangle(
        (pad, header_h, pad + panel_w, header_h + panel_h), outline=(235, 85, 85), width=3
    )

    # Right panel: vlm-zoom overlay
    right_overlay = render_plan_overlay(
        orig,
        plan,
        detections=detections,
        legend_box=legend_box,
        max_preview_width=panel_w,
    ).resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    canvas.paste(right_overlay, (rx, header_h))
    draw.rectangle(
        (rx, header_h, rx + panel_w, header_h + panel_h), outline=(25, 210, 125), width=3
    )

    # Bottom detail callout strip: zoomed-in comparison of the exact same annotation
    cbox = zoom_callout_box or (
        detections[0].box.pad(0.6, orig.width, orig.height)
        if detections
        else PixelBox(orig.width * 0.4, orig.height * 0.4, orig.width * 0.55, orig.height * 0.55)
    )
    fy = header_h + panel_h + 18
    crop_h = footer_h - 48
    crop_w = 420

    # Left zoomed-in degraded crop
    sx = direct_est.resized_width / orig.width
    sy = direct_est.resized_height / orig.height
    deg_crop = degraded.crop(
        (
            int(cbox.x1 * sx),
            int(cbox.y1 * sy),
            max(int(cbox.x1 * sx) + 2, int(cbox.x2 * sx)),
            max(int(cbox.y1 * sy) + 2, int(cbox.y2 * sy)),
        )
    ).resize((crop_w, crop_h), Image.Resampling.NEAREST)
    canvas.paste(deg_crop, (pad, fy + 26))
    draw.rectangle((pad, fy + 26, pad + crop_w, fy + 26 + crop_h), outline=(235, 85, 85), width=2)
    draw.text(
        (pad, fy + 2),
        "Zoomed Crop (Direct VLM sees pixel mush):",
        fill=(245, 130, 130),
        font=tag_font,
    )

    # Left-bottom bottleneck summary card (symmetric to right-bottom card)
    lcard_x = pad + crop_w + 24
    draw.text(
        (lcard_x, fy + 28),
        "Why Direct VLM Fails on 4K-8K Maps:",
        fill=(255, 135, 135),
        font=sub_font,
    )
    left_bullets = [
        f"• Effective scale collapses to {direct_est.min_scale:.2f}x",
        "• Stratigraphic codes & dip numbers blur out",
        "• Map body + Legend compete for 1000 tokens",
        f"• Raw full-res needs {plan.direct_estimate.visual_tokens:,} visual tokens",
    ]
    for idx, line in enumerate(left_bullets):
        draw.text((lcard_x, fy + 58 + idx * 28), line, fill=(210, 215, 225), font=tag_font)

    # Right zoomed-in native tile crop
    nat_crop = orig.crop(cbox.to_int_tuple()).resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    canvas.paste(nat_crop, (rx, fy + 26))
    draw.rectangle((rx, fy + 26, rx + crop_w, fy + 26 + crop_h), outline=(25, 210, 125), width=2)
    draw.text(
        (rx, fy + 2),
        "Zoomed Crop (vlm-zoom tile preserves crisp text):",
        fill=(95, 235, 165),
        font=tag_font,
    )

    # Right-bottom summary card
    rcard_x = rx + crop_w + 24
    draw.text((rcard_x, fy + 28), "Key Engineering Wins:", fill=(255, 215, 90), font=sub_font)
    right_bullets = [
        f"• Exact {plan.direct_estimate.model_family} smart_resize token planner",
        f"• Detail retention: {direct_est.min_scale:.2f}x -> {plan.min_effective_scale:.2f}x",
        "• Pinned Legend prefix -> vLLM KV cache reuse",
        "• Cross-tile WBF/NMS + World-File GeoTransform",
    ]
    for idx, line in enumerate(right_bullets):
        draw.text((rcard_x, fy + 58 + idx * 28), line, fill=(215, 228, 245), font=tag_font)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    return canvas

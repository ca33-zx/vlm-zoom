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


def _load_mono_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return _load_font(size)


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
    pinned_label: str = "PINNED LEGEND (KV)",
    glance_rois: list[PixelBox] | None = None,
    max_preview_width: int = 1600,
) -> Image.Image:
    """Render tile boundaries, pinned context box, Glance ROIs, and fused detections."""
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

    # 2. Pinned legend / header / title-block box (gold)
    if legend_box is not None:
        lcoords = _s(legend_box)
        draw.rectangle(lcoords, fill=(255, 195, 0, 28), outline=(245, 180, 0, 255), width=3)
        pill_y = lcoords[1] - 20 if lcoords[1] >= 24 else lcoords[3] + 4
        _draw_pill(
            draw,
            lcoords[0],
            max(2, pill_y),
            pinned_label,
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
        draw.rectangle(rcoords, outline=(255, 70, 180, 245), width=4)
        pill_y = rcoords[1] - 22 if rcoords[1] >= 26 else rcoords[1] + 4
        _draw_pill(
            draw,
            rcoords[0] + 4,
            pill_y,
            f"STAGE-1 GLANCE ROI #{idx}",
            font=tile_font,
            bg=(205, 30, 135, 235),
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
    """
    img = Image.new("RGB", (width, height), (247, 245, 238))
    draw = ImageDraw.Draw(img)

    map_w = int(width * 0.78)
    legend_x1 = map_w + 40
    legend_box = PixelBox(float(legend_x1), 120.0, float(width - 50), float(height - 120))

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

    for gx in range(280, map_w, 400):
        draw.line([(gx, 120), (gx, height - 120)], fill=(140, 140, 130), width=1)
    for gy in range(320, height - 120, 400):
        draw.line([(80, gy), (map_w, gy)], fill=(140, 140, 130), width=1)

    draw.line([(450, 2650), (2600, 1400), (3550, 320)], fill=(195, 25, 35), width=8)
    draw.line([(1050, 320), (2100, 1580), (3350, 2600)], fill=(195, 25, 35), width=6)

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


def make_synthetic_4k_gui_dashboard(
    width: int = 3840,
    height: int = 2160,
) -> tuple[Image.Image, PixelBox, list[Detection]]:
    """Generate a realistic 4K desktop GUI / dense financial analytics table screenshot.

    Returns `(image, pinned_header_box, target_ui_detections)` to showcase GUI Agents
    and dense table OCR with a Pinned Table Header cached via vLLM Prefix Caching.
    """
    img = Image.new("RGB", (width, height), (18, 22, 30))
    draw = ImageDraw.Draw(img)
    title_font = _load_font(26)
    ui_font = _load_font(19)
    mono_font = _load_mono_font(18)
    small_mono = _load_mono_font(16)

    # Top OS / IDE toolbar
    draw.rectangle((0, 0, width, 74), fill=(26, 32, 44))
    draw.line([(0, 74), (width, 74)], fill=(55, 65, 85), width=2)
    for idx, col in enumerate(((240, 90, 85), (245, 190, 65), (50, 200, 100))):
        draw.ellipse((24 + idx * 28, 26, 42 + idx * 28, 44), fill=col)
    draw.text(
        (130, 22),
        "FinOps Cluster Console  |  Q4 Production Ledger & GPU Quota Table (3840x2160)",
        fill=(230, 238, 250),
        font=title_font,
    )
    # Action buttons in top-right toolbar
    buttons = [
        ("Sync", 3220, (52, 64, 88)),
        ("Export CSV", 3360, (52, 64, 88)),
        ("Approve Batch", 3560, (24, 145, 92)),
    ]
    for label, bx, bg in buttons:
        bw = max(110, len(label) * 14 + 34)
        draw.rectangle((bx, 16, bx + bw, 58), fill=bg, outline=(95, 115, 145), width=1)
        draw.text((bx + 16, 26), label, fill=(245, 250, 255), font=ui_font)

    # Left navigation sidebar
    sidebar_w = 380
    draw.rectangle((0, 74, sidebar_w, height), fill=(22, 27, 37))
    draw.line([(sidebar_w, 74), (sidebar_w, height)], fill=(50, 60, 78), width=2)
    nav_items = [
        "Overview",
        "GPU Inference Pools",
        "Prefix Cache Hit Rate",
        "Ledger Audit (Active)",
        "Alerts & Anomalies (3)",
        "Billing & Cost Centers",
        "API Keys & Quotas",
        "Cluster Telemetry",
    ]
    for i, item in enumerate(nav_items):
        ny = 120 + i * 74
        if "Active" in item:
            draw.rectangle((16, ny - 10, sidebar_w - 16, ny + 44), fill=(35, 68, 118))
        draw.text((38, ny + 4), item, fill=(215, 226, 242), font=ui_font)

    # Pinned Table Column Header Bar (critical for dense multi-row vertical tiling!)
    table_x1 = sidebar_w + 32
    table_x2 = width - 40
    header_y1 = 104
    header_y2 = 210
    pinned_header = PixelBox(float(table_x1), float(header_y1), float(table_x2), float(header_y2))
    draw.rectangle(
        (table_x1, header_y1, table_x2, header_y2),
        fill=(30, 42, 62),
        outline=(85, 110, 150),
        width=2,
    )

    cols = [
        ("JOB_ID", 0),
        ("REGION", 260),
        ("MODEL_ENDPOINT", 520),
        ("BATCH_TOK", 1040),
        ("PREFIX_HIT%", 1320),
        ("TTFT_MS", 1620),
        ("UNIT_COST_USD", 1880),
        ("STATUS", 2240),
        ("ACTION", 2620),
    ]
    draw.text(
        (table_x1 + 20, header_y1 + 14),
        "PINNED TABLE SCHEMA (must remain visible across every zoomed row tile)",
        fill=(145, 185, 235),
        font=small_mono,
    )
    for cname, dx in cols:
        draw.text((table_x1 + 20 + dx, header_y1 + 58), cname, fill=(250, 215, 110), font=mono_font)

    # Dense 26-row table with tiny actionable cells and 3 highlighted anomalies
    regions = ["us-west-2", "eu-central-1", "ap-northeast-1", "us-east-1"]
    models = [
        "Qwen3-VL-4B-vLLM",
        "Qwen2.5-VL-7B-vLLM",
        "Qwen3-VL-32B-FP8",
        "InternVL3-8B-vLLM",
    ]
    detections: list[Detection] = []
    row_h = 70
    for r in range(26):
        ry = header_y2 + 8 + r * row_h
        if ry + row_h > height - 20:
            break
        bg = (24, 30, 41) if r % 2 == 0 else (20, 25, 35)
        is_alert = r in (6, 14, 21)
        if is_alert:
            bg = (58, 26, 34)
        draw.rectangle(
            (table_x1, ry, table_x2, ry + row_h - 4), fill=bg, outline=(42, 50, 66), width=1
        )

        job_id = f"INV-{20260100 + r}"
        region = regions[r % len(regions)]
        model = models[r % len(models)]
        tokens = f"{(1420 + r * 317):,}"
        hit = "12.4% (!)" if is_alert else f"{84.5 + (r % 12):.1f}%"
        ttft = "1,842 ms" if is_alert else f"{148 + (r * 13) % 90} ms"
        cost = "$184.90" if r == 14 else f"${12.40 + r * 1.35:.2f}"
        status = "ANOMALY" if is_alert else "HEALTHY"
        action = "Review Flag" if is_alert else "Inspect"

        vals = [job_id, region, model, tokens, hit, ttft, cost, status, action]
        for (cname, dx), val in zip(cols, vals, strict=True):
            color = (
                (255, 120, 125)
                if (is_alert and cname in ("PREFIX_HIT%", "TTFT_MS", "STATUS"))
                else (215, 224, 238)
            )
            if cname == "ACTION":
                bx1 = table_x1 + 16 + dx
                bx2 = bx1 + 175
                draw.rectangle(
                    (bx1, ry + 12, bx2, ry + row_h - 16),
                    fill=(185, 45, 65) if is_alert else (42, 58, 84),
                    outline=(240, 130, 140) if is_alert else (85, 105, 138),
                    width=1,
                )
                draw.text((bx1 + 18, ry + 22), val, fill=(255, 255, 255), font=small_mono)
                if is_alert:
                    detections.append(
                        Detection(
                            label=f"{job_id} {action}",
                            box=PixelBox(
                                float(bx1), float(ry + 8), float(bx2), float(ry + row_h - 12)
                            ),
                            score=0.97,
                            source_tile="gt",
                        )
                    )
            else:
                draw.text((table_x1 + 20 + dx, ry + 22), val, fill=color, font=small_mono)

    return img, pinned_header, detections


def make_synthetic_cad_blueprint(
    width: int = 4200,
    height: int = 2600,
) -> tuple[Image.Image, PixelBox, list[Detection]]:
    """Generate a 4.2K dark-blueprint engineering PCB schematic with tiny component labels
    and a right-bottom Pinned Title Block & BOM Legend.
    """
    img = Image.new("RGB", (width, height), (14, 34, 62))
    draw = ImageDraw.Draw(img)
    title_font = _load_mono_font(34)
    label_font = _load_mono_font(20)
    tiny_font = _load_mono_font(16)

    # Blueprint grid
    for gx in range(80, width - 80, 120):
        draw.line([(gx, 80), (gx, height - 80)], fill=(24, 54, 92), width=1)
    for gy in range(80, height - 80, 120):
        draw.line([(80, gy), (width - 80, gy)], fill=(24, 54, 92), width=1)

    # Outer engineering sheet frame
    draw.rectangle((60, 60, width - 60, height - 60), outline=(185, 225, 255), width=4)
    draw.rectangle((76, 76, width - 76, height - 76), outline=(120, 175, 225), width=2)

    # Main MCU IC U1 + Power Regulator U2 + Sensor Interface U3
    ics = [
        ("U1: STM32H743VIT6 (ARM Cortex-M7)", (1300, 760, 2180, 1540)),
        ("U2: TPS54331 (24V -> 3.3V BUCK)", (320, 520, 940, 1080)),
        ("U3: ADS1256 (24-BIT SEISMIC ADC)", (2580, 520, 3260, 1140)),
        ("J4: RS-485 ISOLATED BUS", (2580, 1460, 3180, 1840)),
    ]
    for name, (x1, y1, x2, y2) in ics:
        draw.rectangle((x1, y1, x2, y2), fill=(20, 52, 92), outline=(220, 240, 255), width=3)
        draw.text((x1 + 22, y1 + 22), name, fill=(240, 250, 255), font=label_font)
        # Draw IC pins along left/right edges
        for py in range(y1 + 90, y2 - 40, 52):
            draw.line([(x1 - 45, py), (x1, py)], fill=(150, 215, 255), width=3)
            draw.line([(x2, py), (x2 + 45, py)], fill=(150, 215, 255), width=3)

    # Schematic bus wires
    buses = [
        [(985, 680), (1255, 680), (1255, 900)],
        [(985, 840), (1180, 840), (1180, 1060), (1255, 1060)],
        [(2225, 860), (2535, 860)],
        [(2225, 964), (2535, 964)],
        [(2225, 1068), (2440, 1068), (2440, 1600), (2535, 1600)],
        [(620, 1080), (620, 1780), (1850, 1780), (1850, 1540)],
    ]
    for path in buses:
        draw.line(path, fill=(95, 225, 255), width=4)

    # Tiny critical engineering callouts (resistors, test points, impedance notes)
    parts = [
        ("R14 4.7k (0603, 1%)", (1020, 625)),
        ("C22 100nF X7R (DECOUPLE)", (1020, 785)),
        ("TP7 SPI1_SCLK (24 MHz)", (2250, 805)),
        ("R31 120R TERM (DIFF 100 OHM)", (2230, 1545)),
        ("FB2 BLM18PG (AGND BRIDGE)", (1180, 1725)),
        ("C08 22uF 35V (BUCK OUT)", (420, 1180)),
    ]
    detections: list[Detection] = []
    for label, (px, py) in parts:
        tw, th = _text_size(draw, label, tiny_font)
        draw.rectangle(
            (px - 8, py - 6, px + tw + 14, py + th + 12),
            fill=(16, 44, 78),
            outline=(255, 215, 90),
            width=2,
        )
        draw.text((px + 3, py + 2), label, fill=(255, 232, 130), font=tiny_font)
        detections.append(
            Detection(
                label=label.split(" ")[0],
                box=PixelBox(
                    float(px - 10), float(py - 8), float(px + tw + 16), float(py + th + 14)
                ),
                score=0.95,
                source_tile="gt",
            )
        )

    # Pinned Right-Bottom Title Block & BOM Legend
    tb = PixelBox(3220.0, 1740.0, float(width - 90), float(height - 90))
    tx1, ty1, tx2, ty2 = tb.to_int_tuple()
    draw.rectangle((tx1, ty1, tx2, ty2), fill=(18, 42, 74), outline=(245, 210, 95), width=4)
    draw.text(
        (tx1 + 24, ty1 + 24), "TITLE BLOCK & BOM LEGEND", fill=(255, 220, 105), font=title_font
    )
    draw.line([(tx1 + 20, ty1 + 78), (tx2 - 20, ty1 + 78)], fill=(150, 195, 240), width=2)
    bom_lines = [
        "DWG NO : GEO-DAQ-REV3.2  |  SHEET 1 OF 4",
        "NETS   : CYAN=DIGITAL BUS  GOLD=CRITICAL BOM",
        "R14    : 4.7k Ohm 1% 0603 Pull-Up (3.3V_DIG)",
        "C22    : 100nF 50V X7R MLCC Decoupling Cap",
        "R31    : 120 Ohm RS-485 Differential Terminator",
        "TP7    : 24 MHz SPI Clock Test Point (50 Ohm)",
        "TOL    : +/-0.05 mm  |  4-LAYER FR4 1.6mm",
    ]
    for i, line in enumerate(bom_lines):
        draw.text((tx1 + 26, ty1 + 104 + i * 74), line, fill=(225, 240, 255), font=label_font)

    return img, tb, detections


def render_hero_comparison(
    source: str | Path | Image.Image,
    plan: TilingPlan,
    *,
    detections: list[Detection],
    legend_box: PixelBox | None = None,
    pinned_label: str = "PINNED LEGEND (KV)",
    glance_rois: list[PixelBox] | None = None,
    zoom_callout_box: PixelBox | None = None,
    direct_max_tokens: int = 1024,
    scenario_title: str | None = None,
    after_banner: str | None = None,
    zoom_caption_left: str = "Zoomed Crop (Direct VLM sees pixel mush):",
    zoom_caption_right: str = "Zoomed Crop (vlm-zoom tile preserves crisp detail):",
    left_bullets: list[str] | None = None,
    right_bullets: list[str] | None = None,
    output_path: str | Path | None = None,
) -> Image.Image:
    """Render a publication-ready side-by-side comparison image for README & demos."""
    orig = load_image(source)
    direct_est = estimate_visual_tokens(
        orig.width,
        orig.height,
        plan.direct_estimate.model_family,
        max_tokens=direct_max_tokens,
    )
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

    top_title = scenario_title or (
        "vlm-zoom: Direct Downscaled VLM Input vs. "
        "Adaptive Tiling + Pinned Legend (vLLM Prefix Caching)"
    )
    draw.text((pad, 16), top_title, fill=(240, 246, 255), font=h_font)
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

    draw.rectangle((pad, by, pad + panel_w, by + banner_h), fill=(175, 35, 45))
    draw.text(
        (pad + 14, by + 9),
        f"BEFORE: Direct Input ({direct_est.resized_width}x{direct_est.resized_height}, "
        f"{direct_est.visual_tokens} visual tok)  ->  Fine Details Lost",
        fill=(255, 255, 255),
        font=tag_font,
    )

    right_banner_text = after_banner or (
        f"AFTER: vlm-zoom ({plan.rows}x{plan.cols} tiles, <= {plan.max_tokens_per_tile} tok/tile, "
        f"Pinned Context KV Cache + WBF)"
    )
    draw.rectangle((rx, by, rx + panel_w, by + banner_h), fill=(16, 138, 82))
    draw.text((rx + 14, by + 9), right_banner_text, fill=(255, 255, 255), font=tag_font)

    left_img = degraded.resize((panel_w, panel_h), Image.Resampling.NEAREST)
    canvas.paste(left_img, (pad, header_h))
    draw.rectangle(
        (pad, header_h, pad + panel_w, header_h + panel_h), outline=(235, 85, 85), width=3
    )

    right_overlay = render_plan_overlay(
        orig,
        plan,
        detections=detections,
        legend_box=legend_box,
        pinned_label=pinned_label,
        glance_rois=glance_rois,
        max_preview_width=panel_w,
    ).resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    canvas.paste(right_overlay, (rx, header_h))
    draw.rectangle(
        (rx, header_h, rx + panel_w, header_h + panel_h), outline=(25, 210, 125), width=3
    )

    cbox = zoom_callout_box or (
        detections[0].box.pad(0.6, orig.width, orig.height)
        if detections
        else PixelBox(orig.width * 0.4, orig.height * 0.4, orig.width * 0.55, orig.height * 0.55)
    )
    fy = header_h + panel_h + 18
    crop_h = footer_h - 48
    crop_w = 420

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
    draw.text((pad, fy + 2), zoom_caption_left, fill=(245, 130, 130), font=tag_font)

    lcard_x = pad + crop_w + 24
    draw.text((lcard_x, fy + 28), "Direct VLM Bottleneck:", fill=(255, 135, 135), font=sub_font)
    l_lines = left_bullets or [
        f"• Effective scale collapses to {direct_est.min_scale:.2f}x",
        "• Small labels, symbols & targets blur out",
        "• Full canvas + legend compete for 1000 tok",
        f"• Raw full-res needs {plan.direct_estimate.visual_tokens:,} visual tokens",
    ]
    for idx, line in enumerate(l_lines[:4]):
        draw.text((lcard_x, fy + 58 + idx * 28), line, fill=(210, 215, 225), font=tag_font)

    nat_crop = orig.crop(cbox.to_int_tuple()).resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    canvas.paste(nat_crop, (rx, fy + 26))
    draw.rectangle((rx, fy + 26, rx + crop_w, fy + 26 + crop_h), outline=(25, 210, 125), width=2)
    draw.text((rx, fy + 2), zoom_caption_right, fill=(95, 235, 165), font=tag_font)

    rcard_x = rx + crop_w + 24
    draw.text((rcard_x, fy + 28), "Key Engineering Wins:", fill=(255, 215, 90), font=sub_font)
    r_lines = right_bullets or [
        f"• Exact {plan.direct_estimate.model_family} smart_resize token planner",
        f"• Detail retention: {direct_est.min_scale:.2f}x -> {plan.min_effective_scale:.2f}x",
        "• Pinned context prefix -> vLLM KV cache reuse",
        "• Cross-tile WBF/NMS + World-File GeoTransform",
    ]
    for idx, line in enumerate(r_lines[:4]):
        draw.text((rcard_x, fy + 58 + idx * 28), line, fill=(215, 228, 245), font=tag_font)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    return canvas

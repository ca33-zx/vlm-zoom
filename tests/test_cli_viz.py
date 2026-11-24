"""Unit tests for CLI commands (`inspect`, `tile`, `demo`) and visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vlm_zoom.cli import main as cli_main
from vlm_zoom.tiler import AdaptiveTiler, PixelBox
from vlm_zoom.viz import make_synthetic_geological_map, render_plan_overlay


def test_cli_inspect_json(tmp_path: Path, capsys):
    img_path = tmp_path / "large.png"
    Image.new("RGB", (4000, 2400), (128, 128, 128)).save(img_path)

    rc = cli_main(["inspect", str(img_path), "--model", "qwen3_vl", "--json"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dimensions"] == [4000, 2400]
    assert report["model_family"] == "qwen3_vl"
    assert report["num_tiles"] >= 4
    assert report["total_tile_tokens"] > 0


def test_cli_tile_with_pinned_legend(tmp_path: Path):
    img_path = tmp_path / "map.png"
    out_dir = tmp_path / "tiles"
    Image.new("RGB", (2400, 1600), (220, 230, 220)).save(img_path)

    rc = cli_main(
        [
            "tile",
            str(img_path),
            "--out-dir",
            str(out_dir),
            "--max-tokens-per-tile",
            "512",
            "--roi",
            "0,0,1800,1600",
            "--legend-box",
            "1850,100,2350,1500",
            "--format",
            "PNG",
        ]
    )
    assert rc == 0
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    assert (out_dir / "pinned_legend.png").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["tiles"]) >= 2
    assert manifest["pinned_legend"]["file"] == "pinned_legend.png"
    for entry in manifest["tiles"]:
        assert (out_dir / entry["file"]).exists()


def test_cli_demo_and_render_plan_overlay(tmp_path: Path):
    out_png = tmp_path / "demo.png"
    rc = cli_main(["demo", "--out", str(out_png), "--width", "2400", "--height", "1500"])
    assert rc == 0
    assert out_png.exists()
    with Image.open(out_png) as rendered:
        assert rendered.width > 1200
        assert rendered.height > 600

    img, legend_box, dets = make_synthetic_geological_map(width=2400, height=1500)
    plan = AdaptiveTiler("qwen3_vl", max_tokens_per_tile=512).plan(img.width, img.height)
    overlay = render_plan_overlay(
        img,
        plan,
        detections=dets,
        legend_box=legend_box,
        glance_rois=[PixelBox(400.0, 400.0, 1000.0, 900.0)],
        max_preview_width=1000,
    )
    assert overlay.width == 1000
    assert overlay.mode == "RGB"

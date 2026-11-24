<div align="center">

# `vlm-zoom`

**面向 vLLM 多模态推理的高分辨率自适应切片、精确视觉 Token 预算规划、图例前缀缓存（Prefix Caching）与 Glance-and-Zoom 引擎**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](tests/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

[English](README.md) | 简体中文

</div>

---

在实际业务与科研场景中，当把 **1:50,000 区域地质图、高分遥感卫星正射影像、工程 CAD 图纸、4K/8K GUI 截图或密集图文页面** 输入 `Qwen2.5-VL / Qwen3-VL / Qwen3.5 / InternVL` 等多模态大模型时，几乎都会遇到两难问题：

- **整图强行缩放到 1024 token 预算**：地质图上的地层代号（`J3`、`K1`、`gamma5`）、产状倾角、断层注记、遥感小目标直接糊成马赛克；
- **原图直出推理**：视觉 Token 和显存 KV Cache 瞬间爆炸；而简单粗暴地切成网格后，每个切片又会**丢失全局视角与右侧图例（Legend）**，导致模型“看得清局部，却认不出图例含义”，且切片交界处还会出现大量重复检测框。

**`vlm-zoom`** 用纯 Python（零 PyTorch / 零 GDAL 重依赖）一站式解决以上痛点，专为 **vLLM / SGLang** 生产部署设计。

![vlm-zoom 在 4.8K 地质图上的直出压缩 vs 自适应切片对比](assets/geological_map_demo.png)

---

## 核心特性

1. **精确视觉 Token 预算器（零像素解码）**：
   - 内置 `Qwen2-VL / Qwen2.5-VL`（factor=28）、`Qwen3-VL / Qwen3.5`（factor=32）的精确 `smart_resize` 算法，以及 `InternVL` 动态切片 + 缩略图计数模型；
   - 仅读图片文件头即可毫秒级规划最优 `(rows, cols)` 重叠网格，保证每个切片严格不超过 `max_tokens_per_tile` 且清晰度维持在近原生分辨率（`0.85x–1.0x`）。
2. **Pinned Context 图例固定 + vLLM Prefix Caching 零冗余计算**：
   - 支持把**地质图图例（`legend_box`）**或**全局缩略图（`pin_overview=True`）**以无损 PNG 固定在所有并发切片请求的消息前缀中；
   - 保证所有切片请求的前缀字节 100% 一致，配合 vLLM `--enable-prefix-caching`，图例与全局概览的视觉 KV Cache 只需计算 **1 次**，所有切片直接复用！
3. **两阶段 `Glance-and-Zoom` 聚焦推理**：
   - Stage 1 先用极低 Token 概览全图定位候选 ROI 区域；
   - Stage 2 仅对 ROI 区域做高分切片并行推理，跳过大面积无关空白背景。
4. **跨切片坐标还原与 WBF / NMS 融合**：
   - 自动解析 Qwen JSON (`bbox_2d`)、`<|box_start|>` 旧标签及方括号坐标，精准映射回原图全局像素坐标，并通过 **Weighted Boxes Fusion (WBF)** 或 **NMS + Containment** 消除切片重叠区的重复框与残缺框。
5. **纯 Python 地理坐标投影（免 GDAL）**：
   - 直接解析 ESRI 世界文件（`.tfw` / `.pgw` / `.jgw` / `.wld`），将融合后的检测框一键转换为真实地图坐标（WGS84 / UTM）。

---

## 快速开始

### 安装

```bash
pip install -e .
```

### 命令行一键体验（无需 GPU）

```bash
# 1. 生成 4.8K 地质图直出 vs vlm-zoom 效果对比图
vlm-zoom demo --out assets/geological_map_demo.png

# 2. 零解码检查任意大图在 Qwen3-VL 下的 Token 消耗与最优切片方案
vlm-zoom inspect your_8k_map.png --model Qwen/Qwen3-VL-4B-Instruct --json

# 3. 导出受预算约束的切片、无损图例与坐标清单 manifest.json
vlm-zoom tile your_8k_map.png --out-dir ./tiles_out --legend-box 3780,120,4750,2880
```

### Python API：对接 vLLM 自动前缀缓存

先启动 vLLM 服务（开启 `--enable-prefix-caching`）：

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --enable-prefix-caching \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 4}'
```

再调用 `ZoomEngine`：

```python
from vlm_zoom import AsyncVLLMClient, GeoTransform, PixelBox, ZoomEngine

client = AsyncVLLMClient(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen3-VL-4B-Instruct",
    max_concurrency=8,
)

engine = ZoomEngine(
    client,
    model_family="qwen3_vl",
    max_tokens_per_tile=1024,
    overlap=0.15,
    fusion_method="wbf",
)

result = engine.run(
    "regional_geology_8k.png",
    prompt=(
        "结合固定图例，定位图中所有断层注记与地层代号，"
        '输出 JSON [{"label": str, "bbox_2d": [x1, y1, x2, y2], "score": float}]。'
    ),
    mode="glance_and_zoom",
    legend_box=PixelBox(3780, 120, 4750, 2880),
    pin_overview=True,
    geo_transform=GeoTransform.from_world_file("regional_geology_8k.pgw", crs="EPSG:32650"),
)

print("切片数:", result.token_usage.num_tiles)
print("Prefix Caching 节省的视觉 Token 估算:", result.token_usage.estimated_prefix_tokens_saved)
for det, gbox in result.geo_detections:
    print(det.label, det.box.to_int_tuple(), "->", gbox)
```

---

## 更多示例与测试

- 地质图完整示例：[`examples/geological_map_legend_zoom.py`](examples/geological_map_legend_zoom.py)
- 遥感大图 Glance-and-Zoom 示例：[`examples/remote_sensing_vllm_batch.py`](examples/remote_sensing_vllm_batch.py)
- 运行测试与代码检查：

```bash
pytest -v
ruff check src tests examples
ruff format --check src tests examples
```

## 开源协议

本项目基于 [Apache-2.0](LICENSE) 协议开源。

"""vlm-zoom: High-resolution adaptive tiling, visual token budgeting, pinned-context
prefix caching, and Glance-and-Zoom inference for VLMs on vLLM.
"""

from .budget import (
    MODEL_SPECS,
    ModelFamily,
    TokenEstimate,
    VisionModelSpec,
    estimate_visual_tokens,
    resolve_model_spec,
    smart_resize,
)
from .client import AsyncVLLMClient, VLMBackend, VLMResponse, build_tile_messages
from .engine import TileResult, TokenUsageSummary, ZoomEngine, ZoomResult
from .fusion import (
    Detection,
    box_containment,
    box_iou,
    fuse_detections_nms,
    fuse_detections_wbf,
    parse_raw_boxes,
)
from .geo import GeoBox, GeoTransform
from .tiler import (
    AdaptiveTiler,
    PinnedContext,
    PixelBox,
    Tile,
    TileSpec,
    TilingPlan,
    build_pinned_context,
    normalize_scientific_image,
    read_image_size,
    resize_to_token_budget,
)

__version__ = "0.2.0"

__all__ = [
    "AdaptiveTiler",
    "AsyncVLLMClient",
    "Detection",
    "GeoBox",
    "GeoTransform",
    "MODEL_SPECS",
    "ModelFamily",
    "PinnedContext",
    "PixelBox",
    "Tile",
    "TileResult",
    "TileSpec",
    "TilingPlan",
    "TokenEstimate",
    "TokenUsageSummary",
    "VLMBackend",
    "VLMResponse",
    "VisionModelSpec",
    "ZoomEngine",
    "ZoomResult",
    "__version__",
    "box_containment",
    "box_iou",
    "build_pinned_context",
    "build_tile_messages",
    "estimate_visual_tokens",
    "fuse_detections_nms",
    "fuse_detections_wbf",
    "normalize_scientific_image",
    "parse_raw_boxes",
    "read_image_size",
    "resize_to_token_budget",
    "resolve_model_spec",
    "smart_resize",
]

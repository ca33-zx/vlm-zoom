"""Pure-Python geospatial coordinate projection via standard world files.

Supports ESRI world files (`.tfw`, `.jgw`, `.pgw`, `.wld`) used alongside GeoTIFFs,
georeferenced remote sensing imagery, and scanned geological maps — without requiring
heavy native GDAL / rasterio installations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tiler import PixelBox


@dataclass(frozen=True)
class GeoBox:
    """Bounding box in map/geographic coordinates `(min_x, min_y, max_x, max_y)`."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    crs: str | None = None


@dataclass(frozen=True)
class GeoTransform:
    """2D affine transform matching ESRI world-file / GDAL conventions.

    Map coordinates `(X, Y)` from pixel center `(col, row)`:
        X = origin_x + col * pixel_width + row * row_rotation
        Y = origin_y + col * col_rotation + row * pixel_height
    """

    pixel_width: float  # A: x-scale (positive east)
    col_rotation: float  # D: y-skew
    row_rotation: float  # B: x-skew
    pixel_height: float  # E: y-scale (usually negative north-up)
    origin_x: float  # C: x-coordinate of upper-left pixel center
    origin_y: float  # F: y-coordinate of upper-left pixel center
    crs: str | None = None

    @classmethod
    def from_world_file(cls, path: str | Path, *, crs: str | None = None) -> GeoTransform:
        """Load a 6-line ESRI world file (`.tfw`, `.jgw`, `.pgw`, `.wld`)."""
        lines = [
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) < 6:
            raise ValueError(f"World file {path} must have 6 numeric lines, got {len(lines)}.")
        a, d, b, e, c, f = (float(lines[i]) for i in range(6))
        return cls(
            pixel_width=a,
            col_rotation=d,
            row_rotation=b,
            pixel_height=e,
            origin_x=c,
            origin_y=f,
            crs=crs,
        )

    def pixel_to_map(self, x: float, y: float) -> tuple[float, float]:
        """Project pixel coordinate `(x, y)` to `(map_x, map_y)`."""
        mx = self.origin_x + x * self.pixel_width + y * self.row_rotation
        my = self.origin_y + x * self.col_rotation + y * self.pixel_height
        return mx, my

    def map_to_pixel(self, map_x: float, map_y: float) -> tuple[float, float]:
        """Inverse-project `(map_x, map_y)` back to pixel `(x, y)`."""
        det = self.pixel_width * self.pixel_height - self.row_rotation * self.col_rotation
        if abs(det) < 1e-18:
            raise ValueError("Singular affine GeoTransform cannot be inverted.")
        dx = map_x - self.origin_x
        dy = map_y - self.origin_y
        px = (self.pixel_height * dx - self.row_rotation * dy) / det
        py = (-self.col_rotation * dx + self.pixel_width * dy) / det
        return px, py

    def pixel_box_to_geo(self, box: PixelBox) -> GeoBox:
        """Project all 4 corners of a `PixelBox` to an axis-aligned `GeoBox`."""
        corners = [
            self.pixel_to_map(box.x1, box.y1),
            self.pixel_to_map(box.x2, box.y1),
            self.pixel_to_map(box.x1, box.y2),
            self.pixel_to_map(box.x2, box.y2),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return GeoBox(min(xs), min(ys), max(xs), max(ys), crs=self.crs)

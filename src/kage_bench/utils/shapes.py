"""
Geometric shapes rendering utilities for a JAX platformer (pure JAX, no NumPy, no PIL).

Design goals
------------
1) Single source of truth for shape geometry (esp. stars).
2) JAX-friendly polygon fill (even-odd rule), reusable by renderer.py.
3) Provide a JAX-only "preview/debug" renderer: render_shape_on_image_jax.

Notes
-----
- Image coordinates: x rightwards, y downwards.
- Angles for JAX geometry are in radians internally.
- render_shape_on_image_jax accepts angle in degrees (to match historical PIL helper usage),
  but you can pass radians by setting angle_is_degrees=False.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------

SHAPE_TYPES: List[str] = [
    "circle",
    "square",
    "line",
    "ellipse",
    "triangle",
    "star",
    "polygon",
    "cross",
    "diamond",
]

SHAPE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "violet": (128, 0, 128),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "purple": (128, 0, 128),
    "lime": (0, 255, 0),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "teal": (0, 128, 128),
    "indigo": (75, 0, 130),
    "coral": (255, 127, 80),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "white": (255, 255, 255),
}


# ---------------------------------------------------------------------
# Geometry helpers (pure JAX)
# ---------------------------------------------------------------------

def _regular_star_inner_ratio_5() -> float:
    r"""
    For the regular {5/2} star polygon (pentagram), with outer radius R and inner-vertex radius r:
        r / R = sin(pi/10) / sin(3pi/10).
    """
    return math.sin(math.pi / 10.0) / math.sin(3.0 * math.pi / 10.0)


def create_star_points(
    center_x: float | jnp.ndarray,
    center_y: float | jnp.ndarray,
    outer_radius: float | jnp.ndarray,
    inner_radius: float | None = None,
    num_points: int = 5,
    angle: float | jnp.ndarray = 0.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """
    Regular n-point star polygon with alternating outer/inner vertices.

    Returns
    -------
    pts : jnp.ndarray, shape (2*num_points, 2)
        Vertices ordered by increasing angle (CCW in math coords; in image coords y-down).
        Suitable for even-odd fill.
    """
    if inner_radius is None:
        if num_points != 5:
            raise ValueError("inner_radius=None only supported for num_points=5; pass inner_radius explicitly.")
        inner_radius = jnp.asarray(outer_radius, dtype) * jnp.asarray(_regular_star_inner_ratio_5(), dtype)

    # Keep parity index integer (fixes your bitwise/float crash).
    k_int = jnp.arange(2 * num_points, dtype=jnp.int32)
    k = k_int.astype(dtype)

    theta0 = jnp.asarray(-jnp.pi / 2.0, dtype) + jnp.asarray(angle, dtype)
    theta = theta0 + k * (jnp.asarray(jnp.pi / num_points, dtype))

    R = jnp.asarray(outer_radius, dtype)
    r = jnp.asarray(inner_radius, dtype)
    radii = jnp.where((k_int % 2) == 0, R, r)

    cx = jnp.asarray(center_x, dtype)
    cy = jnp.asarray(center_y, dtype)
    xs = cx + radii * jnp.cos(theta)
    ys = cy + radii * jnp.sin(theta)
    return jnp.stack([xs, ys], axis=-1)


def create_regular_polygon_points(
    center_x: float | jnp.ndarray,
    center_y: float | jnp.ndarray,
    radius: float | jnp.ndarray,
    num_sides: int,
    angle: float | jnp.ndarray = 0.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Regular polygon vertices, shape (num_sides, 2). angle=0 puts a vertex at top."""
    k = jnp.arange(num_sides, dtype=dtype)
    theta0 = jnp.asarray(-jnp.pi / 2.0, dtype) + jnp.asarray(angle, dtype)
    theta = theta0 + k * (jnp.asarray(2.0 * jnp.pi / num_sides, dtype))
    cx = jnp.asarray(center_x, dtype)
    cy = jnp.asarray(center_y, dtype)
    rad = jnp.asarray(radius, dtype)
    xs = cx + rad * jnp.cos(theta)
    ys = cy + rad * jnp.sin(theta)
    return jnp.stack([xs, ys], axis=-1)


# ---------------------------------------------------------------------
# Rasterization: even-odd polygon fill (ray casting to +∞ in x)
# ---------------------------------------------------------------------

def polygon_evenodd_mask(
    H: int,
    W: int,
    pts_xy: jnp.ndarray,
    *,
    pixel_center: bool = True,
) -> jnp.ndarray:
    """
    Rasterize a simple (possibly concave) polygon via even-odd rule.

    Parameters
    ----------
    pts_xy : (V,2) float32 vertices in image coordinates.
    pixel_center : test at (x+0.5, y+0.5) if True.

    Returns
    -------
    mask : (H,W) bool
    """
    pts_xy = jnp.asarray(pts_xy, dtype=jnp.float32)
    x0 = pts_xy[:, 0]
    y0 = pts_xy[:, 1]
    x1 = jnp.roll(x0, shift=-1)
    y1 = jnp.roll(y0, shift=-1)

    ys = jnp.arange(H, dtype=jnp.float32)[:, None]
    xs = jnp.arange(W, dtype=jnp.float32)[None, :]
    if pixel_center:
        px = xs + 0.5
        py = ys + 0.5
    else:
        px = xs
        py = ys

    px3 = px[:, :, None]  # (H,W,1)
    py3 = py[:, :, None]

    x0e = x0[None, None, :]  # (1,1,V)
    y0e = y0[None, None, :]
    x1e = x1[None, None, :]
    y1e = y1[None, None, :]

    # Straddle test: y0>py xor y1>py
    cond_straddle = (y0e > py3) != (y1e > py3)

    # x intersection with horizontal line y=py
    denom = y1e - y0e
    denom = jnp.where(denom == 0.0, 1e-12, denom)
    x_int = x0e + (py3 - y0e) * (x1e - x0e) / denom

    cond_right = px3 < x_int
    crossings = jnp.sum(cond_straddle & cond_right, axis=-1)  # int
    return (crossings & 1) == 1


def apply_mask_color(img: jnp.ndarray, mask: jnp.ndarray, color_rgb_u8: jnp.ndarray) -> jnp.ndarray:
    """Set img[mask] = color (uint8 RGB) using where."""
    color_rgb_u8 = jnp.asarray(color_rgb_u8, dtype=jnp.uint8)
    return jnp.where(mask[:, :, None], color_rgb_u8[None, None, :], img)


# ---------------------------------------------------------------------
# Convenience masks (pure JAX)
# ---------------------------------------------------------------------

def _coords(H: int, W: int, *, pixel_center: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    ys = jnp.arange(H, dtype=jnp.float32)[:, None]
    xs = jnp.arange(W, dtype=jnp.float32)[None, :]
    if pixel_center:
        return ys + 0.5, xs + 0.5
    return ys, xs


def circle_mask(H: int, W: int, cx: float, cy: float, radius: float, *, pixel_center: bool = True) -> jnp.ndarray:
    ys, xs = _coords(H, W, pixel_center=pixel_center)
    dx = xs - jnp.asarray(cx, jnp.float32)
    dy = ys - jnp.asarray(cy, jnp.float32)
    r = jnp.asarray(radius, jnp.float32)
    return (dx * dx + dy * dy) <= r * r


def rotated_box_mask(
    H: int,
    W: int,
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
    angle: float,
    *,
    pixel_center: bool = True,
) -> jnp.ndarray:
    ys, xs = _coords(H, W, pixel_center=pixel_center)
    x = xs - jnp.asarray(cx, jnp.float32)
    y = ys - jnp.asarray(cy, jnp.float32)
    a = jnp.asarray(angle, jnp.float32)
    ca = jnp.cos(a)
    sa = jnp.sin(a)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    return (jnp.abs(xr) <= jnp.asarray(half_w, jnp.float32)) & (jnp.abs(yr) <= jnp.asarray(half_h, jnp.float32))


# ---------------------------------------------------------------------
# JAX-only preview/debug renderer
# ---------------------------------------------------------------------

def render_shape_on_image_jax(
    img: jnp.ndarray,
    shape_type: str,
    center_x: int | jnp.ndarray,
    center_y: int | jnp.ndarray,
    width: int | jnp.ndarray,
    height: int | jnp.ndarray,
    color: tuple[int, int, int] | jnp.ndarray,
    angle: float | jnp.ndarray = 0.0,
    *,
    angle_is_degrees: bool = True,
) -> jnp.ndarray:
    """
    JAX-only shape renderer (no PIL). Intended for preview/debug pipelines and tests.

    Semantics aim to match the previous PIL helper:
      - center_x/center_y in image pixel coordinates.
      - width/height define a bounding box; "size" is derived from min(width,height).
      - angle defaults to degrees (PIL-style); set angle_is_degrees=False for radians.
    """
    H, W = img.shape[:2]
    cx = jnp.asarray(center_x, jnp.float32)
    cy = jnp.asarray(center_y, jnp.float32)

    w = jnp.asarray(width, jnp.float32)
    h = jnp.asarray(height, jnp.float32)
    radius = 0.5 * jnp.minimum(w, h)

    if isinstance(color, tuple):
        col = jnp.array(color, dtype=jnp.uint8)
    else:
        col = jnp.asarray(color, dtype=jnp.uint8)

    a = jnp.asarray(angle, jnp.float32)
    if angle_is_degrees:
        a = a * (jnp.pi / 180.0)

    st = shape_type.lower().strip()

    # Use pixel-center convention for consistent rasterization.
    cxp = cx + 0.5
    cyp = cy + 0.5

    def _circle():
        m = circle_mask(H, W, cxp, cyp, radius, pixel_center=True)
        return apply_mask_color(img, m, col)

    def _square():
        m = rotated_box_mask(H, W, cxp, cyp, radius, radius, a, pixel_center=True)
        return apply_mask_color(img, m, col)

    def _triangle():
        pts = create_regular_polygon_points(cxp, cyp, radius, 3, angle=a, dtype=jnp.float32)
        m = polygon_evenodd_mask(H, W, pts, pixel_center=True)
        return apply_mask_color(img, m, col)

    def _star():
        pts = create_star_points(cxp, cyp, outer_radius=radius, inner_radius=None, num_points=5, angle=a, dtype=jnp.float32)
        m = polygon_evenodd_mask(H, W, pts, pixel_center=True)
        return apply_mask_color(img, m, col)

    def _polygon():
        pts = create_regular_polygon_points(cxp, cyp, radius, 6, angle=a, dtype=jnp.float32)
        m = polygon_evenodd_mask(H, W, pts, pixel_center=True)
        return apply_mask_color(img, m, col)

    def _diamond():
        # L1 ball scaled, then rotated.
        ys, xs = _coords(H, W, pixel_center=True)
        x = xs - cxp
        y = ys - cyp
        ca = jnp.cos(a)
        sa = jnp.sin(a)
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        # axis-aligned diamond: |x|/a + |y|/b <= 1
        aa = radius + 1e-12
        bb = radius + 1e-12
        m = (jnp.abs(xr) / aa + jnp.abs(yr) / bb) <= 1.0
        return apply_mask_color(img, m, col)

    def _cross():
        # Union of two rotated rectangles
        half_len = radius
        thickness = jnp.maximum(1.0, radius / 4.0)
        m1 = rotated_box_mask(H, W, cxp, cyp, half_len, 0.5 * thickness, a, pixel_center=True)
        m2 = rotated_box_mask(H, W, cxp, cyp, 0.5 * thickness, half_len, a, pixel_center=True)
        return apply_mask_color(img, (m1 | m2), col)

    def _ellipse():
        ys, xs = _coords(H, W, pixel_center=True)
        x = xs - cxp
        y = ys - cyp
        ca = jnp.cos(a)
        sa = jnp.sin(a)
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        rx = 0.5 * w
        ry = 0.5 * h
        m = (xr / (rx + 1e-12)) ** 2 + (yr / (ry + 1e-12)) ** 2 <= 1.0
        return apply_mask_color(img, m, col)

    def _line():
        # Render as a rotated thin rectangle of length ~ min(w,h)
        half_len = radius
        thickness = jnp.maximum(1.0, radius / 4.0)
        m = rotated_box_mask(H, W, cxp, cyp, half_len, 0.5 * thickness, a, pixel_center=True)
        return apply_mask_color(img, m, col)

    # Dispatch with lax.switch over a small fixed set would require integer ids; here string dispatch is fine
    # for non-hot preview/debug use. If you need it JIT-fast, wrap st->id outside and switch on id.
    if st == "circle":
        return _circle()
    if st == "square":
        return _square()
    if st == "triangle":
        return _triangle()
    if st == "star":
        return _star()
    if st == "polygon":
        return _polygon()
    if st == "diamond":
        return _diamond()
    if st == "cross":
        return _cross()
    if st == "ellipse":
        return _ellipse()
    if st == "line":
        return _line()
    return _circle()

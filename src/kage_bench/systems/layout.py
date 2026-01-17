from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class LayoutConfig:
    # Canvas
    length: int = 128
    height_px: int = 128

    # Geometry
    base_ground_y: int = 96         # reference y for height=0 (pixel row)
    pix_per_unit: int = 2           # pixels per height unit
    ground_thickness: int = 2       # thickness of horizontal ground band in pixels

    # Step structure
    run_width: int = 20             # segment width in columns

    # Event distribution on segments
    p_change: float = 0.5
    p_up_given_change: float = 0.5
    min_step_height: int = 5
    max_step_height: int = 10

    # Clamp heights (in height units, not pixels)
    min_height: int = 0
    max_height: int = 20
    
    # Visual
    layout_colors: tuple[str, ...] | str = "gray"  # Color(s) for layout: single or tuple for random selection


@struct.dataclass
class Layout:
    ground_top: jnp.ndarray      # (W,) int32
    ground_bottom: jnp.ndarray   # (W,) int32
    solid_mask: jnp.ndarray      # (H,W) bool


def generate_layout(key: jax.Array, cfg: LayoutConfig) -> Layout:
    W = int(cfg.length)
    H = int(cfg.height_px)
    w = int(cfg.run_width)
    if w <= 0:
        raise ValueError("run_width must be positive")

    Lc = (W + w - 1) // w
    key_evt, key_step = jax.random.split(key, 2)

    # Sample coarse events: {-1, 0, +1} per segment.
    u = jax.random.uniform(key_evt, shape=(Lc,), dtype=jnp.float32)
    t_flat = 1.0 - cfg.p_change
    t_up = t_flat + cfg.p_change * cfg.p_up_given_change
    e = jnp.where(u < t_flat, 0, jnp.where(u < t_up, 1, -1)).astype(jnp.int32)  # (Lc,)

    # Step magnitudes on segments.
    step_mag = jax.random.randint(
        key_step,
        shape=(Lc,),
        minval=int(cfg.min_step_height),
        maxval=int(cfg.max_step_height) + 1,
        dtype=jnp.int32,
    )
    dh = e * step_mag  # (Lc,)

    # Integrate bounded random walk in height units.
    h0 = jnp.int32((cfg.min_height + cfg.max_height) // 2)

    def scan_step(h, delta):
        h2 = jnp.clip(h + delta, cfg.min_height, cfg.max_height)
        return h2, h2

    _, h_c = jax.lax.scan(scan_step, h0, dh)              # (Lc,)
    h_c = jnp.concatenate([h0[None], h_c], axis=0)        # (Lc+1,)

    # Expand segment heights to per-column heights (piecewise-constant).
    h_col = jnp.repeat(h_c[:-1], w, axis=0)[:W]           # (W,)

    # Convert to pixel ground band.
    ground_top = (jnp.int32(cfg.base_ground_y) - jnp.int32(cfg.pix_per_unit) * h_col).astype(jnp.int32)  # (W,)
    ground_bottom = (ground_top + jnp.int32(cfg.ground_thickness)).astype(jnp.int32)                      # (W,)

    # Build solid mask: horizontal ground + vertical step connectors.
    ys = jnp.arange(H, dtype=jnp.int32)[:, None]          # (H,1)

    ground_mask = (ys >= ground_top[None, :]) & (ys < ground_bottom[None, :])  # (H,W)

    # Vertical connectors at boundaries where ground_top changes.
    dy = jnp.diff(ground_top)                             # (W-1,)
    wall_x = dy != 0                                      # (W-1,)

    y_lo = jnp.minimum(ground_top[:-1], ground_top[1:])   # (W-1,)
    y_hi = jnp.maximum(ground_top[:-1], ground_top[1:])   # (W-1,)

    wall_hw_m1 = wall_x[None, :] & (ys >= y_lo[None, :]) & (ys < y_hi[None, :])  # (H,W-1)
    wall_mask = jnp.pad(wall_hw_m1, ((0, 0), (1, 0)), mode="constant", constant_values=False)             # (H,W)

    solid = ground_mask | wall_mask

    return Layout(
        ground_top=ground_top,
        ground_bottom=ground_bottom,
        solid_mask=solid,
    )

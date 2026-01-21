"""Pure functional rendering system for JAX platformer.

All rendering functions are pure and JIT-compatible:
render(state, config) -> uint8 RGB image
"""

from __future__ import annotations

from typing import List, Optional

import jax
import jax.numpy as jnp

from ..core.state import EnvState
from ..core.config import EnvConfig
from ..entities.character import render_sprite_from_list, render_character_rect
from ..entities.npc import render_npc_by_type
from ..systems.generation.background import apply_background
from ..systems.generation.effects import apply_effects
from ..systems.generation.filters import apply_filters, select_filter_by_index
from ..utils.shapes import SHAPE_COLORS, create_star_points, polygon_evenodd_mask, apply_mask_color, render_shape_on_image_jax


def render(
    state: EnvState,
    config: EnvConfig,
    character_sprites: List[jnp.ndarray],
    npc_sprite_sets: List[List[jnp.ndarray]],
    sticky_npc_sprite_sets: List[List[jnp.ndarray]],
    character_skin_sets: Optional[List[List[jnp.ndarray]]] = None,
) -> jnp.ndarray:
    H, W = config.H, config.W
    img = jnp.zeros((H, W, 3), dtype=jnp.uint8)

    # Render background first as the base layer
    img = apply_background(img, state.bg_image, state.camera_x, config.background)

    if config.distractors.enabled:
        img = _render_distractors(img, state, config)

    img = _render_layout(img, state, config)

    if config.npc.enabled and npc_sprite_sets:
        img = _render_world_npcs(img, state, config, npc_sprite_sets)

    if config.npc.sticky_enabled and sticky_npc_sprite_sets:
        img = _render_sticky_npcs(img, state, config, sticky_npc_sprite_sets)

    img = _render_character(img, state, config, character_sprites, character_skin_sets)

    img = _apply_filters(img, state, config)
    img = apply_effects(img, config.effects, light_positions=state.light_positions)
    return img


# -----------------------------------------------------------------------------
# Distractor primitives
# -----------------------------------------------------------------------------

def _coords(H: int, W: int, cx_internal: float = 0.0, cy_internal: float = 0.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return coordinate grids relative to center if provided."""
    ys = jnp.arange(H, dtype=jnp.float32)[:, None] - cy_internal
    xs = jnp.arange(W, dtype=jnp.float32)[None, :] - cx_internal
    return ys, xs


def _rotate_local(xs: jnp.ndarray, ys: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, angle: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_local = xs - cx
    y_local = ys - cy
    ca = jnp.cos(angle)
    sa = jnp.sin(angle)
    xr = x_local * ca - y_local * sa
    yr = x_local * sa + y_local * ca
    return xr, yr


def _regular_polygon_pts(cx: jnp.ndarray, cy: jnp.ndarray, radius: jnp.ndarray, n: int, angle: jnp.ndarray) -> jnp.ndarray:
    k = jnp.arange(n, dtype=jnp.float32)
    theta0 = -jnp.pi / 2.0 + angle.astype(jnp.float32)
    theta = theta0 + k * (2.0 * jnp.pi / float(n))
    xs = cx.astype(jnp.float32) + radius.astype(jnp.float32) * jnp.cos(theta)
    ys = cy.astype(jnp.float32) + radius.astype(jnp.float32) * jnp.sin(theta)
    return jnp.stack([xs, ys], axis=-1)


def _render_localized_shape(
    img: jnp.ndarray,
    cx: jnp.ndarray,
    cy: jnp.ndarray,
    size: jnp.ndarray,
    color: jnp.ndarray,
    angle: jnp.ndarray,
    shape_fn
) -> jnp.ndarray:
    """Helper to render a shape on a small localized patch instead of full image."""
    H, W = img.shape[:2]
    # Patch size MUST be a static integer for JIT.
    # Distractor max size is usually <= 20. 32 is a safe static size.
    patch_size = 32
    half_patch = patch_size // 2
    
    # Boundary check for visibility
    visible = (cx > -half_patch) & (cx < W + half_patch) & (cy > -half_patch) & (cy < H + half_patch)
    
    # Padding approach for safe slicing (padding must be a static integer)
    padded_img = jnp.pad(img, ((patch_size, patch_size), (patch_size, patch_size), (0, 0)), mode='constant')
    py_start = (jnp.round(cy).astype(jnp.int32) - half_patch) + patch_size
    px_start = (jnp.round(cx).astype(jnp.int32) - half_patch) + patch_size
    
    # Slice patch (patch_size must be static)
    patch = jax.lax.dynamic_slice(padded_img, (py_start, px_start, 0), (patch_size, patch_size, 3))
    
    # Render shape on patch (cx, cy are now localized to the patch center)
    local_cx = (cx - jnp.round(cx)) + float(half_patch)
    local_cy = (cy - jnp.round(cy)) + float(half_patch)
    
    rendered_patch = shape_fn(patch, local_cx, local_cy, size, color, angle)
    
    # Update slice
    updated_padded = jax.lax.dynamic_update_slice(padded_img, rendered_patch, (py_start, px_start, 0))
    
    # Crop (indices must be static if not using dynamic_slice)
    res = updated_padded[patch_size:patch_size+H, patch_size:patch_size+W]
    return jnp.where(visible, res, img)


def _render_distractor_circle(img: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, size: jnp.ndarray, color: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    def shape_fn(patch, lcx, lcy, s, col, ang):
        h, w = patch.shape[:2]
        ys, xs = _coords(h, w, lcx, lcy)
        radius = 0.5 * s.astype(jnp.float32)
        mask = (xs**2 + ys**2) <= (radius**2)
        return apply_mask_color(patch, mask, col)
    return _render_localized_shape(img, cx, cy, size, color, angle, shape_fn)


def _render_distractor_square(img: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, size: jnp.ndarray, color: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    def shape_fn(patch, lcx, lcy, s, col, ang):
        h, w = patch.shape[:2]
        ys, xs = _coords(h, w)
        xr, yr = _rotate_local(xs, ys, lcx, lcy, ang)
        half = 0.5 * s.astype(jnp.float32)
        mask = (xr >= -half) & (xr <= half) & (yr >= -half) & (yr <= half)
        return apply_mask_color(patch, mask, col)
    return _render_localized_shape(img, cx, cy, size, color, angle, shape_fn)


def _render_distractor_triangle(img: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, size: jnp.ndarray, color: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    def shape_fn(patch, lcx, lcy, s, col, ang):
        h, w = patch.shape[:2]
        R = 0.5 * s.astype(jnp.float32)
        pts = _regular_polygon_pts(lcx + 0.5, lcy + 0.5, R, 3, ang)
        mask = polygon_evenodd_mask(h, w, pts, pixel_center=True)
        return apply_mask_color(patch, mask, col)
    return _render_localized_shape(img, cx, cy, size, color, angle, shape_fn)


def _render_distractor_diamond(img: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, size: jnp.ndarray, color: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    def shape_fn(patch, lcx, lcy, s, col, ang):
        h, w = patch.shape[:2]
        ys, xs = _coords(h, w)
        a = 0.5 * s.astype(jnp.float32)
        b = s.astype(jnp.float32)
        xr, yr = _rotate_local(xs, ys, lcx, lcy, ang)
        mask = (jnp.abs(xr) / (a + 1e-12) + jnp.abs(yr) / (b + 1e-12)) <= 1.0
        return apply_mask_color(patch, mask, col)
    return _render_localized_shape(img, cx, cy, size, color, angle, shape_fn)


def _render_distractor_star(img: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, size: jnp.ndarray, color: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    def shape_fn(patch, lcx, lcy, s, col, ang):
        h, w = patch.shape[:2]
        R = 0.5 * s.astype(jnp.float32)
        pts = create_star_points(lcx + 0.5, lcy + 0.5, R, None, 5, ang)
        mask = polygon_evenodd_mask(h, w, pts, pixel_center=True)
        return apply_mask_color(patch, mask, col)
    return _render_localized_shape(img, cx, cy, size, color, angle, shape_fn)


def _render_distractor_polygon(img: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, size: jnp.ndarray, color: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    def shape_fn(patch, lcx, lcy, s, col, ang):
        h, w = patch.shape[:2]
        R = 0.5 * s.astype(jnp.float32)
        pts = _regular_polygon_pts(lcx + 0.5, lcy + 0.5, R, 6, ang)
        mask = polygon_evenodd_mask(h, w, pts, pixel_center=True)
        return apply_mask_color(patch, mask, col)
    return _render_localized_shape(img, cx, cy, size, color, angle, shape_fn)


def _render_distractor_cross(img: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, size: jnp.ndarray, color: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    def shape_fn(patch, lcx, lcy, s, col, ang):
        h, w = patch.shape[:2]
        ys, xs = _coords(h, w)
        xr, yr = _rotate_local(xs, ys, lcx, lcy, ang)
        half_len = 0.5 * s.astype(jnp.float32)
        ht = 0.5 * jnp.maximum(1.0, s.astype(jnp.float32) / 8.0)
        mask = ((xr >= -half_len) & (xr <= half_len) & (yr >= -ht) & (yr <= ht)) | \
               ((yr >= -half_len) & (yr <= half_len) & (xr >= -ht) & (xr <= ht))
        return apply_mask_color(patch, mask, col)
    return _render_localized_shape(img, cx, cy, size, color, angle, shape_fn)


def _render_distractor_ellipse(img: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, size: jnp.ndarray, color: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    def shape_fn(patch, lcx, lcy, s, col, ang):
        h, w = patch.shape[:2]
        ys, xs = _coords(h, w)
        rx, ry = 0.5 * s.astype(jnp.float32), 0.3 * s.astype(jnp.float32)
        xr, yr = _rotate_local(xs, ys, lcx, lcy, ang)
        mask = (xr / (rx + 1e-12)) ** 2 + (yr / (ry + 1e-12)) ** 2 <= 1.0
        return apply_mask_color(patch, mask, col)
    return _render_localized_shape(img, cx, cy, size, color, angle, shape_fn)


def _render_distractor_line(img: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray, size: jnp.ndarray, color: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    def shape_fn(patch, lcx, lcy, s, col, ang):
        h, w = patch.shape[:2]
        ys, xs = _coords(h, w)
        xr, yr = _rotate_local(xs, ys, lcx, lcy, ang)
        half_len, half_thick = 0.5 * s.astype(jnp.float32), 0.5 * jnp.maximum(1.0, s.astype(jnp.float32) / 8.0)
        mask = (jnp.abs(yr) <= half_thick) & (xr >= -half_len) & (xr <= half_len)
        return apply_mask_color(patch, mask, col)
    return _render_localized_shape(img, cx, cy, size, color, angle, shape_fn)


# -----------------------------------------------------------------------------
# Distractors driver (JAX loop)
# -----------------------------------------------------------------------------

def _normalize_list(x) -> list:
    return [x] if isinstance(x, str) else list(x)


def _palette_from_color_names(names: list[str]) -> jnp.ndarray:
    rgb = [SHAPE_COLORS.get(n, (255, 0, 0)) for n in names]
    return jnp.array(rgb, dtype=jnp.uint8)


def _shape_branches_for_names(names: list[str]):
    def norm(s: str) -> str:
        return s.lower().strip()

    branches = []
    for nm in names:
        s = norm(nm)
        if s in {"circle", "round", "circ"}:
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_circle(img, cx, cy, size, color, angle))
        elif s in {"square", "rectangle", "rect", "box"}:
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_square(img, cx, cy, size, color, angle))
        elif s in {"triangle", "tri"}:
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_triangle(img, cx, cy, size, color, angle))
        elif s in {"diamond", "dia"}:
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_diamond(img, cx, cy, size, color, angle))
        elif s == "star":
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_star(img, cx, cy, size, color, angle))
        elif s == "polygon":
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_polygon(img, cx, cy, size, color, angle))
        elif s == "cross":
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_cross(img, cx, cy, size, color, angle))
        elif s == "ellipse":
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_ellipse(img, cx, cy, size, color, angle))
        elif s == "line":
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_line(img, cx, cy, size, color, angle))
        else:
            branches.append(lambda img, cx, cy, size, color, angle: _render_distractor_circle(img, cx, cy, size, color, angle))
    return branches


def _render_distractors(img: jnp.ndarray, state: EnvState, config: EnvConfig) -> jnp.ndarray:
    shape_names = _normalize_list(config.distractors.shape_types)
    color_names = _normalize_list(config.distractors.shape_colors)

    palette = _palette_from_color_names(color_names)  # (K,3) uint8
    shape_branches = _shape_branches_for_names(shape_names)
    Kc = palette.shape[0]
    Ks = len(shape_branches)

    max_d = state.dist_x.shape[0]
    count = jnp.asarray(config.distractors.count, dtype=jnp.int32)

    def body(i: jnp.ndarray, img_acc: jnp.ndarray) -> jnp.ndarray:
        is_valid = i < count

        cx = jnp.round(state.dist_x[i]).astype(jnp.float32)
        cy = jnp.round(state.dist_y[i]).astype(jnp.float32)
        size = state.dist_sizes[i].astype(jnp.int32)
        angle = state.dist_angles[i].astype(jnp.float32)

        ci = jnp.clip(state.dist_color_indices[i].astype(jnp.int32), 0, Kc - 1)
        si = jnp.clip(state.dist_shape_indices[i].astype(jnp.int32), 0, Ks - 1)
        color = palette[ci]  # uint8 (3,)

        def render_one():
            # IMPORTANT: avoid late-binding by capturing fn as default arg.
            return jax.lax.switch(
                si,
                [lambda fn=fn: fn(img_acc, cx, cy, size, color, angle) for fn in shape_branches],
            )

        return jax.lax.cond(is_valid, render_one, lambda: img_acc)

    return jax.lax.fori_loop(0, max_d, body, img)


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------

def _render_layout(img: jnp.ndarray, state: EnvState, config: EnvConfig) -> jnp.ndarray:
    H, W = config.H, config.W
    solid = state.layout_solid_mask
    layout_width = solid.shape[1]

    ys = jnp.arange(H, dtype=jnp.int32)[:, None]
    xs = jnp.arange(W, dtype=jnp.int32)[None, :]

    xs_world = xs + jnp.round(state.camera_x).astype(jnp.int32)
    xs_world = jnp.clip(xs_world, 0, layout_width - 1)

    solid_view = solid[ys, xs_world]
    return jnp.where(solid_view[:, :, None], state.layout_color[None, None, :], img)


# -----------------------------------------------------------------------------
# NPCs
# -----------------------------------------------------------------------------

def _render_world_npcs(
    img: jnp.ndarray,
    state: EnvState,
    config: EnvConfig,
    npc_sprite_sets: List[List[jnp.ndarray]],
) -> jnp.ndarray:
    H, W = config.H, config.W
    char_w = config.character.width
    char_h = config.character.height
    cam_x = jnp.round(state.camera_x).astype(jnp.int32)

    max_npcs = state.npc_x.shape[0]

    def body(i, img_acc):
        npc_world_x = state.npc_x[i]
        npc_world_y = state.npc_y[i]
        npc_screen_x = npc_world_x - cam_x
        npc_screen_y = npc_world_y

        margin = char_w
        visible = (npc_screen_x >= -margin) & (npc_screen_x < W + margin) & (npc_world_x > 0)

        def do():
            return render_npc_by_type(
                img_acc,
                npc_sprite_sets,
                state.npc_types[i],
                state.npc_sprite_indices[i],
                npc_screen_x,
                npc_screen_y,
                char_w,
                char_h,
            )

        return jax.lax.cond(visible, do, lambda: img_acc)

    return jax.lax.fori_loop(0, max_npcs, body, img)


def _render_sticky_npcs(
    img: jnp.ndarray,
    state: EnvState,
    config: EnvConfig,
    sticky_npc_sprite_sets: List[List[jnp.ndarray]],
) -> jnp.ndarray:
    char_w = config.character.width
    char_h = config.character.height

    hero_x_screen = state.x - state.camera_x
    agent_x_int = jnp.round(hero_x_screen).astype(jnp.int32)

    max_sticky = state.sticky_x_offsets.shape[0]
    count = state.actual_sticky_count.astype(jnp.int32)

    def body(i, img_acc):
        valid = i < count
        sx = agent_x_int + state.sticky_x_offsets[i]
        sy = state.sticky_y[i]

        def do():
            return render_npc_by_type(
                img_acc,
                sticky_npc_sprite_sets,
                state.sticky_types[i],
                state.sticky_sprite_indices[i],
                sx,
                sy,
                char_w,
                char_h,
            )

        return jax.lax.cond(valid, do, lambda: img_acc)

    return jax.lax.fori_loop(0, max_sticky, body, img)


# -----------------------------------------------------------------------------
# Player
# -----------------------------------------------------------------------------

def _render_character(
    img: jnp.ndarray,
    state: EnvState,
    config: EnvConfig,
    character_sprites: List[jnp.ndarray],
    character_skin_sets: Optional[List[List[jnp.ndarray]]] = None,
) -> jnp.ndarray:
    char_w = config.character.width
    char_h = config.character.height

    hero_x_screen = state.x - state.camera_x
    hero_y_screen = state.y
    cx = jnp.round(hero_x_screen).astype(jnp.int32)
    cy = jnp.round(hero_y_screen + 1).astype(jnp.int32)  # Shift bounding box down by 1 pixel

    if config.character.use_sprites:
        if character_skin_sets and len(character_skin_sets) > 0:
            def branch_for_skin(skin_sprites):
                return lambda: render_sprite_from_list(
                    img,
                    skin_sprites,
                    state.sprite_idx,
                    cx,
                    cy,
                    char_w,
                    char_h,
                )
            branches = [branch_for_skin(skin) for skin in character_skin_sets]
            idx = jnp.clip(state.char_skin_idx, 0, len(branches) - 1)
            return jax.lax.switch(idx, branches)

        if character_sprites:
            return render_sprite_from_list(img, character_sprites, state.sprite_idx, cx, cy, char_w, char_h)
        return img

    if config.character.use_shape:
        shape_names = [config.character.shape_types] if isinstance(config.character.shape_types, str) else list(config.character.shape_types)
        color_names = [config.character.shape_colors] if isinstance(config.character.shape_colors, str) else list(config.character.shape_colors)
        
        palette = _palette_from_color_names(color_names)
        K_color = palette.shape[0]
        color_idx = jnp.clip(state.shape_color_idx.astype(jnp.int32), 0, K_color - 1)
        color = palette[color_idx]
        
        K_shape = len(shape_names)
        shape_idx = jnp.clip(state.shape_type_idx.astype(jnp.int32), 0, K_shape - 1)
        
        def render_shape_by_name(name):
            return lambda: render_shape_on_image_jax(
                img, name, cx, cy, char_w, char_h, color, state.shape_angle, angle_is_degrees=True
            )
            
        branches = [render_shape_by_name(name) for name in shape_names]
        return jax.lax.switch(shape_idx, branches)

    return render_character_rect(img, cx, cy, char_w, char_h, config.character.fallback_color)


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------

def _apply_filters(img: jnp.ndarray, state: EnvState, config: EnvConfig) -> jnp.ndarray:
    filter_key = state.filter_key

    # Determine which filter to apply if pop_filter_list is present
    selected_filter = None
    if len(config.filters.pop_filter_list) > 0:
        # We handle the selection by index inside apply_filters to keep it clean
        # but since we have multiple filter systems, we'll just pass the index
        # to select_filter_by_index or similar.
        # Actually, the most efficient is to pass the selected name string (if static)
        # or handle it inside a JIT function.
        
        # Original logic had a branch. Let's simplify.
        selected_filter_idx = state.selected_filter_idx
        # This is a bit tricky because select_filter_by_index needs the names.
        # We'll just use the existing apply_filters but smarter.
        pass

    # Simplified call to avoid double astype if possible
    # We'll modify select_filter_by_index to accept a 'None' index or similar
    # if we wanted to be extreme, but for now just merging is good.
    
    if len(config.filters.pop_filter_list) > 0:
        # If we have a list, we pick one based on the state's index
        img_f = img.astype(jnp.float32)
        # Apply the chosen preset
        img_f = select_filter_by_index(state.selected_filter_idx, config.filters.pop_filter_list, img_f)
        # Then apply the rest of the config (already in float32 path)
        # We need a version of apply_filters that takes float32...
        # Or just call apply_filters with the rest of the config.
        # Let's just use the existing one but fix the astype overhead.
        return apply_filters(img_f.astype(jnp.uint8), config.filters, key=filter_key)

    return apply_filters(img, config.filters, key=filter_key)

"""
Advanced visual effects for JAX platformer observations.

Includes dynamic lighting, glow effects, and other post-processing effects.
All effects are JIT-compatible and GPU-friendly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp


# ============================================================================
# Light Color Presets
# ============================================================================

LIGHT_COLORS = {
    "warm_white": (255, 245, 230),
    "cool_white": (230, 240, 255),
    "yellow": (255, 255, 100),
    "orange": (255, 160, 50),
    "red": (255, 80, 80),
    "green": (100, 255, 100),
    "cyan": (100, 255, 255),
    "blue": (100, 150, 255),
    "purple": (200, 100, 255),
    "pink": (255, 150, 200),
    "gold": (255, 215, 0),
    "fire": (255, 140, 0),
}


# ============================================================================
# Effect Configuration
# ============================================================================

@dataclass
class EffectConfig:
    """Configuration for visual effects."""
    
    # Point light effect
    point_light_enabled: bool = False
    point_light_intensity: float = 3.0  # [0.1, 5.0] - default reduced to avoid flickering
    point_light_radius: float = 0.1  # [0.1, 1.0], fraction of image size
    point_light_falloff: float = 2.0  # [1.0, 4.0], higher = sharper falloff
    
    # Multiple random lights
    point_light_count: int = 1  # [1, 5]
    point_light_color_names: list[str] = field(default_factory=lambda: ["warm_white"])


# ============================================================================
# Light Effect Functions
# ============================================================================

def create_point_light_mask(
    H: int,
    W: int,
    center_x: float,
    center_y: float,
    radius: float,
    falloff: float,
) -> jnp.ndarray:
    """
    Create radial light falloff mask from a point.
    
    Args:
        H: Image height
        W: Image width
        center_x: Light center x coordinate [0, 1]
        center_y: Light center y coordinate [0, 1]
        radius: Light radius as fraction of image size [0, 1]
        falloff: Falloff exponent [1.0, 4.0]
    
    Returns:
        mask: (H, W) float32 in [0, 1], 1 at center, fades to 0 at edges
    """
    # Convert normalized coordinates to pixel coordinates
    cx = center_x * W
    cy = center_y * H
    
    # Create coordinate grids
    y, x = jnp.ogrid[:H, :W]
    y = y.astype(jnp.float32)
    x = x.astype(jnp.float32)
    
    # Distance from light center
    dx = x - cx
    dy = y - cy
    dist = jnp.sqrt(dx * dx + dy * dy)
    
    # Normalize by radius (in pixels)
    max_radius = jnp.sqrt(H * H + W * W) * radius
    normalized_dist = dist / max_radius
    
    # Apply falloff (inverse square law with adjustable exponent)
    mask = 1.0 / (1.0 + normalized_dist ** falloff)
    
    # Clip to [0, 1]
    mask = jnp.clip(mask, 0.0, 1.0)
    
    return mask


def apply_point_light(
    img: jnp.ndarray,
    center_x: float,
    center_y: float,
    color: tuple[int, int, int],
    intensity: float,
    radius: float,
    falloff: float,
) -> jnp.ndarray:
    """
    Apply point light effect to image.
    
    Args:
        img: (H, W, 3) uint8 image
        center_x: Light x position [0, 1]
        center_y: Light y position [0, 1]
        color: Light color RGB
        intensity: Light intensity [0.5, 2.0]
        radius: Light radius [0.1, 1.0]
        falloff: Light falloff [1.0, 4.0]
    
    Returns:
        img: (H, W, 3) uint8 image with light applied
    """
    H, W = img.shape[:2]
    img_f = img.astype(jnp.float32)
    
    # Create light mask
    mask = create_point_light_mask(H, W, center_x, center_y, radius, falloff)
    
    # Light color and intensity
    light_color = jnp.array(color, dtype=jnp.float32) * intensity
    
    # Apply light as additive blend weighted by mask
    light_contribution = mask[..., None] * light_color[None, None, :]
    
    # Add light to image
    img_lit = img_f + light_contribution
    
    return jnp.clip(img_lit, 0.0, 255.0).astype(jnp.uint8)


def apply_multiple_point_lights(
    img: jnp.ndarray,
    positions: list[tuple[float, float]],
    colors: list[tuple[int, int, int]],
    intensity: float,
    radius: float,
    falloff: float,
) -> jnp.ndarray:
    """
    Apply multiple point lights to image.
    
    Args:
        img: (H, W, 3) uint8 image
        positions: List of (x, y) positions in [0, 1]
        colors: List of RGB tuples
        intensity: Light intensity
        radius: Light radius
        falloff: Light falloff
    
    Returns:
        img: (H, W, 3) uint8 image with lights applied
    """
    H, W = img.shape[:2]
    img_f = img.astype(jnp.float32)
    
    # Accumulate light contributions from all sources
    total_light = jnp.zeros((H, W, 3), dtype=jnp.float32)
    
    for (cx, cy), color in zip(positions, colors):
        mask = create_point_light_mask(H, W, cx, cy, radius, falloff)
        light_color = jnp.array(color, dtype=jnp.float32) * intensity
        light_contribution = mask[..., None] * light_color[None, None, :]
        total_light = total_light + light_contribution
    
    # Add accumulated light to image
    img_lit = img_f + total_light
    
    return jnp.clip(img_lit, 0.0, 255.0).astype(jnp.uint8)


def select_light_positions_and_colors(
    key: jax.Array,
    n_lights: int,
    color_names: list[str],
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    """
    Generate random light positions and select colors.
    
    Args:
        key: JAX PRNG key
        n_lights: Number of lights
        color_names: List of color names to sample from
    
    Returns:
        positions: List of (x, y) tuples in [0, 1]
        colors: List of RGB tuples
    """
    positions = []
    colors = []
    
    for i in range(n_lights):
        key, subkey = jax.random.split(key)
        
        # Random position
        pos = jax.random.uniform(subkey, shape=(2,), minval=0.0, maxval=1.0)
        x, y = float(pos[0]), float(pos[1])
        positions.append((x, y))
        
        # Random color from list
        if color_names:
            color_idx = i % len(color_names)
            color_name = color_names[color_idx]
            if color_name in LIGHT_COLORS:
                colors.append(LIGHT_COLORS[color_name])
            else:
                colors.append((255, 255, 255))
        else:
            colors.append((255, 255, 255))
    
    return positions, colors


def apply_point_light_jit_compatible(
    img: jnp.ndarray,
    key: jax.Array,
    config: EffectConfig,
) -> jnp.ndarray:
    """
    Apply point light effect (JIT-compatible version).
    
    This version uses fixed position if provided, or derives position from key.
    
    Args:
        img: (H, W, 3) uint8 image
        key: JAX PRNG key for random position
        config: EffectConfig
    
    Returns:
        img: (H, W, 3) uint8 image with light applied
    """
    if not config.point_light_enabled:
        return img
    
    H, W = img.shape[:2]
    img_f = img.astype(jnp.float32)
    
    # Accumulate light from all sources
    total_light = jnp.zeros((H, W, 3), dtype=jnp.float32)
    
    # Fixed maximum number of lights for JIT compatibility
    max_lights = 5
    n_lights = min(config.point_light_count, max_lights)
    
    # Generate all random positions at once
    key, subkey = jax.random.split(key)
    positions = jax.random.uniform(subkey, shape=(max_lights, 2), minval=0.1, maxval=0.9)
    
    for i in range(max_lights):
        # Only apply light if i < n_lights
        apply_this_light = i < n_lights
        
        cx, cy = positions[i, 0], positions[i, 1]
        
        # Select color (cycle through color list)
        color_idx = i % len(config.point_light_color_names)
        color_name = config.point_light_color_names[color_idx]
        if color_name in LIGHT_COLORS:
            color = jnp.array(LIGHT_COLORS[color_name], dtype=jnp.float32)
        else:
            color = jnp.array((255.0, 255.0, 255.0), dtype=jnp.float32)
        
        # Create light mask
        mask = create_point_light_mask(
            H, W, cx, cy, 
            config.point_light_radius, 
            config.point_light_falloff
        )
        
        # Light contribution
        light_color = color * config.point_light_intensity
        light_contribution = mask[..., None] * light_color[None, None, :]
        
        # Only add if within count
        light_contribution = jnp.where(apply_this_light, light_contribution, 0.0)
        total_light = total_light + light_contribution
    
    # Add accumulated light
    img_lit = img_f + total_light
    return jnp.clip(img_lit, 0.0, 255.0).astype(jnp.uint8)


# ============================================================================
# Main Effect Application
# ============================================================================

def generate_light_positions(
    key: jax.Array,
    n_lights: int,
) -> jnp.ndarray:
    """
    Generate random light positions (for use in reset, not render).

    Args:
        key: JAX PRNG key
        n_lights: Number of lights

    Returns:
        positions: (n_lights, 2) array of (x, y) positions in [0, 1]
    """
    max_lights = 5
    n_lights = min(n_lights, max_lights)
    positions = jax.random.uniform(key, shape=(max_lights, 2), minval=0.15, maxval=0.85)
    return positions


def generate_light_properties(
    key: jax.Array,
    n_lights: int,
    config: EffectConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate light positions, radii, and intensities for episode initialization.

    Args:
        key: JAX PRNG key
        n_lights: Number of lights
        config: EffectConfig with light parameters

    Returns:
        positions: (max_lights, 2) array of (x, y) positions in [0, 1]
        radii: (max_lights,) array of light radii
        intensities: (max_lights,) array of light intensities
    """
    max_lights = 5
    n_lights = min(n_lights, max_lights)

    # Generate positions
    key, subkey = jax.random.split(key)
    positions = jax.random.uniform(subkey, shape=(max_lights, 2), minval=0.15, maxval=0.85)

    # Generate radii (with some variation around config value)
    key, subkey = jax.random.split(key)
    radius_variation = jax.random.uniform(subkey, shape=(max_lights,), minval=0.8, maxval=1.2)
    radii = jnp.full(max_lights, config.point_light_radius) * radius_variation

    # Generate intensities (with some variation around config value)
    key, subkey = jax.random.split(key)
    intensity_variation = jax.random.uniform(subkey, shape=(max_lights,), minval=0.8, maxval=1.2)
    intensities = jnp.full(max_lights, config.point_light_intensity) * intensity_variation

    return positions, radii, intensities


def apply_effects(
    img: jnp.ndarray,
    config: EffectConfig,
    light_positions: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Apply all configured effects to image.
    
    Args:
        img: (H, W, 3) uint8 observation
        config: EffectConfig
        light_positions: (max_lights, 2) array of pre-generated light positions
    
    Returns:
        img: (H, W, 3) uint8 with effects applied
    """
    # Apply point light effect
    if config.point_light_enabled and light_positions is not None:
        img = apply_point_light_with_positions(img, light_positions, config)
    
    return img


def apply_point_light_with_positions(
    img: jnp.ndarray,
    positions: jnp.ndarray,
    config: EffectConfig,
) -> jnp.ndarray:
    """
    Apply point light effect with pre-generated positions (vectorized for speed).
    
    Args:
        img: (H, W, 3) uint8 image
        positions: (max_lights, 2) array of light positions
        config: EffectConfig
    
    Returns:
        img: (H, W, 3) uint8 image with light applied
    """
    H, W = img.shape[:2]
    img_f = img.astype(jnp.float32)
    
    # Fixed maximum number of lights for JIT compatibility
    max_lights = 5
    n_lights = min(config.point_light_count, max_lights)
    
    # Pre-compute color branches for JIT compatibility
    def get_color_for_idx(idx):
        """Select color using lax.switch."""
        color_branches = []
        for color_name in config.point_light_color_names:
            if color_name in LIGHT_COLORS:
                color_rgb = LIGHT_COLORS[color_name]
            else:
                color_rgb = (255.0, 255.0, 255.0)
            color_branches.append(lambda c=color_rgb: jnp.array(c, dtype=jnp.float32))
        
        # Cycle through colors if idx >= len(colors)
        idx_mod = idx % len(color_branches)
        return jax.lax.switch(idx_mod, color_branches)
    
    # Vectorized light computation
    def compute_single_light(i):
        apply_this_light = i < n_lights
        
        cx, cy = positions[i, 0], positions[i, 1]
        
        # Select color using switch (JIT-compatible)
        color = get_color_for_idx(i)
        
        # Create light mask
        mask = create_point_light_mask(
            H, W, cx, cy, 
            config.point_light_radius, 
            config.point_light_falloff
        )
        
        # Light contribution
        light_color = color * config.point_light_intensity
        light_contribution = mask[..., None] * light_color[None, None, :]
        
        # Only add if within count
        return jnp.where(apply_this_light, light_contribution, 0.0)
    
    # Vectorize across all lights (unrolled by compiler for small max_lights)
    light_indices = jnp.arange(max_lights)
    all_lights = jax.vmap(compute_single_light)(light_indices)  # (max_lights, H, W, 3)
    total_light = jnp.sum(all_lights, axis=0)  # (H, W, 3)
    
    # Add accumulated light
    img_lit = img_f + total_light
    return jnp.clip(img_lit, 0.0, 255.0).astype(jnp.uint8)


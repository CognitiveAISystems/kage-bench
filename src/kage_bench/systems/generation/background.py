"""
Background image rendering for JAX platformer.

Supports:
- Black background (default, no overhead)
- Image background with tiling and parallax scrolling
- Noise background (random white noise)
- Color backgrounds (solid colors, optionally randomized per episode)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp


# Predefined color palette (RGB uint8 values)
COLOR_PALETTE = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (220, 20, 60),
    "orange": (255, 140, 0),
    "yellow": (255, 215, 0),
    "green": (34, 139, 34),
    "cyan": (0, 206, 209),
    "blue": (30, 144, 255),
    "purple": (138, 43, 226),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
    "gray": (128, 128, 128),
    "lime": (50, 205, 50),
    "teal": (0, 128, 128),
    "indigo": (75, 0, 130),
    "magenta": (255, 0, 255),
}


@dataclass
class BackgroundConfig:
    """Configuration for background rendering."""

    mode: str = "black"  # "black", "image", "noise", or "color"
    image_path: Optional[str] = None
    image_dir: Optional[str] = None  # Path to directory containing backgrounds
    image_paths: list[str] = field(default_factory=list)  # For multiple images switching
    parallax_factor: float = 0.5  # <1.0 = slower than camera, >1.0 = faster
    tile_horizontal: bool = True  # Tile image horizontally for infinite scrolling
    
    # For color mode: list of color names to sample from, or single color name
    color_names: list[str] = field(default_factory=lambda: ["black"])

    # Frequency of background switching (0.0 = no switching, 1.0 = every step)
    switch_frequency: float = 0.0


def generate_noise_background(key: jax.Array, H: int, W: int) -> jnp.ndarray:
    """
    Generate white noise background.

    Args:
        key: JAX PRNG key
        H: Height in pixels
        W: Width in pixels

    Returns:
        noise_image: (H, W, 3) uint8 random noise
    """
    noise = jax.random.uniform(key, shape=(H, W, 3), minval=0.0, maxval=256.0)
    return noise.astype(jnp.uint8)


def generate_color_background(color_name: str, H: int, W: int) -> jnp.ndarray:
    """
    Generate solid color background.

    Args:
        color_name: Name of color from COLOR_PALETTE
        H: Height in pixels
        W: Width in pixels

    Returns:
        color_image: (H, W, 3) uint8 solid color
    """
    if color_name not in COLOR_PALETTE:
        raise ValueError(
            f"Unknown color: {color_name}. "
            f"Available colors: {list(COLOR_PALETTE.keys())}"
        )
    
    rgb = COLOR_PALETTE[color_name]
    color_array = jnp.array(rgb, dtype=jnp.uint8)
    return jnp.broadcast_to(color_array[None, None, :], (H, W, 3))


def select_color_by_index(idx: jax.Array, color_names: list[str], H: int, W: int) -> jnp.ndarray:
    """
    Select color background by index (JAX-compatible).

    Args:
        idx: Integer index into color_names (JAX array)
        color_names: List of color names
        H: Height in pixels
        W: Width in pixels

    Returns:
        color_image: (H, W, 3) uint8 solid color
    """
    if not color_names:
        return generate_color_background("black", H, W)
    if len(color_names) == 1:
        return generate_color_background(color_names[0], H, W)
    
    # Create branch functions for jax.lax.switch (expects callables)
    def make_branch(color_name):
        return lambda: generate_color_background(color_name, H, W)
    
    branches = [make_branch(c) for c in color_names]
    
    # Use jax.lax.switch to select based on index (JIT-compatible)
    return jax.lax.switch(idx, branches)


def load_background_image(
    image_path: str,
    target_height: int,
    target_width: Optional[int] = None,
) -> jnp.ndarray:
    """
    Load and preprocess background image for JAX rendering.

    Args:
        image_path: Path to image file (jpeg, png, etc.)
        target_height: Target height in pixels (will resize maintaining aspect ratio if width not set)
        target_width: Optional target width (if set, will force exact size)

    Returns:
        image_array: (H, W, 3) uint8 JAX array
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "PIL/Pillow is required for image backgrounds. "
            "Install via `pip install Pillow`."
        ) from e

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Background image not found: {image_path}")

    # Load image
    img = Image.open(img_path)

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    if target_width is not None:
        # Resize to exact dimensions
        img = img.resize((target_width, target_height), Image.LANCZOS)
    else:
        # Resize to target height, maintaining aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        img = img.resize((new_width, target_height), Image.LANCZOS)

    # Convert to numpy array then JAX array
    import numpy as np

    img_array = np.array(img, dtype=np.uint8)
    return jnp.array(img_array)


def apply_background(
    base_img: jnp.ndarray,
    bg_image: Optional[jnp.ndarray],
    camera_x: jax.Array,
    config: BackgroundConfig,
) -> jnp.ndarray:
    """
    Apply background to base image (black).

    Args:
        base_img: (H, W, 3) uint8 image (currently black)
        bg_image: (H_bg, W_bg, 3) uint8 background image, or None for black
        camera_x: Camera x position in world coordinates
        config: Background configuration

    Returns:
        img_with_bg: (H, W, 3) uint8 image with background applied
    """
    if config.mode == "black" or bg_image is None:
        # No background, return as-is
        return base_img

    H, W = base_img.shape[:2]
    bg_H, bg_W = bg_image.shape[:2]

    # For static backgrounds (noise, color): no camera offset, no tiling
    if config.mode in ("noise", "color"):
        # Background size should match screen size exactly
        return bg_image  # (H, W, 3)

    # For image mode: parallax scrolling with tiling
    if config.mode == "image":
        # Compute parallax offset
        parallax_offset = jnp.round(camera_x * config.parallax_factor).astype(jnp.int32)

        # Generate screen pixel coordinates
        ys = jnp.arange(H, dtype=jnp.int32)[:, None]  # (H, 1)
        xs = jnp.arange(W, dtype=jnp.int32)[None, :]  # (1, W)

        # Map screen coords to background image coords
        bg_xs = xs + parallax_offset

        if config.tile_horizontal:
            # Wrap horizontally for infinite tiling
            bg_xs = bg_xs % bg_W
        else:
            # Clamp to image bounds
            bg_xs = jnp.clip(bg_xs, 0, bg_W - 1)

        bg_ys = jnp.clip(ys, 0, bg_H - 1)

        # Sample background image
        return bg_image[bg_ys, bg_xs]  # (H, W, 3)

    else:
        raise ValueError(f"Unknown background mode: {config.mode}")

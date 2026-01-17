"""Character sprite system for JAX platformer.

Supports loading and animating character sprites from directories.
All operations are JIT-compatible.

Functions
---------
load_sprites_from_directory
    Load .png sprites from directory
render_sprite
    Render single sprite with alpha blending
render_sprite_from_list
    Select and render sprite by index (JIT-compatible)
render_character_rect
    Render character as colored rectangle (fallback)
update_animation_state
    Update sprite animation state
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp


@dataclass
class CharacterConfig:
    """Configuration for character sprites or geometric shapes."""
    
    # Character bounding box (pixels)
    width: int = 16  # Width in pixels
    height: int = 24  # Height in pixels
    
    # Sprite configuration
    use_sprites: bool = False
    sprite_dir: Optional[str] = None  # Path to directory containing sprite subdirectories (all subdirs used)
    sprite_paths: list[str] = None  # List of directories for multiple skins (random selection)
    sprite_path: Optional[str] = None  # Path to single sprite directory
    
    # Animation settings
    enable_animation: bool = True  # Enable sprite animation (if False, use only first frame)
    animation_fps: float = 10.0  # Frames per second for sprite animation
    idle_sprite_idx: int = 0  # Which sprite to use when idle (typically first)
    
    # Geometric shape configuration (alternative to sprites)
    use_shape: bool = False  # Use geometric shape instead of sprite
    shape_types: list[str] | str = "circle"  # Type(s) of shape: single or list for random selection
    shape_colors: list[str] | str = "red"  # Color(s) for the shape: single or list for random selection
    shape_rotate: bool = False  # Whether to rotate the shape
    shape_rotation_speed: float = 2.0  # Rotation speed in degrees per frame
    
    # Fallback (if neither sprites nor shapes are used)
    fallback_color: tuple[int, int, int] = (255, 0, 0)  # Red


def load_sprites_from_directory(sprite_dir: str) -> list[jnp.ndarray]:
    """
    Load all .png sprites from directory.
    
    Args:
        sprite_dir: Directory containing .png files
    
    Returns:
        List of JAX arrays (H, W, 4) uint8
    """
    path = Path(sprite_dir)
    if not path.exists():
        # Return dummy sprite if not found (prevents crash, but warns)
        print(f"Warning: Sprite directory not found: {sprite_dir}")
        return [jnp.zeros((16, 16, 4), dtype=jnp.uint8)]
    
    image_files = sorted(list(path.glob("**/*.png")))
    if not image_files:
        print(f"Warning: No .png files found in {sprite_dir}")
        return [jnp.zeros((16, 16, 4), dtype=jnp.uint8)]
    
    sprites = []
    for img_path in image_files:
        # Load with PIL
        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGBA")
            img_arr = jnp.array(img, dtype=jnp.uint8)
            sprites.append(img_arr)
        except Exception as e:
            print(f"Error loading sprite {img_path}: {e}")
            
    return sprites


def load_character_sprite_sets(sprite_dirs: list[str]) -> list[list[jnp.ndarray]]:
    """
    Load sprite sets from multiple directories.
    """
    sprite_sets = []
    for d in sprite_dirs:
        sprites = load_sprites_from_directory(d)
        sprite_sets.append(sprites)
    return sprite_sets


def render_sprite(
    img: jnp.ndarray,
    sprite: jnp.ndarray,
    center_x: int,
    center_y: int,
    char_width: int,
    char_height: int,
) -> jnp.ndarray:
    """
    Render sprite onto image with alpha blending.
    
    Args:
        img: (H, W, 3) uint8 background image
        sprite: (sprite_H, sprite_W, 4) uint8 sprite with alpha channel
        center_x: Center x position in image
        center_y: Center y position in image
        char_width: Character width (for positioning)
        char_height: Character height (for positioning)
    
    Returns:
        img: (H, W, 3) uint8 image with sprite rendered
    """
    H, W = img.shape[:2]
    sprite_h, sprite_w = sprite.shape[:2]
    
    # Calculate sprite position (top-left corner)
    half_h = char_height // 2
    
    # Align sprite bottom to the bottom of the logical bounding box (center_y + half_h)
    x_start = center_x - (sprite_w // 2)
    y_start = (center_y + half_h) - sprite_h
    
    # Fast path for off-screen sprites
    # (Optional, as JAX might still execute the code, but good for clarity)
    visible = (x_start < W) & (x_start + sprite_w > 0) & (y_start < H) & (y_start + sprite_h > 0)
    
    # To handle off-screen sprites without complex masking, we pad the image,
    # perform the blend on a slice, and then crop back.
    # We pad by sprite dimensions to ensure any partially visible sprite is covered.
    pad_h, pad_w = sprite_h, sprite_w
    padded_img = jnp.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    
    # Adjust start coordinates for padding
    py_start = y_start + pad_h
    px_start = x_start + pad_w
    
    # Extract the background patch where the sprite will be drawn
    # dynamic_slice(operand, start_indices, slice_sizes)
    bg_patch = jax.lax.dynamic_slice(padded_img, (py_start, px_start, 0), (sprite_h, sprite_w, 3))
    
    # Extract alpha channel and RGB
    sprite_rgb = sprite[..., :3].astype(jnp.float32)
    sprite_alpha = sprite[..., 3:4].astype(jnp.float32) / 255.0
    bg_patch_f = bg_patch.astype(jnp.float32)
    
    # Alpha blending: result = alpha * sprite + (1 - alpha) * background
    blended_patch = sprite_alpha * sprite_rgb + (1.0 - sprite_alpha) * bg_patch_f
    blended_patch_u8 = jnp.clip(blended_patch, 0.0, 255.0).astype(jnp.uint8)
    
    # Update the padded image with the blended patch
    updated_padded = jax.lax.dynamic_update_slice(padded_img, blended_patch_u8, (py_start, px_start, 0))
    
    # Crop back to original size
    res = updated_padded[pad_h:pad_h+H, pad_w:pad_w+W]
    
    # If not visible, return original image (important for JAX branches)
    return jnp.where(visible, res, img)


def render_sprite_from_list(
    img: jnp.ndarray,
    sprites: list[jnp.ndarray],
    sprite_idx: int,
    center_x: int,
    center_y: int,
    char_width: int,
    char_height: int,
) -> jnp.ndarray:
    """
    Render sprite from list by index (JIT-compatible).
    
    Args:
        img: (H, W, 3) uint8 background image
        sprites: List of sprite arrays
        sprite_idx: Index of sprite to render
        center_x: Center x position
        center_y: Center y position
        char_width: Character width
        char_height: Character height
    
    Returns:
        img: (H, W, 3) uint8 image with sprite
    """
    if not sprites:
        return img
    
    # Use jax.lax.switch for JIT compatibility
    def make_render_branch(sprite):
        return lambda: render_sprite(img, sprite, center_x, center_y, char_width, char_height)
    
    branches = [make_render_branch(s) for s in sprites]
    
    # Clamp index to valid range
    sprite_idx = jnp.clip(sprite_idx, 0, len(sprites) - 1)
    
    return jax.lax.switch(sprite_idx, branches)


def render_character_rect(
    img: jnp.ndarray,
    center_x: int | jnp.ndarray,
    center_y: int | jnp.ndarray,
    char_width: int | jnp.ndarray,
    char_height: int | jnp.ndarray,
    color: tuple[int, int, int] | jnp.ndarray,
) -> jnp.ndarray:
    """
    Render character as simple colored rectangle (fallback).
    
    Args:
        img: (H, W, 3) uint8 image
        center_x: Center x position
        center_y: Center y position
        char_width: Character width
        char_height: Character height
        color: RGB color tuple or JAX array
    
    Returns:
        img: (H, W, 3) uint8 image with rectangle
    """
    H, W = img.shape[:2]
    
    half_w = char_width // 2
    half_h = char_height // 2
    
    # Screen coordinates
    ys_screen = jnp.arange(H, dtype=jnp.int32)[:, None]
    xs_screen = jnp.arange(W, dtype=jnp.int32)[None, :]
    
    # Character mask
    char_mask = (
        (xs_screen >= center_x - half_w) & (xs_screen <= center_x + half_w) &
        (ys_screen >= center_y - half_h) & (ys_screen <= center_y + half_h)
    )
    
    # Convert color to JAX array if it's a tuple
    if isinstance(color, tuple):
        char_color = jnp.array(color, dtype=jnp.uint8)
    else:
        char_color = color  # Already a JAX array
    
    img = jnp.where(char_mask[:, :, None], char_color[None, None, :], img)
    
    return img


def update_animation_state(
    current_idx: int,
    animation_timer: float,
    dt: float,
    fps: float,
    n_sprites: int,
    is_moving: bool,
    idle_idx: int,
    enable_animation: bool = True,
) -> tuple[int, float]:
    """
    Update sprite animation state (JIT-compatible).
    
    Args:
        current_idx: Current sprite index
        animation_timer: Current timer value
        dt: Time delta (typically 1/60 for 60fps game)
        fps: Animation FPS
        n_sprites: Total number of sprites
        is_moving: Whether character is moving
        idle_idx: Index to use when idle
    
    Returns:
        next_idx: Next sprite index
        next_timer: Next timer value
    """
    # Moving: advance animation
    next_timer_moving = animation_timer + dt
    frame_duration = 1.0 / fps
    
    # Check if we should advance frame
    should_advance = next_timer_moving >= frame_duration
    next_idx_moving = jnp.where(
        should_advance,
        (current_idx + 1) % n_sprites,
        current_idx
    )
    next_timer_moving = jnp.where(
        should_advance,
        next_timer_moving - frame_duration,
        next_timer_moving
    )
    
    # If animation is disabled, always use idle sprite
    next_idx_anim = jnp.where(is_moving, next_idx_moving, jnp.int32(idle_idx))
    next_timer_anim = jnp.where(is_moving, next_timer_moving, jnp.float32(0.0))

    # Choose between animated and static sprite
    next_idx = jnp.where(enable_animation, next_idx_anim, jnp.int32(idle_idx))
    next_timer = jnp.where(enable_animation, next_timer_anim, jnp.float32(0.0))

    return next_idx, next_timer


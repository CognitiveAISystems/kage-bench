from __future__ import annotations

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class CameraConfig:
    """Camera deadzone configuration.
    
    The camera follows the hero when the hero moves outside a rectangular frame.
    Frame is defined by left/right margins from screen edges.
    
    Attributes
    ----------
    left_margin : int
        Pixels from left edge of screen (default: 40)
    right_margin : int
        Pixels from right edge of screen (default: 40)
    """
    left_margin: int = 40    # pixels from left edge of screen
    right_margin: int = 40   # pixels from right edge of screen
    

def update_camera(
    camera_x: jnp.ndarray,
    hero_x: jnp.ndarray,
    screen_width: int,
    cfg: CameraConfig,
) -> jnp.ndarray:
    """
    Update camera x-position based on hero position and deadzone.
    
    Args:
        camera_x: Current camera x position (world space), float32 scalar
        hero_x: Hero x position (world space), float32 scalar
        screen_width: Width of screen in pixels
        cfg: Camera configuration
        
    Returns:
        Updated camera_x (world space), float32 scalar
        
    Logic:
        - Hero screen position = hero_x - camera_x
        - Left boundary = left_margin
        - Right boundary = screen_width - right_margin
        - If hero_screen_x < left_margin, push camera left
        - If hero_screen_x > right_boundary, push camera right
    """
    # Hero position in screen space.
    hero_screen_x = hero_x - camera_x
    
    # Deadzone boundaries in screen space.
    left_bound = jnp.float32(cfg.left_margin)
    right_bound = jnp.float32(screen_width - cfg.right_margin)
    
    # Compute how far hero is outside deadzone.
    left_overshoot = left_bound - hero_screen_x   # positive if hero is left of left_bound
    right_overshoot = hero_screen_x - right_bound  # positive if hero is right of right_bound
    
    # Move camera to keep hero within deadzone.
    delta_x = jnp.where(
        left_overshoot > 0,
        -left_overshoot,  # move camera left (decrease camera_x)
        jnp.where(
            right_overshoot > 0,
            right_overshoot,  # move camera right (increase camera_x)
            0.0
        )
    )
    
    new_camera_x = camera_x + delta_x
    
    return new_camera_x


def world_to_screen(world_x: jnp.ndarray, camera_x: jnp.ndarray) -> jnp.ndarray:
    """Convert world x-coordinate to screen x-coordinate."""
    return world_x - camera_x


def screen_to_world(screen_x: jnp.ndarray, camera_x: jnp.ndarray) -> jnp.ndarray:
    """Convert screen x-coordinate to world x-coordinate."""
    return screen_x + camera_x


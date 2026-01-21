"""Visual distractors system for JAX platformer.

Visual distractors are geometric shapes that move around the screen
to increase visual complexity without affecting gameplay.

Functions
---------
spawn_distractors
    Spawn distractors at random positions with random properties
update_distractors
    Update distractor positions and rotation angles
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import jax
import jax.numpy as jnp


@dataclass
class DistractorConfig:
    """Configuration for visual distractors."""
    
    # Enable/disable distractors
    enabled: bool = False
    
    # Number of distractors
    count: int = 3
    
    # Shape configuration
    shape_types: list[str] | str = "circle"  # Single or list for random selection
    shape_colors: list[str] | str = "red"  # Single or list for random selection
    
    # Movement
    can_move: bool = True  # Whether distractors move around
    min_speed: float = 0.5  # Minimum movement speed (pixels per step)
    max_speed: float = 2.0  # Maximum movement speed (pixels per step)
    
    # Rotation
    can_rotate: bool = True  # Whether distractors rotate
    min_rotation_speed: float = -10.0  # Min rotation speed (degrees per step)
    max_rotation_speed: float = 10.0  # Max rotation speed (degrees per step)
    
    # Size
    min_size: int = 3  # Minimum size
    max_size: int = 8  # Maximum size


def spawn_distractors(
    key: jax.Array,
    config: DistractorConfig,
    screen_width: int,
    screen_height: int,
    num_shape_types: int,
    num_color_types: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Spawn distractors at random positions with random properties.
    
    Args:
        key: PRNG key
        config: Distractor configuration
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        num_shape_types: Number of available shape types
        num_color_types: Number of available color types
    
    Returns:
        x_positions: X positions (count,)
        y_positions: Y positions (count,)
        x_velocities: X velocities (count,)
        y_velocities: Y velocities (count,)
        angles: Rotation angles (count,)
        rotation_speeds: Rotation speeds (count,)
        shape_type_indices: Shape type indices (count,)
        color_indices: Color indices (count,)
        sizes: Sizes (count,)
    """
    count = config.count
    
    # Random positions (with margin to avoid edges)
    margin = 15
    key, k_x = jax.random.split(key)
    x_positions = jax.random.uniform(k_x, (count,), minval=margin, maxval=screen_width - margin)
    
    key, k_y = jax.random.split(key)
    y_positions = jax.random.uniform(k_y, (count,), minval=margin, maxval=screen_height - margin)
    
    # Random velocities
    if config.can_move:
        key, k_vx = jax.random.split(key)
        x_velocities = jax.random.uniform(
            k_vx, (count,), 
            minval=-config.max_speed, 
            maxval=config.max_speed
        )
        
        key, k_vy = jax.random.split(key)
        y_velocities = jax.random.uniform(
            k_vy, (count,), 
            minval=-config.max_speed, 
            maxval=config.max_speed
        )
    else:
        x_velocities = jnp.zeros(count, dtype=jnp.float32)
        y_velocities = jnp.zeros(count, dtype=jnp.float32)
    
    # Random rotation angles
    key, k_angle = jax.random.split(key)
    angles = jax.random.uniform(k_angle, (count,), minval=0.0, maxval=360.0)
    
    # Random rotation speeds
    if config.can_rotate:
        key, k_rot_speed = jax.random.split(key)
        rotation_speeds = jax.random.uniform(
            k_rot_speed, (count,),
            minval=config.min_rotation_speed,
            maxval=config.max_rotation_speed
        )
    else:
        rotation_speeds = jnp.zeros(count, dtype=jnp.float32)
    
    # Random shape types
    key, k_shape = jax.random.split(key)
    if num_shape_types > 0:
        shape_type_indices = jax.random.randint(k_shape, (count,), 0, num_shape_types)
    else:
        shape_type_indices = jnp.zeros(count, dtype=jnp.int32)
    
    # Random colors
    key, k_color = jax.random.split(key)
    if num_color_types > 0:
        color_indices = jax.random.randint(k_color, (count,), 0, num_color_types)
    else:
        color_indices = jnp.zeros(count, dtype=jnp.int32)
    
    # Random sizes
    key, k_size = jax.random.split(key)
    sizes = jax.random.randint(k_size, (count,), config.min_size, config.max_size + 1)
    
    return (
        x_positions.astype(jnp.float32),
        y_positions.astype(jnp.float32),
        x_velocities.astype(jnp.float32),
        y_velocities.astype(jnp.float32),
        angles.astype(jnp.float32),
        rotation_speeds.astype(jnp.float32),
        shape_type_indices.astype(jnp.int32),
        color_indices.astype(jnp.int32),
        sizes.astype(jnp.int32),
    )


def update_distractors(
    x_positions: jnp.ndarray,
    y_positions: jnp.ndarray,
    x_velocities: jnp.ndarray,
    y_velocities: jnp.ndarray,
    angles: jnp.ndarray,
    rotation_speeds: jnp.ndarray,
    screen_width: int,
    screen_height: int,
    margin: int = 15,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Update distractor positions and angles.
    
    Args:
        x_positions: Current X positions (count,)
        y_positions: Current Y positions (count,)
        x_velocities: X velocities (count,)
        y_velocities: Y velocities (count,)
        angles: Current rotation angles (count,)
        rotation_speeds: Rotation speeds (count,)
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        margin: Edge margin for bouncing
    
    Returns:
        new_x_positions: Updated X positions
        new_y_positions: Updated Y positions
        new_x_velocities: Updated X velocities (with bouncing)
        new_y_velocities: Updated Y velocities (with bouncing)
        new_angles: Updated rotation angles
    """
    # Update positions
    new_x = x_positions + x_velocities
    new_y = y_positions + y_velocities
    
    # Update velocities (bounce off walls)
    new_vx = jnp.where(
        (new_x <= margin) | (new_x >= screen_width - margin),
        -x_velocities,
        x_velocities
    )
    
    new_vy = jnp.where(
        (new_y <= margin) | (new_y >= screen_height - margin),
        -y_velocities,
        y_velocities
    )
    
    # Clamp positions to screen bounds
    new_x = jnp.clip(new_x, margin, screen_width - margin)
    new_y = jnp.clip(new_y, margin, screen_height - margin)
    
    # Update rotation angles
    new_angles = (angles + rotation_speeds) % 360.0
    
    return new_x, new_y, new_vx, new_vy, new_angles


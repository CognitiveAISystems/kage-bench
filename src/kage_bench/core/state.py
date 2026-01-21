"""State representations for JAX platformer environment.

All state is represented using Flax struct dataclasses for:
- Immutability
- JAX pytree compatibility
- Efficient JIT compilation
- Clear type signatures
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EnvState:
    """Complete environment state (immutable JAX pytree).
    
    All per-step state required for simulation and rendering.
    State is immutable; step() returns new EnvState.
    
    Attributes
    ----------
    x : jnp.ndarray
        Agent x position in world coordinates, shape (), dtype float32
    y : jnp.ndarray
        Agent y position in world coordinates, shape (), dtype float32
    vx : jnp.ndarray
        Agent x velocity in px/step, shape (), dtype float32
    vy : jnp.ndarray
        Agent y velocity in px/step, shape (), dtype float32
    grounded : jnp.ndarray
        Whether agent is on ground, shape (), dtype bool
    coyote_timer : jnp.ndarray
        Frames since leaving ground (for coyote time), shape (), dtype int32
    camera_x : jnp.ndarray
        Camera x position in world coordinates, shape (), dtype float32
    t : jnp.ndarray
        Current timestep, shape (), dtype int32
    x_max : jnp.ndarray
        Maximum x position reached (for reward), shape (), dtype float32
    episode_return : jnp.ndarray
        Cumulative return, shape (), dtype float32
    jump_count : jnp.ndarray
        Number of jumps taken, shape (), dtype int32
    initial_x : jnp.ndarray
        Initial x position at reset, shape (), dtype float32
    success_once : jnp.ndarray
        Whether success was achieved at least once, shape (), dtype bool
    vx_mean : jnp.ndarray
        Running mean of vx over steps, shape (), dtype float32
    vy_mean : jnp.ndarray
        Running mean of vy over steps, shape (), dtype float32
    layout_solid_mask : jnp.ndarray
        Solid collision mask, shape (H, world_width), dtype bool
    layout_ground_top : jnp.ndarray
        Ground top y-coordinates, shape (world_width,), dtype int32
    layout_ground_bottom : jnp.ndarray
        Ground bottom y-coordinates, shape (world_width,), dtype int32
    layout_color : jnp.ndarray
        Layout color RGB, shape (3,), dtype uint8
    bg_image : Optional[jnp.ndarray]
        Background image if using image mode, shape (H, bg_width, 3), dtype uint8
    filter_key : jnp.ndarray
        PRNG key for stochastic filters, shape (2,), dtype uint32
    selected_filter_idx : jnp.ndarray
        Selected filter preset index, shape (), dtype int32
    light_positions : jnp.ndarray
        Point light positions (normalized), shape (max_lights, 2), dtype float32
    light_radii : jnp.ndarray
        Point light radii, shape (max_lights,), dtype float32
    light_intensities : jnp.ndarray
        Point light intensities, shape (max_lights,), dtype float32
    sprite_idx : jnp.ndarray
        Current character sprite index, shape (), dtype int32
    animation_timer : jnp.ndarray
        Character animation timer, shape (), dtype float32
    shape_angle : jnp.ndarray
        Character shape rotation angle (degrees), shape (), dtype float32
    shape_type_idx : jnp.ndarray
        Selected shape type index, shape (), dtype int32
    shape_color_idx : jnp.ndarray
        Selected shape color index, shape (), dtype int32
    npc_x : jnp.ndarray
        NPC world x positions, shape (max_npcs,), dtype int32
    npc_y : jnp.ndarray
        NPC world y positions, shape (max_npcs,), dtype int32
    npc_types : jnp.ndarray
        NPC type indices, shape (max_npcs,), dtype int32
    npc_sprite_indices : jnp.ndarray
        NPC animation frame indices, shape (max_npcs,), dtype int32
    npc_animation_timers : jnp.ndarray
        NPC animation timers, shape (max_npcs,), dtype float32
    sticky_x_offsets : jnp.ndarray
        Sticky NPC camera-relative x offsets, shape (max_sticky,), dtype int32
    sticky_y : jnp.ndarray
        Sticky NPC screen y positions, shape (max_sticky,), dtype int32
    sticky_types : jnp.ndarray
        Sticky NPC type indices, shape (max_sticky,), dtype int32
    sticky_vy : jnp.ndarray
        Sticky NPC y velocities, shape (max_sticky,), dtype float32
    sticky_on_ground : jnp.ndarray
        Sticky NPC ground contact flags, shape (max_sticky,), dtype bool
    sticky_home_y : jnp.ndarray
        Sticky NPC home y positions, shape (max_sticky,), dtype int32
    sticky_sprite_indices : jnp.ndarray
        Sticky NPC sprite indices, shape (max_sticky,), dtype int32
    sticky_animation_timers : jnp.ndarray
        Sticky NPC animation timers, shape (max_sticky,), dtype float32
    actual_sticky_count : jnp.ndarray
        Actual number of sticky NPCs spawned, shape (), dtype int32
    dist_x : jnp.ndarray
        Distractor x positions, shape (max_dist,), dtype float32
    dist_y : jnp.ndarray
        Distractor y positions, shape (max_dist,), dtype float32
    dist_vx : jnp.ndarray
        Distractor x velocities, shape (max_dist,), dtype float32
    dist_vy : jnp.ndarray
        Distractor y velocities, shape (max_dist,), dtype float32
    dist_angles : jnp.ndarray
        Distractor rotation angles, shape (max_dist,), dtype float32
    dist_rot_speeds : jnp.ndarray
        Distractor rotation speeds, shape (max_dist,), dtype float32
    dist_shape_indices : jnp.ndarray
        Distractor shape type indices, shape (max_dist,), dtype int32
    dist_color_indices : jnp.ndarray
        Distractor color indices, shape (max_dist,), dtype int32
    dist_sizes : jnp.ndarray
        Distractor sizes, shape (max_dist,), dtype int32
    """
    
    # Agent state
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    grounded: jnp.ndarray
    coyote_timer: jnp.ndarray
    
    # Camera
    camera_x: jnp.ndarray
    
    # Time and progress
    t: jnp.ndarray
    x_max: jnp.ndarray
    episode_return: jnp.ndarray
    jump_count: jnp.ndarray
    initial_x: jnp.ndarray
    success_once: jnp.ndarray
    vx_mean: jnp.ndarray
    vy_mean: jnp.ndarray
    
    # Layout (immutable after reset)
    layout_solid_mask: jnp.ndarray
    layout_ground_top: jnp.ndarray
    layout_ground_bottom: jnp.ndarray
    layout_color: jnp.ndarray
    
    # Background and effects (immutable after reset)
    bg_image: Optional[jnp.ndarray]
    filter_key: jnp.ndarray
    selected_filter_idx: jnp.ndarray
    light_positions: jnp.ndarray
    light_radii: jnp.ndarray
    light_intensities: jnp.ndarray
    
    # Character animation
    sprite_idx: jnp.int32 # Current sprite index (0 to num_sprites-1)
    animation_timer: jnp.float32 # Timer for animation frame transition
    shape_angle: jnp.float32  # Rotation angle for shape mode
    shape_type_idx: jnp.int32  # Index of shape type (if using shape mode)
    shape_color_idx: jnp.int32  # Index of shape color (if using shape mode)
    char_skin_idx: jnp.int32 # Index of character skin (if using sprite_dirs)
    
    # NPCs (world-fixed)
    npc_x: jnp.ndarray
    npc_y: jnp.ndarray
    npc_types: jnp.ndarray
    npc_sprite_indices: jnp.ndarray
    npc_animation_timers: jnp.ndarray
    
    # Sticky NPCs (camera-relative)
    sticky_x_offsets: jnp.ndarray
    sticky_y: jnp.ndarray
    sticky_types: jnp.ndarray
    sticky_vy: jnp.ndarray
    sticky_on_ground: jnp.ndarray
    sticky_home_y: jnp.ndarray
    sticky_sprite_indices: jnp.ndarray
    sticky_animation_timers: jnp.ndarray
    actual_sticky_count: jnp.ndarray

    # Distractors
    dist_x: jnp.ndarray
    dist_y: jnp.ndarray
    dist_vx: jnp.ndarray
    dist_vy: jnp.ndarray
    dist_angles: jnp.ndarray
    dist_rot_speeds: jnp.ndarray
    dist_shape_indices: jnp.ndarray
    dist_color_indices: jnp.ndarray
    dist_sizes: jnp.ndarray

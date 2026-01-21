"""NPC (non-player character) system for JAX platformer.

NPCs are decorative characters that:
- Stand at fixed positions in the level (world-fixed NPCs)
- Or follow camera (sticky NPCs, always visible)
- Play walk animations in place
- Have no collision (player can walk through them)
- Are randomly selected from available sprite directories

Functions
---------
load_npc_sprite_sets
    Load sprite sets for all NPC types
generate_npc_positions
    Generate random world-fixed NPC positions
render_npc_by_type
    Render NPC with type selection
update_npc_animation
    Update NPC animation states
spawn_sticky_npcs
    Spawn camera-relative NPCs
update_sticky_npc_physics
    Update sticky NPC physics (gravity, jumps)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp

from .character import load_sprites_from_directory, render_sprite_from_list


@dataclass
class NPCConfig:
    """Configuration for NPC system."""

    # Regular NPC spawn settings (fixed world positions)
    enabled: bool = False

    # Sprite configuration (choose one of the following options)
    sprite_dir: Optional[str] = None  # Path to directory containing sprite subdirectories (auto-discovers all subdirs)
    sprite_paths: list[str] = field(default_factory=list)  # List of sprite directories (explicit list)
    sprite_path: Optional[str] = None  # Path to single sprite directory
    
    # Spawn distribution
    min_npc_count: int = 2
    max_npc_count: int = 5
    
    # Animation settings
    animation_fps: float = 8.0  # Animation speed for NPCs
    
    # Positioning
    spawn_y_offset: int = 0  # Offset from ground (0 = on ground)
    
    # Sticky NPCs (camera-relative, always visible)
    sticky_enabled: bool = False
    min_sticky_count: int = 1  # Minimum number of sticky NPCs
    max_sticky_count: int = 3  # Maximum number of sticky NPCs
    sticky_sprite_dir: Optional[str] = None  # Path to directory containing sticky sprite subdirectories (auto-discovers all subdirs)
    sticky_sprite_dirs: list[str] = field(default_factory=list)  # List of sticky sprite directories (explicit list)
    sticky_sprite_path: Optional[str] = None  # Path to single sticky sprite directory
    sticky_can_jump: bool = False  # Whether sticky NPCs can jump
    sticky_jump_probability: float = 0.01  # Per-step probability of jump
    sticky_x_offsets: list[int] = field(default_factory=lambda: [-40, 40])  # Camera-relative X positions
    sticky_y_randomize: bool = True  # Randomize Y positions
    sticky_y_min_offset: int = -30  # Min Y offset from ground (negative = higher)
    sticky_y_max_offset: int = 0  # Max Y offset from ground (0 = on ground)
    sticky_x_min: Optional[int] = None  # Min X offset from center
    sticky_x_max: Optional[int] = None  # Max X offset from center


def load_npc_sprite_sets(config: NPCConfig) -> list[list[jnp.ndarray]]:
    """
    Load sprite sets for world-fixed NPC types based on config.

    Args:
        config: NPCConfig with sprite configuration

    Returns:
        npc_sprite_sets: List of sprite lists, one per NPC type
    """
    # Determine which sprite paths to use
    sprite_dirs = []

    if config.sprite_dir is not None:
        # Auto-discover all subdirectories in sprite_dir
        from pathlib import Path
        base_path = Path(config.sprite_dir)
        if base_path.exists() and base_path.is_dir():
            sprite_dirs = [str(p) for p in base_path.iterdir() if p.is_dir()]
            sprite_dirs.sort()  # Ensure consistent ordering
        else:
            print(f"Warning: NPC sprite_dir not found: {config.sprite_dir}")
    elif config.sprite_paths:
        # Use explicitly provided paths
        sprite_dirs = config.sprite_paths
    elif config.sprite_path is not None:
        # Single sprite directory
        sprite_dirs = [config.sprite_path]

    if not sprite_dirs:
        return []

    sprite_sets = []
    for sprite_dir in sprite_dirs:
        try:
            sprites = load_sprites_from_directory(sprite_dir)
            sprite_sets.append(sprites)
        except Exception as e:
            print(f"Warning: Failed to load NPC sprites from {sprite_dir}: {e}")
            continue

    if not sprite_sets:
        print("Warning: No NPC sprite sets loaded")

    return sprite_sets


def load_npc_sprite_sets_sticky(config: NPCConfig) -> list[list[jnp.ndarray]]:
    """
    Load sprite sets for sticky NPC types based on config.

    Args:
        config: NPCConfig with sticky sprite configuration

    Returns:
        sticky_sprite_sets: List of sprite lists, one per sticky NPC type
    """
    # Determine which sprite paths to use for sticky NPCs
    sprite_dirs = []

    if config.sticky_sprite_dir is not None:
        # Auto-discover all subdirectories in sticky_sprite_dir
        from pathlib import Path
        base_path = Path(config.sticky_sprite_dir)
        if base_path.exists() and base_path.is_dir():
            sprite_dirs = [str(p) for p in base_path.iterdir() if p.is_dir()]
            sprite_dirs.sort()  # Ensure consistent ordering
        else:
            print(f"Warning: Sticky NPC sprite_dir not found: {config.sticky_sprite_dir}")
    elif config.sticky_sprite_dirs:
        # Use explicitly provided paths (backward compatibility)
        sprite_dirs = config.sticky_sprite_dirs
    elif config.sticky_sprite_path is not None:
        # Single sprite directory
        sprite_dirs = [config.sticky_sprite_path]

    if not sprite_dirs:
        return []

    sprite_sets = []
    for sprite_dir in sprite_dirs:
        try:
            sprites = load_sprites_from_directory(sprite_dir)
            sprite_sets.append(sprites)
        except Exception as e:
            print(f"Warning: Failed to load sticky NPC sprites from {sprite_dir}: {e}")
            continue

    if not sprite_sets:
        print("Warning: No sticky NPC sprite sets loaded")

    return sprite_sets


def generate_npc_positions(
    key: jax.Array,
    config: NPCConfig,
    world_width: int,
    ground_top: jnp.ndarray,
    char_half_h: int,
    num_sprite_sets: int = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate random NPC positions along the level.
    
    Args:
        key: JAX PRNG key
        config: NPCConfig
        world_width: Width of world in pixels
        ground_top: (world_width,) array of ground top positions
        char_half_h: Character half-height for positioning
    
    Returns:
        npc_x: (max_npcs,) x positions
        npc_y: (max_npcs,) y positions
        npc_type: (max_npcs,) type indices (which sprite set to use)
    """
    max_npcs = config.max_npc_count
    
    # Generate random NPC count
    key, subkey = jax.random.split(key)
    n_npcs = jax.random.randint(
        subkey,
        shape=(),
        minval=config.min_npc_count,
        maxval=config.max_npc_count + 1,
    )
    
    # Generate positions with minimum spacing
    key, subkey = jax.random.split(key)
    
    # Simple approach: divide world into segments and place one NPC per segment
    positions_x = jnp.zeros(max_npcs, dtype=jnp.int32)
    positions_y = jnp.zeros(max_npcs, dtype=jnp.int32)
    npc_types = jnp.zeros(max_npcs, dtype=jnp.int32)
    
    for i in range(max_npcs):
        key, subkey = jax.random.split(key)
        
        # Random x position (avoid first 200 and last 200 pixels)
        x_min = 200
        x_max = max(x_min + 100, world_width - 200)
        npc_x = jax.random.randint(subkey, shape=(), minval=x_min, maxval=x_max)
        
        # Clamp to valid range
        npc_x = jnp.clip(npc_x, 0, world_width - 1)
        
        # Y position: on ground at this x
        npc_y = ground_top[npc_x] - char_half_h - 2 - config.spawn_y_offset
        
        # Random NPC type
        key, subkey = jax.random.split(key)
        n_types = num_sprite_sets if num_sprite_sets is not None else (len(config.sprite_paths) if config.sprite_paths else 1)
        npc_type = jax.random.randint(subkey, shape=(), minval=0, maxval=max(1, n_types))
        
        # Only use if i < n_npcs
        positions_x = positions_x.at[i].set(jnp.where(i < n_npcs, npc_x, 0))
        positions_y = positions_y.at[i].set(jnp.where(i < n_npcs, npc_y, 0))
        npc_types = npc_types.at[i].set(jnp.where(i < n_npcs, npc_type, 0))
    
    return positions_x, positions_y, npc_types


def render_npc(
    img: jnp.ndarray,
    npc_sprite_sets: list[list[jnp.ndarray]],
    npc_x: int,
    npc_y: int,
    npc_type: int,
    sprite_idx: int,
    camera_x: jax.Array,
    char_width: int,
    char_height: int,
    is_active: bool,
) -> jnp.ndarray:
    """
    Render single NPC (JIT-compatible).
    
    Args:
        img: (H, W, 3) uint8 image
        npc_sprite_sets: List of sprite lists (one per NPC type)
        npc_x: NPC world x position
        npc_y: NPC world y position
        npc_type: NPC type index
        sprite_idx: Current animation frame index
        camera_x: Camera position
        char_width: Character width
        char_height: Character height
        is_active: Whether to render this NPC
    
    Returns:
        img: (H, W, 3) uint8 image with NPC rendered
    """
    if not is_active or not npc_sprite_sets:
        return img
    
    # Convert world position to screen position
    npc_x_screen = npc_x - jnp.round(camera_x).astype(jnp.int32)
    npc_y_screen = npc_y
    
    # Check if NPC is visible on screen
    H, W = img.shape[:2]
    is_visible = (npc_x_screen >= -char_width) & (npc_x_screen < W + char_width) & \
                 (npc_y_screen >= -char_height) & (npc_y_screen < H + char_height)
    
    # Select sprite set by NPC type
    # For JIT compatibility, we need to handle sprite set selection carefully
    # Clamp type index to valid range
    npc_type_clamped = jnp.clip(npc_type, 0, len(npc_sprite_sets) - 1)
    
    # Use the selected sprite set
    def render_with_type(type_idx):
        sprite_set = npc_sprite_sets[type_idx]
        return render_sprite_from_list(
            img,
            sprite_set,
            sprite_idx,
            npc_x_screen,
            npc_y_screen,
            char_width,
            char_height,
        )
    
    # Only render if visible and active
    img_with_npc = jax.lax.cond(
        is_visible & is_active,
        lambda: render_npc_by_type(img, npc_sprite_sets, npc_type_clamped, sprite_idx, npc_x_screen, npc_y_screen, char_width, char_height),
        lambda: img,
    )
    
    return img_with_npc


def render_npc_by_type(
    img: jnp.ndarray,
    npc_sprite_sets: list[list[jnp.ndarray]],
    npc_type: int,
    sprite_idx: int,
    npc_x_screen: int,
    npc_y_screen: int,
    char_width: int,
    char_height: int,
) -> jnp.ndarray:
    """Render NPC with type selection via jax.lax.switch."""
    if not npc_sprite_sets:
        return img
    
    def make_render_branch(sprite_set):
        return lambda: render_sprite_from_list(
            img, sprite_set, sprite_idx, npc_x_screen, npc_y_screen, char_width, char_height
        )
    
    branches = [make_render_branch(s) for s in npc_sprite_sets]
    
    # Clamp to valid range
    npc_type = jnp.clip(npc_type, 0, len(branches) - 1)
    
    return jax.lax.switch(npc_type, branches)


def update_npc_animation(
    sprite_indices: jnp.ndarray,
    animation_timers: jnp.ndarray,
    dt: float,
    fps: float,
    sprite_counts: list[int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Update NPC animation states.
    
    Args:
        sprite_indices: (max_npcs,) current sprite indices
        animation_timers: (max_npcs,) animation timers
        dt: Time delta
        fps: Animation FPS
        sprite_counts: Number of sprites per NPC type
    
    Returns:
        next_indices: (max_npcs,) updated sprite indices
        next_timers: (max_npcs,) updated timers
    """
    max_npcs = sprite_indices.shape[0]
    frame_duration = 1.0 / fps
    
    next_timers = animation_timers + dt
    should_advance = next_timers >= frame_duration
    
    # For now, use a fixed sprite count (take max or first)
    # In real impl, we'd need per-NPC sprite counts
    max_sprites = max(sprite_counts) if sprite_counts else 2
    
    next_indices = jnp.where(
        should_advance,
        (sprite_indices + 1) % max_sprites,
        sprite_indices
    )
    
    next_timers = jnp.where(
        should_advance,
        next_timers - frame_duration,
        next_timers
    )
    
    return next_indices, next_timers


# ============================================================================
# Sticky NPCs (camera-relative, always visible)
# ============================================================================

def spawn_sticky_npcs(
    key: jax.Array,
    npc_cfg: NPCConfig,
    num_npc_types: int,
    agent_y: float,
    ground_y: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Spawn sticky NPCs at camera-relative positions.

    Args:
        key: PRNG key
        npc_cfg: NPC configuration
        num_npc_types: Number of available sticky NPC sprite sets
        agent_y: Agent's Y position (for relative positioning)
        ground_y: Ground Y coordinate (scalar)

    Returns:
        sticky_x_offsets: Camera-relative X positions (N,)
        sticky_y: Screen Y positions (N,)
        sticky_types: NPC type indices (N,)
        sticky_vy: Vertical velocities (N,)
        sticky_on_ground: Ground contact flags (N,)
        sticky_home_y: Home Y positions for each NPC (where they rest) (N,)
        actual_count: Actual number of sticky NPCs spawned
    """
    # Use max count for array sizes, but randomly select actual count
    max_count = npc_cfg.max_sticky_count
    
    # Generate random actual count
    key, subkey = jax.random.split(key)
    actual_count = jax.random.randint(
        subkey,
        shape=(),
        minval=npc_cfg.min_sticky_count,
        maxval=npc_cfg.max_sticky_count + 1,
    )

    # Parse X offsets
    key, k_x = jax.random.split(key)
    if npc_cfg.sticky_x_offsets:
        # Use provided explicit offsets, cycle if needed
        offsets_list = npc_cfg.sticky_x_offsets
        x_offsets = jnp.array([offsets_list[i % len(offsets_list)] for i in range(max_count)], dtype=jnp.int32)
    elif npc_cfg.sticky_x_min is not None and npc_cfg.sticky_x_max is not None:
        # Generate random offsets in range
        x_offsets = jax.random.randint(
            k_x,
            (max_count,),
            npc_cfg.sticky_x_min,
            npc_cfg.sticky_x_max + 1,
            dtype=jnp.int32
        )
    else:
        # Default: spread around agent
        x_offsets = jnp.linspace(-60, 60, max_count, dtype=jnp.int32)
    
    # Y positions: randomize if enabled
    key, subkey = jax.random.split(key)
    if npc_cfg.sticky_y_randomize:
        # Random Y offset for each NPC
        y_offsets = jax.random.randint(
            subkey,
            (max_count,),
            npc_cfg.sticky_y_min_offset,
            npc_cfg.sticky_y_max_offset + 1,
            dtype=jnp.int32
        )
        y_positions = jnp.full(max_count, int(ground_y), dtype=jnp.int32) + y_offsets
    else:
        # All on same level (use y_min_offset as fixed offset)
        fixed_offset = npc_cfg.sticky_y_min_offset
        y_positions = jnp.full(max_count, int(ground_y) + fixed_offset, dtype=jnp.int32)
    
    # Save home Y positions (where each NPC rests when not jumping)
    home_y = y_positions.copy()
    
    # Randomly assign types
    key, subkey = jax.random.split(key)
    if num_npc_types > 0:
        types = jax.random.randint(subkey, (max_count,), 0, num_npc_types, dtype=jnp.int32)
    else:
        types = jnp.zeros(max_count, dtype=jnp.int32)

    # Initialize physics
    vy = jnp.zeros(max_count, dtype=jnp.float32)
    on_ground = jnp.ones(max_count, dtype=jnp.bool_)

    return x_offsets, y_positions, types, vy, on_ground, home_y, actual_count


def update_sticky_npc_physics(
    key: jax.Array,
    npc_cfg: NPCConfig,
    sticky_y: jnp.ndarray,
    sticky_vy: jnp.ndarray,
    sticky_on_ground: jnp.ndarray,
    sticky_home_y: jnp.ndarray,
    dt: float,
    gravity_accel: float = 0.5,
    jump_velocity: float = -6.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Update sticky NPC physics (gravity, jumps).
    
    Each NPC has its own "home" Y position where it rests when not jumping.
    Uses same physics units as PhysicsConfig (px/step).
    
    Args:
        key: PRNG key for jump decisions
        npc_cfg: NPC configuration
        sticky_y: Current Y positions (N,)
        sticky_vy: Current Y velocities (N,) in px/step
        sticky_on_ground: Ground contact flags (N,)
        sticky_home_y: Home Y positions for each NPC (N,)
        dt: Time step (unused, kept for API compatibility)
        gravity_accel: Gravity acceleration in px/step^2 (default: 0.5)
        jump_velocity: Jump velocity in px/step (negative = upward, default: -6.0)
    
    Returns:
        next_y: Updated Y positions
        next_vy: Updated Y velocities
        next_on_ground: Updated ground contact flags
    """
    count = sticky_y.shape[0]
    
    # Decide jumps (only if on ground and jump enabled)
    if npc_cfg.sticky_can_jump:
        key, subkey = jax.random.split(key)
        jump_rolls = jax.random.uniform(subkey, (count,))
        should_jump = (jump_rolls < npc_cfg.sticky_jump_probability) & sticky_on_ground
    else:
        should_jump = jnp.zeros(count, dtype=jnp.bool_)
    
    # Apply jump impulse
    next_vy = jnp.where(should_jump, jump_velocity, sticky_vy)
    
    # Apply gravity (px/step^2)
    next_vy = next_vy + gravity_accel
    
    # Update position (vy is px/step)
    next_y = sticky_y + next_vy
    
    # Collision with home Y position (each NPC has its own)
    home_collision = next_y >= sticky_home_y
    next_y = jnp.where(home_collision, sticky_home_y, next_y)
    next_vy = jnp.where(home_collision, 0.0, next_vy)
    next_on_ground = home_collision
    
    return next_y.astype(jnp.int32), next_vy, next_on_ground


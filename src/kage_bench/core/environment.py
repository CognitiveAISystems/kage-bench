"""KAGE-Bench: Known-Axis Generalization Evaluation Benchmark.

Core environment implementation with:
- Immutable state (Flax struct dataclasses)
- Pure functions: reset(), step(), render()
- JIT-friendly (no Python control flow)
- Static shapes (no dynamic array resizing)
- Deterministic rendering

Usage
-----
Basic usage:

>>> import jax
>>> from kage_bench.core import KAGE_Env, EnvConfig
>>> 
>>> config = EnvConfig()
>>> env = KAGE_Env(config)
>>> 
>>> key = jax.random.PRNGKey(0)
>>> state = env.reset(key)
>>> 
>>> action = jnp.int32(0)  # NOOP
>>> step_output = env.step(state, action)
>>> next_state, obs, reward, done, info = step_output.state, step_output.obs, step_output.reward, step_output.done, step_output.info

JIT compilation:

>>> reset_jit = jax.jit(env.reset)
>>> step_jit = jax.jit(env.step)
>>> render_jit = jax.jit(env.render)

Vectorization (vmap):

>>> reset_vec = jax.vmap(env.reset)
>>> step_vec = jax.vmap(env.step)
>>> 
>>> keys = jax.random.split(jax.random.PRNGKey(0), 100)
>>> states = reset_vec(keys)  # 100 parallel envs
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import EnvConfig
from .state import EnvState
from ..systems.camera import update_camera
from ..systems.layout import Layout, generate_layout
from ..systems.physics import PhysicsState, apply_physics, resolve_collision_with_physics, is_grounded
from ..systems.renderer import render
from ..systems.generation.background import load_background_image, generate_noise_background, select_color_by_index
from ..systems.generation.effects import generate_light_properties
from ..entities.character import (
    load_sprites_from_directory,
    load_character_sprite_sets,
    render_character_rect,
    update_animation_state,
)
from ..entities.npc import load_npc_sprite_sets, load_npc_sprite_sets_sticky, generate_npc_positions, update_npc_animation, spawn_sticky_npcs, update_sticky_npc_physics
from ..entities.distractors import spawn_distractors, update_distractors
from ..utils.shapes import SHAPE_COLORS


class KAGE_Env:
    """Pure functional JAX platformer environment.
    
    All methods are pure functions operating on immutable EnvState.
    No mutable state in class; all state passed explicitly.
    
    Attributes
    ----------
    config : EnvConfig
        Environment configuration (immutable)
    character_sprites : List[jnp.ndarray]
        Preloaded character sprites (empty if not using sprites)
    npc_sprite_sets : List[List[jnp.ndarray]]
        Preloaded world-fixed NPC sprite sets
    sticky_npc_sprite_sets : List[List[jnp.ndarray]]
        Preloaded sticky NPC sprite sets
    background_image : Optional[jnp.ndarray]
        Preloaded background image (None if not using image mode)
    
    Notes
    -----
    Design principles:
    - Pure functional: reset() and step() have no side effects
    - Immutable state: All state in EnvState, updated functionally
    - JIT-friendly: All per-step code uses jax.lax control flow
    - Static shapes: No dynamic array resizing
    - Deterministic: Same seed -> same trajectory
    
    Action Space
    ------------
    Discrete bitmask {0..7}:
    - LEFT  = 1 (bit 0)
    - RIGHT = 2 (bit 1)
    - JUMP  = 4 (bit 2)
    - Can combine: e.g., action=3 = LEFT + RIGHT (cancels), action=5 = LEFT + JUMP
        0 = NOOP
        1 = LEFT
        2 = RIGHT
        3 = LEFT + RIGHT (cancels out)
        4 = JUMP
        5 = LEFT + JUMP
        6 = RIGHT + JUMP
        7 = LEFT + RIGHT + JUMP (horizontal cancels, jump only)
    
    Observation Space
    -----------------
    RGB image: (H, W, 3) uint8, where H and W are from config
    
    Reward
    ------
    forward_progress - jump_penalty:
    - Forward progress: config.forward_reward_scale * max(0, x_new - x_max_prev)
    - Jump penalty: config.jump_penalty * 1[jump pressed]
    
    Episode Termination
    -------------------
    Truncation at t >= config.episode_length (no task-based termination)
    
    Examples
    --------
    Create environment with custom config:
    
    >>> from kage_bench.core import KAGE_Env, EnvConfig
    >>> from kage_bench.systems import LayoutConfig
    >>> 
    >>> layout_cfg = LayoutConfig(length=512, height_px=128)
    >>> env_cfg = EnvConfig(H=128, W=128, layout=layout_cfg)
    >>> env = KAGE_Env(env_cfg)
    
    Reset and step:
    
    >>> import jax
    >>> key = jax.random.PRNGKey(42)
    >>> state = env.reset(key)
    >>> print(state.x, state.y, state.t)
    >>> 
    >>> action = jnp.int32(2)  # RIGHT
    >>> output = env.step(state, action)
    >>> print(output.reward, output.done)
    
    Render:
    
    >>> img = env.render(state)
    >>> assert img.shape == (env.config.H, env.config.W, 3)
    >>> assert img.dtype == jnp.uint8
    """
    
    # Action constants (bitmask)
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 4
    
    def __init__(self, config: EnvConfig):
        """Initialize environment with configuration.
        
        Parameters
        ----------
        config : EnvConfig
            Complete environment configuration
        
        Notes
        -----
        This loads sprites and background images from disk (host-side, not JIT).
        The loaded data is then embedded in JIT-compiled functions.
        """
        self.config = config
        
        # Load background image(s) if needed
        self.background_image = None
        self.background_images = None
        
        if config.background.mode == "image":
            # Determine which paths to use
            paths = config.background.image_paths
            
            # If image_dir is set, find all images in it
            if not paths and config.background.image_dir:
                from pathlib import Path
                image_dir = Path(config.background.image_dir)
                if image_dir.exists():
                     # Support common image extensions
                     extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
                     found_files = []
                     for ext in extensions:
                         found_files.extend(image_dir.glob(ext))
                     
                     if found_files:
                         # Sort for deterministic order
                         paths = [str(p) for p in sorted(found_files)]
            
            # Fallback to single image path
            if not paths and config.background.image_path:
                paths = [config.background.image_path]
            
            if not paths:
                # If mode is image but no images found, warn and fallback to black or handle gracefully
                # For now raise error as before, but with clearer message
                raise ValueError(
                    "background.mode='image' but no images found. "
                    "Set 'image_paths', 'image_dir', or 'image_path'."
                )
            
            # Load first image to establish dimensions
            first_img = load_background_image(
                paths[0],
                target_height=config.H,
            )
            H, W = first_img.shape[:2]
            
            # Load rest of images, forcing same dimensions
            loaded_images = [first_img]
            for path in paths[1:]:
                img = load_background_image(
                    path,
                    target_height=config.H,
                    target_width=W,
                )
                loaded_images.append(img)
            
            # Stack for JAX indexing
            if len(loaded_images) > 1:
                self.background_images = jnp.stack(loaded_images)
            else:
                self.background_images = jnp.expand_dims(loaded_images[0], axis=0)
            
            # Keep single image reference for non-switching compat if needed (index 0)
            self.background_image = loaded_images[0]
        
        # Load character sprites if needed
        self.character_sprites = [] # List of single sprites (H, W, 4) for single skin mode
        self.character_skin_sets = [] # List of list of sprites [ [s1, s2], [s1, s2] ] for multi-skin
        
        # Handle sprite configuration
        if config.character.use_sprites:
            sprite_paths = []

            # Determine which sprite paths to use
            if config.character.sprite_dir is not None:
                # Auto-discover all subdirectories in sprite_dir
                from pathlib import Path
                base_path = Path(config.character.sprite_dir)
                if base_path.exists():
                    sprite_paths = [str(p) for p in base_path.iterdir() if p.is_dir()]
                    sprite_paths.sort()  # Ensure consistent ordering
                else:
                    print(f"Warning: Character sprite_dir not found: {config.character.sprite_dir}")
            elif config.character.sprite_paths:
                # Use explicitly provided paths
                sprite_paths = config.character.sprite_paths
            elif config.character.sprite_path is not None:
                # Single sprite directory
                sprite_paths = [config.character.sprite_path]

            # Load the sprite sets
            if sprite_paths:
                self.character_skin_sets = load_character_sprite_sets(sprite_paths)
                # Use first skin as default for single-skin logic fallback
                if self.character_skin_sets:
                    self.character_sprites = self.character_skin_sets[0]
        
        # Load NPC sprites (world-fixed)
        self.npc_sprite_sets = []
        self.npc_sprite_counts = []
        if config.npc.enabled and (config.npc.sprite_dir or config.npc.sprite_paths or config.npc.sprite_path):
            self.npc_sprite_sets = load_npc_sprite_sets(config.npc)
            self.npc_sprite_counts = [len(s) for s in self.npc_sprite_sets]
        
        # Load sticky NPC sprites (camera-relative)
        self.sticky_npc_sprite_sets = []
        self.sticky_npc_sprite_counts = []
        if config.npc.sticky_enabled and (config.npc.sticky_sprite_dir or config.npc.sticky_sprite_dirs or config.npc.sticky_sprite_path):
            self.sticky_npc_sprite_sets = load_npc_sprite_sets_sticky(config.npc)
            self.sticky_npc_sprite_counts = [len(s) for s in self.sticky_npc_sprite_sets]
    
    def reset(self, key: jax.Array) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.
        
        Pure function: no side effects, deterministic given key.
        
        Parameters
        ----------
        key : jax.Array
            PRNG key, shape (2,), dtype uint32
        
        Returns
        -------
        obs : jnp.ndarray
            Initial RGB observation, shape (H, W, 3), dtype uint8
        info : Dict[str, Any]
            Auxiliary information dictionary containing 'state'
        
        Notes
        -----
        Reset workflow:
        1. Generate layout (platforms, solid mask)
        2. Spawn agent at safe position
        3. Initialize physics state
        4. Generate visual elements (background, lights, NPCs, distractors)
        5. Initialize animation states
        6. Render initial observation
        
        Examples
        --------
        >>> import jax
        >>> env = KAGE_Env(EnvConfig())
        >>> key = jax.random.PRNGKey(0)
        >>> obs, info = env.reset(key)
        >>> state = info["state"]
        >>> print(obs.shape, state.t)
        (128, 128, 3) 0
        """
        # Split key for subsystems
        key, k_layout, k_bg, k_filter, k_light, k_npc, k_sticky, k_distractor = jax.random.split(key, 8)
        
        # Generate layout
        layout: Layout = generate_layout(k_layout, self.config.layout)
        
        # Spawn agent at safe position
        char_half_w = self.config.character.width // 2
        char_half_h = self.config.character.height // 2
        cx, cy = self._find_safe_spawn(layout, char_half_w, char_half_h)
        x = cx.astype(jnp.float32)
        y = cy.astype(jnp.float32)
        
        # Initialize camera
        world_width = layout.solid_mask.shape[1]
        camera_x = jnp.clip(
            x - jnp.float32(self.config.W // 2),
            0.0,
            jnp.float32(world_width - self.config.W)
        )
        
        # Initialize physics state
        phys_state = PhysicsState(
            vx=jnp.array(0.0, dtype=jnp.float32),
            vy=jnp.array(0.0, dtype=jnp.float32),
            grounded=jnp.array(False),
            coyote_timer=jnp.int32(0),
        )
        
        # Generate background
        bg_image = self._generate_background(k_bg)
        
        # Select filter preset if multiple options provided
        selected_filter_idx = jnp.int32(0)
        if len(self.config.filters.pop_filter_list) > 0:
            n_filters = len(self.config.filters.pop_filter_list)
            selected_filter_idx = jax.random.randint(k_filter, shape=(), minval=0, maxval=n_filters)
        
        # Generate light properties once per episode
        light_positions, light_radii, light_intensities = generate_light_properties(
            k_light, self.config.effects.point_light_count, self.config.effects
        )
        
        # Initialize character animation
        sprite_idx = jnp.int32(self.config.character.idle_sprite_idx)
        animation_timer = jnp.float32(0.0)
        shape_angle = jnp.float32(0.0)
        
        # Select character skin (if multi-skin enabled)
        char_skin_idx = jnp.int32(0)
        if self.character_skin_sets and len(self.character_skin_sets) > 1:
            key, k_skin = jax.random.split(key)
            n_skins = len(self.character_skin_sets)
            char_skin_idx = jax.random.randint(k_skin, shape=(), minval=0, maxval=n_skins).astype(jnp.int32)
        
        # Select shape type and color (if using shapes)
        shape_type_idx, shape_color_idx = self._select_shape_properties(key)
        
        # Generate NPCs (world-fixed)
        npc_x, npc_y, npc_types, npc_sprite_indices, npc_animation_timers = self._spawn_world_npcs(
            k_npc, layout, char_half_h
        )
        
        # Spawn sticky NPCs (camera-relative)
        (sticky_x_offsets, sticky_y, sticky_types, sticky_vy, sticky_on_ground,
         sticky_home_y, sticky_sprite_indices, sticky_animation_timers, actual_sticky_count) = self._spawn_sticky_npcs(
            k_sticky, y, self.config.layout.base_ground_y
        )
        
        # Spawn distractors
        (dist_x, dist_y, dist_vx, dist_vy, dist_angles, dist_rot_speeds,
         dist_shape_indices, dist_color_indices, dist_sizes) = self._spawn_distractors(k_distractor)
        
        # Select layout color
        layout_color = self._select_layout_color(key)
        
        # Construct initial state
        t = jnp.array(0, dtype=jnp.int32)
        x_max = x
        episode_return = jnp.array(0.0, dtype=jnp.float32)
        jump_count = jnp.array(0, dtype=jnp.int32)
        initial_x = x
        success_once = jnp.array(False, dtype=bool)
        vx_mean = jnp.array(0.0, dtype=jnp.float32)
        vy_mean = jnp.array(0.0, dtype=jnp.float32)
        
        state = EnvState(
            # Agent
            x=x,
            y=y,
            vx=phys_state.vx,
            vy=phys_state.vy,
            grounded=phys_state.grounded,
            coyote_timer=phys_state.coyote_timer,
            # Camera
            camera_x=camera_x,
            # Time and progress
            t=t,
            x_max=x_max,
            episode_return=episode_return,
            jump_count=jump_count,
            initial_x=initial_x,
            success_once=success_once,
            vx_mean=vx_mean,
            vy_mean=vy_mean,
            # Layout
            layout_solid_mask=layout.solid_mask,
            layout_ground_top=layout.ground_top,
            layout_ground_bottom=layout.ground_bottom,
            layout_color=layout_color,
            # Background and effects
            bg_image=bg_image,
            filter_key=k_filter,
            selected_filter_idx=selected_filter_idx,
            light_positions=light_positions,
            light_radii=light_radii,
            light_intensities=light_intensities,
            # Character animation
            sprite_idx=sprite_idx,
            animation_timer=animation_timer,
            shape_angle=shape_angle,
            shape_type_idx=shape_type_idx,
            shape_color_idx=shape_color_idx,
            char_skin_idx=char_skin_idx,
            # NPCs (world-fixed)
            npc_x=npc_x,
            npc_y=npc_y,
            npc_types=npc_types,
            npc_sprite_indices=npc_sprite_indices,
            npc_animation_timers=npc_animation_timers,
            # Sticky NPCs (camera-relative)
            sticky_x_offsets=sticky_x_offsets,
            sticky_y=sticky_y,
            sticky_types=sticky_types,
            sticky_vy=sticky_vy,
            sticky_on_ground=sticky_on_ground,
            sticky_home_y=sticky_home_y,
            sticky_sprite_indices=sticky_sprite_indices,
            sticky_animation_timers=sticky_animation_timers,
            actual_sticky_count=actual_sticky_count,
            # Distractors
            dist_x=dist_x,
            dist_y=dist_y,
            dist_vx=dist_vx,
            dist_vy=dist_vy,
            dist_angles=dist_angles,
            dist_rot_speeds=dist_rot_speeds,
            dist_shape_indices=dist_shape_indices,
            dist_color_indices=dist_color_indices,
            dist_sizes=dist_sizes,
        )
        
        # Render initial frame
        obs = self.render(state)
        
        # Return standard Gymnasium reset tuple (obs, info)
        info = {"state": state}
        return obs, info
    
    def step(
        self, state: EnvState, action: jax.Array, render: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Execute one environment step.

        Pure function: step(state, action, render=False) -> (obs, reward, terminated, truncated, info)

        Parameters
        ----------
        state : EnvState
            Current environment state
        action : jax.Array
            Action bitmask, shape (), dtype int32, value in {0..7}
        render : bool
            Whether to render observation (default: False for performance)

        Returns
        -------
        obs : jnp.ndarray
            RGB observation, shape (H, W, 3), dtype uint8 (or 0s if render=False)
        reward : jnp.ndarray
            Scalar reward, shape (), dtype float32
        terminated : jnp.ndarray
            Episode termination flag (logical), shape (), dtype bool
        truncated : jnp.ndarray
            Episode truncation flag (time limit), shape (), dtype bool
        info : Dict[str, Any]
            Auxiliary info containing 'state', 'return', 'timestep', 'x', 'y', 'vx', 'vy',
            'jump_count', 'passed_distance', 'progress', 'success', 'success_once',
            'vx_mean', 'vy_mean'
        
        Notes
        -----
        Step workflow:
        1. Decode action (bitmask -> movement + jump)
        2. Apply physics
        3. Resolve collisions
        4. Update camera
        5. Update animations
        6. Compute reward
        7. Check termination/truncation
        8. Render observation (optional)
        
        Examples
        --------
        >>> obs, info = env.reset(jax.random.PRNGKey(0))
        >>> state = info["state"]
        >>> action = jnp.int32(2 | 4)  # RIGHT + JUMP
        >>> obs, reward, term, trunc, info = env.step(state, action)
        >>> print(reward, term, trunc)
        """
        if isinstance(state, dict):
            if "state" in state:
                raise TypeError(
                    "Received a dict instead of EnvState. "
                    "Did you pass the 'info' dict? Use 'info[\"state\"]' instead."
                )
            raise TypeError("Expected EnvState object, received dict. "
                            "Ensure you are passing the state object, not a dictionary.")
        
        a = jnp.array(action).astype(jnp.int32)
        
        # Decode bitmask action
        left = (a & jnp.int32(self.LEFT)) != 0
        right = (a & jnp.int32(self.RIGHT)) != 0
        jump_pressed = (a & jnp.int32(self.JUMP)) != 0
        action_x = right.astype(jnp.int32) - left.astype(jnp.int32)  # {-1, 0, 1}
        
        # Character dimensions
        char_half_w = self.config.character.width // 2
        char_half_h = self.config.character.height // 2
        
        # Apply physics
        solid = state.layout_solid_mask
        phys_state = PhysicsState(
            vx=state.vx,
            vy=state.vy,
            grounded=state.grounded,
            coyote_timer=state.coyote_timer,
        )
        
        x_new, y_new, phys_state_new = apply_physics(
            state.x,
            state.y,
            phys_state,
            solid,
            char_half_w,
            char_half_h,
            action_x,
            jump_pressed,
            self.config.physics,
        )
        
        # Resolve collisions
        x_new, y_new, phys_state_new = resolve_collision_with_physics(
            state.x,
            state.y,
            x_new,
            y_new,
            phys_state_new,
            solid,
            char_half_w,
            char_half_h,
        )
        
        # Update camera
        world_width = solid.shape[1]
        camera_x = update_camera(
            state.camera_x,
            x_new,
            self.config.W,
            self.config.camera,
        )
        camera_x = jnp.clip(camera_x, 0.0, jnp.float32(world_width - self.config.W))
        
        # Update time
        t = state.t + jnp.array(1, dtype=jnp.int32)
        
        # Update keys (split for various stochastic effects)
        filter_key, k_sticky_update, k_bg_switch, k_bg_new = jax.random.split(state.filter_key, 4)
        
        # Dynamic background switching
        should_switch = jax.random.uniform(k_bg_switch) < self.config.background.switch_frequency
        
        def switch_bg():
            return self._generate_background(k_bg_new)
        
        def keep_bg():
            return state.bg_image
            
        bg_image_new = jax.lax.cond(should_switch, switch_bg, keep_bg)
        
        # Update animations
        (sprite_idx_new, animation_timer_new, shape_angle_new,
         npc_sprite_indices_new, npc_animation_timers_new,
         sticky_y_new, sticky_vy_new, sticky_on_ground_new,
         sticky_sprite_indices_new, sticky_animation_timers_new,
         dist_x_new, dist_y_new, dist_vx_new, dist_vy_new, dist_angles_new) = self._update_animations(
            state, phys_state_new, k_sticky_update
        )
        
        # Compute reward
        dx_newmax = x_new - state.x_max
        forward_rew = jnp.maximum(jnp.array(0.0, dtype=jnp.float32), dx_newmax) * jnp.float32(
            self.config.forward_reward_scale
        )
        jump_cost = jump_pressed.astype(jnp.float32) * jnp.float32(self.config.jump_penalty)
        step_cost = jnp.float32(self.config.timestep_penalty)
        idle = (x_new == state.x).astype(jnp.float32)
        idle_cost = idle * jnp.float32(self.config.idle_penalty)
        reward = forward_rew - jump_cost - step_cost - idle_cost
        episode_return = state.episode_return + reward
        jump_count = state.jump_count + jump_pressed.astype(jnp.int32)
        passed_distance = x_new - state.initial_x
        success = passed_distance >= jnp.float32(self.config.dist_to_success)
        dist_to_success = jnp.float32(self.config.dist_to_success)
        progress = jnp.where(dist_to_success > 0, passed_distance / dist_to_success, 0.0)
        success_once = state.success_once | success
        t_float = t.astype(jnp.float32)
        vx_mean = (state.vx_mean * (t_float - 1.0) + phys_state_new.vx) / t_float
        vy_mean = (state.vy_mean * (t_float - 1.0) + phys_state_new.vy) / t_float
        
        # Check termination (for now just truncation, but interface supports both)
        terminated = jnp.array(False, dtype=bool)
        truncated = t >= jnp.int32(self.config.episode_length)
        
        # Construct next state
        next_state = EnvState(
            # Agent
            x=x_new,
            y=y_new,
            vx=phys_state_new.vx,
            vy=phys_state_new.vy,
            grounded=phys_state_new.grounded,
            coyote_timer=phys_state_new.coyote_timer,
            # Camera
            camera_x=camera_x,
            # Time and progress
            t=t,
            x_max=jnp.maximum(state.x_max, x_new),
            episode_return=episode_return,
            jump_count=jump_count,
            initial_x=state.initial_x,
            success_once=success_once,
            vx_mean=vx_mean,
            vy_mean=vy_mean,
            # Layout (immutable)
            layout_solid_mask=state.layout_solid_mask,
            layout_ground_top=state.layout_ground_top,
            layout_ground_bottom=state.layout_ground_bottom,
            layout_color=state.layout_color,
            # Background and effects (mostly immutable)
            bg_image=bg_image_new,
            filter_key=filter_key,
            selected_filter_idx=state.selected_filter_idx,
            light_positions=state.light_positions,
            light_radii=state.light_radii,
            light_intensities=state.light_intensities,
            # Character animation
            sprite_idx=sprite_idx_new,
            animation_timer=animation_timer_new,
            shape_angle=shape_angle_new,
            shape_type_idx=state.shape_type_idx,
            shape_color_idx=state.shape_color_idx,
            char_skin_idx=state.char_skin_idx,
            # NPCs (world-fixed)
            npc_x=state.npc_x, 
            npc_y=state.npc_y,
            npc_types=state.npc_types,
            npc_sprite_indices=npc_sprite_indices_new,
            npc_animation_timers=npc_animation_timers_new,
            # Sticky NPCs (camera-relative)
            sticky_x_offsets=state.sticky_x_offsets,
            sticky_y=sticky_y_new,
            sticky_types=state.sticky_types,
            sticky_vy=sticky_vy_new,
            sticky_on_ground=sticky_on_ground_new,
            sticky_home_y=state.sticky_home_y,
            sticky_sprite_indices=sticky_sprite_indices_new,
            sticky_animation_timers=sticky_animation_timers_new,
            actual_sticky_count=state.actual_sticky_count,
            # Distractors
            dist_x=dist_x_new,
            dist_y=dist_y_new,
            dist_vx=dist_vx_new,
            dist_vy=dist_vy_new,
            dist_angles=dist_angles_new,
            dist_rot_speeds=state.dist_rot_speeds,
            dist_shape_indices=state.dist_shape_indices,
            dist_color_indices=state.dist_color_indices,
            dist_sizes=state.dist_sizes,
        )
        
        # Render observation
        # Note: if render=False we return zero-array to maintain shape consistency for vmap
        # (Using lax.cond to avoid computation cost if possible, though strict shape requirements exist)
        
        # Render observation (conditionally for benchmarking, but usually True for RL)
        obs = jax.lax.cond(
            render,
            lambda s: self.render(s),
            lambda s: jnp.zeros((self.config.H, self.config.W, 3), dtype=jnp.uint8),
            next_state
        )
        
        info = {
            "state": next_state,
            "return": next_state.episode_return,
            "timestep": t,
            "x": x_new,
            "y": y_new,
            "vx": phys_state_new.vx,
            "vy": phys_state_new.vy,
            "jump_count": next_state.jump_count,
            "passed_distance": passed_distance,
            "progress": progress,
            "success": success,
            "success_once": next_state.success_once,
            "vx_mean": vx_mean,
            "vy_mean": vy_mean,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, state: EnvState) -> jnp.ndarray:
        """Render current state to RGB image.
        
        Pure function: render(state) -> image
        
        Parameters
        ----------
        state : EnvState
            Current environment state
        
        Returns
        -------
        img : jnp.ndarray
            RGB image, shape (H, W, 3), dtype uint8
        
        Notes
        -----
        Rendering is deterministic and pure.
        Same state always produces same image.
        
        Examples
        --------
        >>> state = env.reset(jax.random.PRNGKey(0))
        >>> img = env.render(state)
        >>> assert img.shape == (env.config.H, env.config.W, 3)
        """
        return render(
            state,
            self.config,
            self.character_sprites,
            self.npc_sprite_sets,
            self.sticky_npc_sprite_sets,
            self.character_skin_sets,
        )
    
    # Helper methods (not meant to be called directly, but still pure)
    
    @property
    def action_space(self):
        """Return Gymnasium action space."""
        try:
            from gymnasium import spaces
            return spaces.Discrete(8)
        except ImportError:
            return None

    @property
    def observation_space(self):
        """Return Gymnasium observation space."""
        try:
            from gymnasium import spaces
            import numpy as np
            return spaces.Box(
                low=0,
                high=255,
                shape=(self.config.H, self.config.W, 3),
                dtype=np.uint8,
            )
        except ImportError:
            return None

    def _find_safe_spawn(
        self,
        layout: Layout,
        char_half_w: int,
        char_half_h: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Find safe spawn position without collision."""
        def check_spawn_collision(cx_check, cy_check, solid):
            xs_off = jnp.arange(-char_half_w, char_half_w + 1, dtype=jnp.int32)
            ys_off = jnp.arange(-char_half_h, char_half_h + 1, dtype=jnp.int32)
            xs = jnp.clip(cx_check + xs_off, 0, solid.shape[1] - 1)
            ys = jnp.clip(cy_check + ys_off, 0, solid.shape[0] - 1)
            return jnp.any(solid[ys[:, None], xs[None, :]])
        
        world_width = layout.solid_mask.shape[1]
        max_search_positions = 20
        
        def try_position(i, state_tuple):
            cx_curr, cy_curr, found = state_tuple
            
            test_cx = 64 + i * (self.config.character.width + 4)
            test_cx = jnp.clip(test_cx, char_half_w, world_width - char_half_w - 1)
            
            test_cy_ground_top = layout.ground_top[test_cx]
            test_cy = test_cy_ground_top - char_half_h - 2
            
            has_collision = check_spawn_collision(test_cx, test_cy, layout.solid_mask)
            
            cx_next = jnp.where(found | (~has_collision), jnp.where(found, cx_curr, test_cx), cx_curr)
            cy_next = jnp.where(found | (~has_collision), jnp.where(found, cy_curr, test_cy), cy_curr)
            found_next = found | (~has_collision)
            
            return (cx_next, cy_next, found_next)
        
        initial_state = (jnp.int32(64), jnp.int32(64), jnp.array(False))
        cx_final, cy_final, found = jax.lax.fori_loop(0, max_search_positions, try_position, initial_state)
        
        return cx_final, cy_final
    
    def _generate_background(self, key: jax.Array) -> Optional[jnp.ndarray]:
        """Generate background based on config mode."""
        if self.config.background.mode == "noise":
            return generate_noise_background(key, self.config.H, self.config.W)
        elif self.config.background.mode == "color":
            n_colors = len(self.config.background.color_names)
            color_idx = jax.random.randint(key, shape=(), minval=0, maxval=n_colors)
            return select_color_by_index(color_idx, self.config.background.color_names, self.config.H, self.config.W)
        elif self.config.background.mode == "image":
            if self.background_images is None:
                return None
            
            n_images = self.background_images.shape[0]
            if n_images == 1:
                return self.background_images[0]
            
            idx = jax.random.randint(key, shape=(), minval=0, maxval=n_images)
            return self.background_images[idx]
        else:
            return None
    
    def _select_shape_properties(self, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Select shape type and color indices."""
        if not self.config.character.use_shape:
            return jnp.int32(0), jnp.int32(0)
        
        # Parse shape types
        if isinstance(self.config.character.shape_types, str):
            shape_types_list = [self.config.character.shape_types]
        else:
            shape_types_list = self.config.character.shape_types
        
        # Parse shape colors
        if isinstance(self.config.character.shape_colors, str):
            shape_colors_list = [self.config.character.shape_colors]
        else:
            shape_colors_list = self.config.character.shape_colors
        
        key, k_shape = jax.random.split(key)
        shape_type_idx = jax.random.randint(k_shape, (), 0, len(shape_types_list))
        
        key, k_color = jax.random.split(key)
        shape_color_idx = jax.random.randint(k_color, (), 0, len(shape_colors_list))
        
        return shape_type_idx, shape_color_idx
    
    def _spawn_world_npcs(
        self,
        key: jax.Array,
        layout: Layout,
        char_half_h: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Spawn world-fixed NPCs."""
        if self.config.npc.enabled and self.npc_sprite_sets:
            world_width = layout.solid_mask.shape[1]
            npc_x, npc_y, npc_types = generate_npc_positions(
                key,
                self.config.npc,
                world_width,
                layout.ground_top,
                char_half_h,
                num_sprite_sets=len(self.npc_sprite_sets),
            )
            max_npcs = npc_x.shape[0]
            npc_sprite_indices = jnp.zeros(max_npcs, dtype=jnp.int32)
            npc_animation_timers = jnp.zeros(max_npcs, dtype=jnp.float32)
            return npc_x, npc_y, npc_types, npc_sprite_indices, npc_animation_timers
        else:
            max_npcs = 5
            return (
                jnp.zeros(max_npcs, dtype=jnp.int32),
                jnp.zeros(max_npcs, dtype=jnp.int32),
                jnp.zeros(max_npcs, dtype=jnp.int32),
                jnp.zeros(max_npcs, dtype=jnp.int32),
                jnp.zeros(max_npcs, dtype=jnp.float32),
            )
    
    def _spawn_sticky_npcs(
        self,
        key: jax.Array,
        agent_y: float,
        ground_y: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Spawn camera-relative sticky NPCs."""
        if self.config.npc.sticky_enabled and self.sticky_npc_sprite_sets:
            sticky_x_offsets, sticky_y, sticky_types, sticky_vy, sticky_on_ground, sticky_home_y, actual_sticky_count = spawn_sticky_npcs(
                key,
                self.config.npc,
                len(self.sticky_npc_sprite_sets),
                agent_y,
                ground_y - self.config.npc.spawn_y_offset,
            )
            max_sticky = sticky_x_offsets.shape[0]
            sticky_sprite_indices = jnp.zeros(max_sticky, dtype=jnp.int32)
            sticky_animation_timers = jnp.zeros(max_sticky, dtype=jnp.float32)
            return (sticky_x_offsets, sticky_y, sticky_types, sticky_vy, sticky_on_ground,
                    sticky_home_y, sticky_sprite_indices, sticky_animation_timers, actual_sticky_count)
        else:
            max_sticky = max(self.config.npc.max_sticky_count, 1)
            return (
                jnp.zeros(max_sticky, dtype=jnp.int32),
                jnp.zeros(max_sticky, dtype=jnp.int32),
                jnp.zeros(max_sticky, dtype=jnp.int32),
                jnp.zeros(max_sticky, dtype=jnp.float32),
                jnp.ones(max_sticky, dtype=jnp.bool_),
                jnp.zeros(max_sticky, dtype=jnp.int32),
                jnp.zeros(max_sticky, dtype=jnp.int32),
                jnp.zeros(max_sticky, dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32),  # actual_sticky_count
            )
    
    def _spawn_distractors(
        self,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, ...]:
        """Spawn visual distractors."""
        if self.config.distractors.enabled:
            if isinstance(self.config.distractors.shape_types, str):
                distractor_shape_types = [self.config.distractors.shape_types]
            else:
                distractor_shape_types = self.config.distractors.shape_types
            
            if isinstance(self.config.distractors.shape_colors, str):
                distractor_colors = [self.config.distractors.shape_colors]
            else:
                distractor_colors = self.config.distractors.shape_colors
            
            return spawn_distractors(
                key,
                self.config.distractors,
                self.config.W,
                self.config.H,
                len(distractor_shape_types),
                len(distractor_colors),
            )
        else:
            max_distractors = max(self.config.distractors.count, 1)
            return (
                jnp.zeros(max_distractors, dtype=jnp.float32),
                jnp.zeros(max_distractors, dtype=jnp.float32),
                jnp.zeros(max_distractors, dtype=jnp.float32),
                jnp.zeros(max_distractors, dtype=jnp.float32),
                jnp.zeros(max_distractors, dtype=jnp.float32),
                jnp.zeros(max_distractors, dtype=jnp.float32),
                jnp.zeros(max_distractors, dtype=jnp.int32),
                jnp.zeros(max_distractors, dtype=jnp.int32),
                jnp.zeros(max_distractors, dtype=jnp.int32),
            )
    
    def _select_layout_color(self, key: jax.Array) -> jnp.ndarray:
        """Select layout color from config."""
        layout_colors_list = self.config.layout.layout_colors
        if isinstance(layout_colors_list, str):
            layout_colors_list = (layout_colors_list,)
        
        key, k_layout_color = jax.random.split(key)
        layout_color_idx = jax.random.randint(k_layout_color, (), 0, len(layout_colors_list))
        
        def select_layout_color_by_index(idx):
            branches = [
                lambda: jnp.array(SHAPE_COLORS.get(color_name, (80, 80, 80)), dtype=jnp.uint8)
                for color_name in layout_colors_list
            ]
            return jax.lax.switch(idx, branches)
        
        return select_layout_color_by_index(layout_color_idx)
    
    def _update_animations(
        self,
        state: EnvState,
        phys_state: PhysicsState,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, ...]:
        """Update all animation states."""
        dt = 1.0 / 60.0  # Assume 60 FPS
        
        # Character animation
        is_moving = jnp.abs(phys_state.vx) > 0.1
        n_sprites = len(self.character_sprites) if self.character_sprites else 1
        sprite_idx_new, animation_timer_new = update_animation_state(
            state.sprite_idx,
            state.animation_timer,
            dt,
            self.config.character.animation_fps,
            n_sprites,
            is_moving,
            self.config.character.idle_sprite_idx,
            self.config.character.enable_animation,
        )
        
        # Shape rotation
        if self.config.character.use_shape and self.config.character.shape_rotate:
            shape_angle_new = (state.shape_angle + self.config.character.shape_rotation_speed) % 360.0
        else:
            shape_angle_new = state.shape_angle
        
        # NPC animations
        if self.config.npc.enabled and self.npc_sprite_sets:
            npc_sprite_indices_new, npc_animation_timers_new = update_npc_animation(
                state.npc_sprite_indices,
                state.npc_animation_timers,
                dt,
                self.config.npc.animation_fps,
                self.npc_sprite_counts,
            )
        else:
            npc_sprite_indices_new = state.npc_sprite_indices
            npc_animation_timers_new = state.npc_animation_timers
        
        # Sticky NPC physics and animation
        if self.config.npc.sticky_enabled and self.sticky_npc_sprite_sets:
            sticky_y_new, sticky_vy_new, sticky_on_ground_new = update_sticky_npc_physics(
                key,
                self.config.npc,
                state.sticky_y,
                state.sticky_vy,
                state.sticky_on_ground,
                state.sticky_home_y,
                dt,
                gravity_accel=self.config.physics.gravity,
                jump_velocity=self.config.physics.jump_velocity,
            )
            
            sticky_sprite_indices_new, sticky_animation_timers_new = update_npc_animation(
                state.sticky_sprite_indices,
                state.sticky_animation_timers,
                dt,
                self.config.npc.animation_fps,
                self.sticky_npc_sprite_counts,
            )
        else:
            sticky_y_new = state.sticky_y
            sticky_vy_new = state.sticky_vy
            sticky_on_ground_new = state.sticky_on_ground
            sticky_sprite_indices_new = state.sticky_sprite_indices
            sticky_animation_timers_new = state.sticky_animation_timers
        
        # Distractors
        if self.config.distractors.enabled:
            dist_x_new, dist_y_new, dist_vx_new, dist_vy_new, dist_angles_new = update_distractors(
                state.dist_x,
                state.dist_y,
                state.dist_vx,
                state.dist_vy,
                state.dist_angles,
                state.dist_rot_speeds,
                self.config.W,
                self.config.H,
            )
        else:
            dist_x_new = state.dist_x
            dist_y_new = state.dist_y
            dist_vx_new = state.dist_vx
            dist_vy_new = state.dist_vy
            dist_angles_new = state.dist_angles
        
        return (
            sprite_idx_new, animation_timer_new, shape_angle_new,
            npc_sprite_indices_new, npc_animation_timers_new,
            sticky_y_new, sticky_vy_new, sticky_on_ground_new,
            sticky_sprite_indices_new, sticky_animation_timers_new,
            dist_x_new, dist_y_new, dist_vx_new, dist_vy_new, dist_angles_new
        )

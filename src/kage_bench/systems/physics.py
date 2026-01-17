from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class PhysicsConfig:
    """Continuous physics parameters for platformer.
    
    All velocities and accelerations in pixels per step.
    
    Attributes
    ----------
    gravity : float
        Downward acceleration in px/step^2 (default: 0.5)
    terminal_velocity : float
        Maximum fall speed in px/step (default: 8.0)
    jump_velocity : float
        Initial vertical velocity on jump in px/step (default: -6.0, negative = upward)
    coyote_time : int
        Frames after leaving ground where jump is still allowed (default: 3)
    ground_accel : float
        Acceleration when moving on ground in px/step^2 (default: 1.0)
    air_accel : float
        Acceleration when moving in air in px/step^2 (default: 0.4)
    max_speed_x : float
        Maximum horizontal speed in px/step (default: 3.0)
    ground_friction : float
        Velocity multiplier per step on ground, 0=full stop, 1=no friction (default: 0.8)
    air_resistance : float
        Velocity multiplier per step in air (default: 0.98)
    ground_snap_dist : int
        Max distance to snap down to ground in pixels (default: 2)
    """
    # Gravity
    gravity: float = 0.5              # downward acceleration (px/step^2)
    terminal_velocity: float = 8.0    # max fall speed (px/step)
    
    # Jump
    jump_velocity: float = -6.0       # initial vertical velocity on jump (negative = upward)
    coyote_time: int = 3              # frames after leaving ground where jump is still allowed
    
    # Horizontal movement
    ground_accel: float = 1.0         # acceleration when moving on ground (px/step^2)
    air_accel: float = 0.4            # acceleration when moving in air (px/step^2)
    max_speed_x: float = 3.0          # maximum horizontal speed (px/step)
    
    # Friction/damping
    ground_friction: float = 0.8      # velocity multiplier per step on ground (0=full stop, 1=no friction)
    air_resistance: float = 0.98      # velocity multiplier per step in air
    
    # Ground detection threshold
    ground_snap_dist: int = 2         # max distance to snap down to ground (px)


@struct.dataclass
class PhysicsState:
    """
    Continuous velocity state.
    """
    vx: jnp.ndarray  # horizontal velocity, float32 scalar
    vy: jnp.ndarray  # vertical velocity, float32 scalar
    grounded: jnp.ndarray  # bool scalar, whether agent is on ground
    coyote_timer: jnp.ndarray  # int32 scalar, frames since leaving ground


def is_grounded(
    cx: jnp.ndarray,
    cy: jnp.ndarray,
    solid: jnp.ndarray,
    agent_half_w: int,
    agent_half_h: int,
    snap_dist: int = 2,
) -> jnp.ndarray:
    """
    Check if agent is on ground by testing for solid pixels below feet.
    
    Args:
        cx, cy: agent center in world coords (int32)
        solid: (H, W) bool mask
        agent_half_w: agent half-width in pixels
        agent_half_h: agent half-height in pixels
        snap_dist: how many pixels below to check
        
    Returns:
        bool scalar
    """
    H, W = solid.shape
    foot_y = cy + agent_half_h
    
    # Check pixels directly below agent feet (use width for horizontal extent).
    xs_off = jnp.arange(-agent_half_w, agent_half_w + 1, dtype=jnp.int32)
    xs = jnp.clip(cx + xs_off, 0, W - 1)
    
    # Check from 1 to snap_dist pixels below feet.
    any_ground = jnp.array(False)
    for dy in range(1, snap_dist + 1):
        y_check = jnp.clip(foot_y + dy, 0, H - 1)
        ground_below = jnp.any(solid[y_check, xs])
        any_ground = any_ground | ground_below
    
    return any_ground


def apply_physics(
    x: jnp.ndarray,
    y: jnp.ndarray,
    phys_state: PhysicsState,
    solid: jnp.ndarray,
    agent_half_w: int,
    agent_half_h: int,
    action_x: jnp.ndarray,  # desired horizontal direction: -1, 0, +1
    jump_pressed: jnp.ndarray,  # bool
    cfg: PhysicsConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, PhysicsState]:
    """
    Update position and velocity with continuous physics.
    
    Args:
        x, y: current position (float32 world coords)
        phys_state: current physics state
        solid: (H, W) solid mask
        agent_half_w: agent half-width
        agent_half_h: agent half-height
        action_x: horizontal input {-1, 0, +1}
        jump_pressed: whether jump button is pressed this frame
        cfg: physics configuration
        
    Returns:
        (x_new, y_new, phys_state_new)
        
    Physics order:
        1. Update grounded status and coyote timer
        2. Handle jump input
        3. Apply horizontal acceleration with friction
        4. Apply gravity
        5. Clamp velocities
        6. Integrate position
        7. Collision resolution (caller's responsibility, but we return updated state)
    """
    H, W = solid.shape
    
    cx = jnp.round(x).astype(jnp.int32)
    cy = jnp.round(y).astype(jnp.int32)
    
    # 1. Ground detection.
    # Check spatial contact with ground only if not moving up from previous jump.
    grounded_spatial = is_grounded(cx, cy, solid, agent_half_w, agent_half_h, cfg.ground_snap_dist)
    
    # Count as grounded only if spatial contact AND either:
    # - moving down/stationary (vy >= 0), OR
    # - very close to zero velocity
    # This prevents immediate re-grounding after jump starts.
    grounded = grounded_spatial & (phys_state.vy >= 0.0)
    
    # Update coyote timer.
    coyote_timer = jnp.where(
        grounded,
        jnp.int32(0),
        jnp.minimum(phys_state.coyote_timer + 1, cfg.coyote_time + 1)
    )
    
    can_jump = (coyote_timer <= cfg.coyote_time)
    
    # 2. Jump: only trigger if can jump.
    jump_trigger = jump_pressed & can_jump
    
    vy = jnp.where(
        jump_trigger,
        jnp.float32(cfg.jump_velocity),
        phys_state.vy
    )
    
    # After jump, force grounded = False and consume coyote time.
    grounded = grounded & ~jump_trigger
    coyote_timer = jnp.where(
        jump_trigger,
        jnp.int32(cfg.coyote_time + 1),
        coyote_timer
    )
    
    # 3. Horizontal acceleration.
    # Apply friction first, then add control input.
    friction = jnp.where(grounded, cfg.ground_friction, cfg.air_resistance)
    vx = phys_state.vx * friction
    
    accel = jnp.where(grounded, cfg.ground_accel, cfg.air_accel)
    vx = vx + accel * action_x.astype(jnp.float32)
    
    # Clamp horizontal velocity.
    vx = jnp.clip(vx, -cfg.max_speed_x, cfg.max_speed_x)
    
    # 4. Gravity.
    vy = vy + cfg.gravity
    
    # Clamp vertical velocity.
    vy = jnp.clip(vy, -100.0, cfg.terminal_velocity)  # arbitrary large negative for upward
    
    # 5. Integrate.
    x_new = x + vx
    y_new = y + vy
    
    # Bounds (use appropriate half-sizes for each dimension).
    x_new = jnp.clip(x_new, agent_half_w, W - 1 - agent_half_w)
    y_new = jnp.clip(y_new, agent_half_h, H - 1 - agent_half_h)
    
    phys_state_new = PhysicsState(
        vx=vx,
        vy=vy,
        grounded=grounded,
        coyote_timer=coyote_timer,
    )
    
    return x_new, y_new, phys_state_new


def resolve_collision_with_physics(
    x_prev: jnp.ndarray,
    y_prev: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    phys_state: PhysicsState,
    solid: jnp.ndarray,
    agent_half_w: int,
    agent_half_h: int,
) -> tuple[jnp.ndarray, jnp.ndarray, PhysicsState]:
    """
    Simple collision resolution.
    
    Strategy:
    Axis-separated with backtracking toward previous position (prevents wall climbing):
    1. Resolve horizontal collision by snapping X toward x_prev while keeping y=y_prev.
       If X is adjusted, zero vx.
    2. Resolve vertical collision by snapping Y toward y_prev while keeping x=x_resolved.
       If Y is adjusted, zero vy.
    
    Returns:
        (x_resolved, y_resolved, phys_state_resolved)
    """
    H, W = solid.shape
    cx0 = jnp.round(x_prev).astype(jnp.int32)
    cy0 = jnp.round(y_prev).astype(jnp.int32)
    cx1 = jnp.round(x).astype(jnp.int32)
    cy1 = jnp.round(y).astype(jnp.int32)
    
    xs_off = jnp.arange(-agent_half_w, agent_half_w + 1, dtype=jnp.int32)
    ys_off = jnp.arange(-agent_half_h, agent_half_h + 1, dtype=jnp.int32)
    
    def check_collision(cx_check, cy_check):
        xs = jnp.clip(cx_check + xs_off, 0, W - 1)
        ys = jnp.clip(cy_check + ys_off, 0, H - 1)
        return jnp.any(solid[ys[:, None], xs[None, :]])
    
    vx = phys_state.vx
    vy = phys_state.vy
    
    # Step 1: resolve X by snapping toward previous x while keeping previous y.
    def resolve_x_loop(cx_curr: jnp.ndarray) -> jnp.ndarray:
        def cond(cxv):
            return (cxv != cx0) & check_collision(cxv, cy0)

        def body(cxv):
            step = jnp.sign(cxv - cx0).astype(jnp.int32)
            return cxv - step

        return jax.lax.while_loop(cond, body, cx_curr)

    cx_after_x = jax.lax.cond(
        check_collision(cx1, cy0),
        lambda: resolve_x_loop(cx1),
        lambda: cx1,
    )
    vx_after_x = jnp.where(cx_after_x != cx1, jnp.float32(0.0), vx)

    # Step 2: resolve Y by snapping toward previous y while keeping resolved x.
    def resolve_y_loop(cy_curr: jnp.ndarray) -> jnp.ndarray:
        def cond(cyv):
            return (cyv != cy0) & check_collision(cx_after_x, cyv)

        def body(cyv):
            step = jnp.sign(cyv - cy0).astype(jnp.int32)
            return cyv - step

        return jax.lax.while_loop(cond, body, cy_curr)

    cy_after_y = jax.lax.cond(
        check_collision(cx_after_x, cy1),
        lambda: resolve_y_loop(cy1),
        lambda: cy1,
    )
    vy_after_y = jnp.where(cy_after_y != cy1, jnp.float32(0.0), vy)

    x_resolved = cx_after_x.astype(jnp.float32)
    y_resolved = cy_after_y.astype(jnp.float32)
    
    phys_state_resolved = PhysicsState(
        vx=vx_after_x,
        vy=vy_after_y,
        grounded=phys_state.grounded,
        coyote_timer=phys_state.coyote_timer,
    )
    
    return x_resolved, y_resolved, phys_state_resolved


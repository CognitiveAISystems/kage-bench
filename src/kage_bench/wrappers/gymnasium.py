"""Gymnasium-compatible wrapper for JAX platformer environment.

Adapts the pure functional KAGE_Env to Gymnasium API.
Manages state externally and provides familiar reset()/step() interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core import KAGE_Env, EnvConfig
from ..entities.npc import NPCConfig


try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    gym = None
    spaces = None
    _GYMNASIUM_IMPORT_ERROR = e


_GymBase = gym.Env if gym is not None else object


class KAGE_Env_Gymnasium(_GymBase):
    """Gymnasium wrapper around KAGE_Env.
    
    Provides familiar Gymnasium API (reset, step, render, close) while
    maintaining JAX environment as pure functional core.
    
    Attributes
    ----------
    jax_env : KAGE_Env
        Underlying pure functional JAX environment
    render_mode : Optional[str]
        Render mode ("rgb_array" or None)
    action_space : spaces.Discrete
        Discrete(8) for bitmask actions
    observation_space : spaces.Box
        Box(0, 255, (H, W, 3), uint8) for RGB images
    
    Notes
    -----
    Semantics:
    - Observation: uint8 RGB image (H, W, 3)
    - Action: Discrete(8) bitmask in {0..7} for {LEFT=1, RIGHT=2, JUMP=4}
    - Reward: float32 scalar (from underlying JAX env)
    - Termination: no task termination signal in base env
    - Truncation: time limit (`t >= episode_length`)
    
    Examples
    --------
    Create environment:
    
    >>> import gymnasium as gym
    >>> from kage_bench.wrappers import KAGE_Env_Gymnasium
    >>> 
    >>> env = KAGE_Env_Gymnasium(render_mode="rgb_array", H=128, W=128)
    >>> obs, info = env.reset(seed=42)
    >>> print(obs.shape, info.keys())
    
    Training loop:
    
    >>> for episode in range(100):
    ...     obs, info = env.reset()
    ...     done = False
    ...     while not done:
    ...         action = env.action_space.sample()
    ...         obs, reward, terminated, truncated, info = env.step(action)
    ...         done = terminated or truncated
    """
    
    # Convenience constants (match KAGE_Env)
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 4
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        H: int = 128,
        W: int = 128,
        episode_length: int = 500,
        config_path: Optional[str] = None,
        env_config: Optional[EnvConfig] = None,
        jax_env: Optional[KAGE_Env] = None,
        # Legacy params for backward compatibility
        agent_half: int = 4,
        ground_y: int = 96,
        ground_thickness: int = 2,
        pix_per_unit: int = 2,
        layout_cfg=None,
        camera_cfg=None,
        physics_cfg=None,
        background_cfg=None,
        filter_cfg=None,
        effect_cfg=None,
        char_cfg=None,
        npc_cfg=None,
        distractor_cfg=None,
        # Convenience params for NPCs
        npc_enabled: bool = False,
        npc_dirs: Optional[list[str]] = None,
        npc_min_count: int = 2,
        npc_max_count: int = 5,
        sticky_npc_enabled: bool = False,
        sticky_npc_count: int = 1,
        sticky_npc_dirs: Optional[list[str]] = None,
        sticky_npc_can_jump: bool = False,
        sticky_npc_jump_probability: float = 0.01,
        sticky_npc_x_offsets: Optional[list[int]] = None,
        sticky_npc_y_randomize: bool = True,
        sticky_npc_y_min_offset: int = -30,
        sticky_npc_y_max_offset: int = 0,
    ):
        """Initialize Gymnasium wrapper.
        
        Parameters
        ----------
        render_mode : Optional[str]
        config_path : Optional[str]
            Path to YAML configuration file (load base config)
        H : int
        W : int
        episode_length : int
        env_config : Optional[EnvConfig]
            Complete EnvConfig object (overrides all loading)
        jax_env : Optional[KAGE_Env]
            Pre-built JAX environment to wrap (overrides config loading)
        **kwargs
            Legacy parameters
        """
        if gym is None or spaces is None:  # pragma: no cover
            raise ImportError(
                "gymnasium is required to use KAGE_Env_Gymnasium. "
                "Install via `pip install gymnasium`."
            ) from _GYMNASIUM_IMPORT_ERROR
        
        self.render_mode = render_mode
        
        if jax_env is None:
            # Priority:
            # 1. env_config object (explicit)
            # 2. config_path (YAML)
            # 3. Arguments (H, W, etc.) override defaults or loaded config
            if env_config is None:
                # Load from YAML if provided
                if config_path:
                    from ..utils.config_loader import load_config_from_yaml
                    env_config = load_config_from_yaml(config_path)
                else:
                    env_config = EnvConfig()
                
                # Helper to optionally override if argument differs from default
                # We assume if argument is not default, it was user-provided
                # (Limitation: can't distinguish explicit default vs default-default)
                # But checking against default args is reasonable.
                
                # For simplicity, we only apply overrides for complex configs if explicitly passed
                if layout_cfg: env_config.layout = layout_cfg
                if camera_cfg: env_config.camera = camera_cfg
                if physics_cfg: env_config.physics = physics_cfg
                if char_cfg: env_config.character = char_cfg
                if background_cfg: env_config.background = background_cfg
                if filter_cfg: env_config.filters = filter_cfg
                if effect_cfg: env_config.effects = effect_cfg
                if npc_cfg: env_config.npc = npc_cfg
                if distractor_cfg: env_config.distractors = distractor_cfg
                
                # Apply top-level overrides if they differ from EnvConfig default
                # Note: H and W in EnvConfig init default to 128
                if H != 128: env_config.H = H
                if W != 128: env_config.W = W
                if episode_length != 500: env_config.episode_length = episode_length
                
                # Process NPC arguments if not using explicit config
                if npc_cfg is None and (npc_enabled or sticky_npc_enabled):
                     # This logic constructs NPCConfig from flat args
                     # We should only do this if npc_cfg wasn't loaded from YAML either
                     # If YAML has enabled=True, we might want to respect that.
                     # But flat args are essentially overrides.
                     
                     # Create a temporary config from arguments
                     arg_npc_config = NPCConfig(
                        enabled=npc_enabled,
                        sprite_dirs=npc_dirs or [],
                        min_npc_count=npc_min_count,
                        max_npc_count=npc_max_count,
                        sticky_enabled=sticky_npc_enabled,
                        min_sticky_count=sticky_npc_count,
                        max_sticky_count=sticky_npc_count,
                        sticky_sprite_dirs=sticky_npc_dirs or [],
                        sticky_can_jump=sticky_npc_can_jump,
                        sticky_jump_probability=sticky_npc_jump_probability,
                        sticky_x_offsets=sticky_npc_x_offsets or [-40, 40],
                        sticky_y_randomize=sticky_npc_y_randomize,
                        sticky_y_min_offset=sticky_npc_y_min_offset,
                        sticky_y_max_offset=sticky_npc_y_max_offset,
                    )
                     
                     # Using a simple heuristic: if arguments enable NPCs, they override YAML disabled
                     if arg_npc_config.enabled or arg_npc_config.sticky_enabled:
                         env_config.npc = arg_npc_config
            
            # Create JAX environment
            self.jax_env = KAGE_Env(env_config)
        else:
            self.jax_env = jax_env
        
        # Create JIT-compiled versions for performance
        # Without these, JAX recompiles on every call which is very slow
        self._reset_jit = jax.jit(self.jax_env.reset)
        self._step_jit = jax.jit(self.jax_env.step)
        self._render_jit = jax.jit(self.jax_env.render)
        
        # Gymnasium spaces
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.jax_env.config.H, self.jax_env.config.W, 3),
            dtype=np.uint8,
        )
        
        # Internal state
        self._state = None
        self._key = None
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.
        
        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducibility
        options : Optional[Dict[str, Any]]
            Additional options (unused)
        
        Returns
        -------
        obs : np.ndarray
            Initial observation, shape (H, W, 3), dtype uint8
        info : Dict[str, Any]
            Info dictionary with seed
        
        Examples
        --------
        >>> obs, info = env.reset(seed=42)
        >>> print(obs.shape, info["seed"])
        (128, 128, 3) 42
        """
        # Mirror Gymnasium seeding semantics
        if gym is not None:
            super().reset(seed=seed, options=options)  # type: ignore[misc]
        else:  # pragma: no cover
            del options
        
        if seed is None:
            # Derive a 32-bit seed without relying on gymnasium internals
            seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        seed32 = int(seed) & 0xFFFFFFFF
        if seed is not None:
             self._key = jax.random.PRNGKey(seed)
        elif self._key is None:
             self._key = jax.random.PRNGKey(0)
             
        # Execute JAX reset (returns obs, info)
        obs, info = self._reset_jit(self._key)
        
        # Extract state from info for internal use
        # JAX info is usually a pytree, so "state" should be there
        self._state = info["state"]
        
        # Convert to numpy for Gymnasium API
        obs_np = np.asarray(obs)
        
        # Clean up info for return (Gymnasium expects specific serialization, 
        # but passing JAX pytree state in info might break some wrappers.
        # We'll just return basic info + seed.)
        return_info = {"seed": seed32}
        
        return obs_np, return_info
    
    def step(self, action: int):
        """Execute one environment step.
        
        Parameters
        ----------
        action : int
            Action bitmask in {0..7}
        
        Returns
        -------
        obs : np.ndarray
            Observation, shape (H, W, 3), dtype uint8
        reward : float
            Scalar reward
        terminated : bool
            Whether episode ended (always False for this env)
        truncated : bool
            Whether episode was truncated (time limit)
        info : Dict[str, Any]
            Info dictionary with agent state
        
        Examples
        --------
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> print(reward, terminated, truncated, info["x"])
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        
        a = int(action)
        if not (0 <= a <= 7):
            raise ValueError(f"Action must be in [0, 7] (bitmask), got {a}.")
        
        # Execute JAX step
        obs, reward, terminated, truncated, info = self._step_jit(self._state, jnp.int32(a))
        
        # Update internal state
        self._state = info["state"]

        # Convert to numpy/python scalars for Gymnasium API
        # self._step_jit already returns rendered obs if render=True in call, 
        # but Wrapper init sets valid render mode. 
        # KAGE_Env.step now returns obs by default.
        obs_np = np.asarray(obs)
        reward_py = float(reward)
        terminated_py = bool(terminated)
        truncated_py = bool(truncated)
        
        
        # Extract scalar info for logging/debugging.
        return_info = {
            "return": float(jax.device_get(info["return"])),
            "timestep": int(jax.device_get(info["timestep"])),
            "x": float(jax.device_get(info["x"])),
            "y": float(jax.device_get(info["y"])),
            "vx": float(jax.device_get(info["vx"])),
            "vy": float(jax.device_get(info["vy"])),
            "vx_mean": float(jax.device_get(info["vx_mean"])),
            "vy_mean": float(jax.device_get(info["vy_mean"])),
            "jump_count": int(jax.device_get(info["jump_count"])),
            "passed_distance": float(jax.device_get(info["passed_distance"])),
            "progress": float(jax.device_get(info["progress"])),
            "success": bool(jax.device_get(info["success"])),
            "success_once": bool(jax.device_get(info["success_once"])),
            "camera_x": float(jax.device_get(self._state.camera_x)),
            "t": int(jax.device_get(self._state.t)),
            "x_max": float(jax.device_get(self._state.x_max)),
        }
        
        return obs_np, reward_py, terminated_py, truncated_py, return_info
    
    def render(self):
        """Render current state.

        Returns
        -------
        img : Optional[np.ndarray]
            RGB image if state exists, None otherwise

        Examples
        --------
        >>> img = env.render()
        >>> if img is not None:
        ...     print(img.shape, img.dtype)
        (128, 128, 3) uint8
        """
        if self._state is None:
            return None
        return np.array(jax.device_get(self._render_jit(self._state)), copy=False)
    
    def close(self):
        """Close environment (no-op for JAX env)."""
        return None


class GymnasiumWrapper(KAGE_Env_Gymnasium):
    """Gymnasium wrapper for an existing KAGE_Env instance."""

    def __init__(self, jax_env: KAGE_Env, *, render_mode: Optional[str] = None):
        super().__init__(render_mode=render_mode, jax_env=jax_env)

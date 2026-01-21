"""JAX-compatible wrappers for vectorized environments."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any, Tuple, Dict, NamedTuple, Optional
from flax import struct

from ..core import KAGE_Env, EnvState


class AutoResetWrapper:
    """
    Auto-reset wrapper for JAX environment.
    
    When an episode ends (terminated or truncated), the environment is automatically
    reset using a fresh random key. The observation returned is the initial observation
    of the new episode, effectively creating an infinite horizon environment.
    
    Notes:
        - The `step` method signature changes: it requires `key` argument for potential reset.
        - reset() returns `(obs, info)`
        - step() returns `(obs, reward, terminated, truncated, info)`
        - The returned `obs` is from the *next* state (or reset state if done).
        - The `info` dict contains "reset_obs" if a reset occurred, which is 
          useful for bootstrapping if needed, but not strictly required.
          Actually, standard practice is:
          - obs: observation of the new reset state (if done) or next state (if not done)
          - reward: reward of the last step
          - terminated/truncated: true for the last step
          
          This allows the agent to see the "final" transition before swap, 
          but for value estimation of the *new* state, `obs` is correct.
    """
    
    def __init__(self, env: KAGE_Env):
        self.env = env
    
    def reset(self, key: jax.Array) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Reset environment normally."""
        return self.env.reset(key)
    
    def step(
        self, state: EnvState, action: jax.Array, key: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Step environment with auto-reset.
        
        Args:
            state: Current state
            action: Action to take
            key: PRNG key for potential reset (consumed only if reset needed? 
                 No, usually we consume it unconditionally or split it.)
                 
        Returns:
            obs: Observation (reset obs if done, else next obs)
            reward: Reward from step
            terminated: Termination flag
            truncated: Truncation flag
            info: Info dict
        """
        # 1. Step environment
        obs_st, reward, terminated, truncated, info_st = self.env.step(state, action)
        
        # 2. Check done
        done = terminated | truncated
        
        # 3. Define reset function
        def perform_reset(k):
            return self.env.reset(k)
            
        def no_reset(k):
            # Return dummy structure matching reset output
            # We need (obs, info)
            # info contains {"state": state}
            next_state = info_st["state"]
            return obs_st, {"state": next_state}
            
        # 4. Conditionally reset
        # We use lax.cond to execute reset logic
        # Note: reset(key) returns (initial_obs, info_with_initial_state)
        # We ignore 'info' from step if done, and use info from reset.
        # BUT we still need 'reward', 'terminated', 'truncated' from the step *before* reset.
        
        reset_obs, reset_info = jax.lax.cond(
            done,
            perform_reset,
            no_reset,
            key
        )
        
        # 5. Select output
        # If done, we return reset_obs. If not done, we return obs_st.
        # (This logic is already handled by whatever 'perform_reset' or 'no_reset' returns)
        final_obs = reset_obs
        final_state = reset_info["state"]
        
        # 6. Construct info
        # We might want to preserve the *terminal* observation in info["final_observation"]
        # for proper value estimation (bootstrap from terminal state).
        # We also need to return the 'state' corresponding to 'final_obs' for the next loop iter.
        
        final_info = {
            "state": final_state,
             # If done, store the transition info (terminal observation)
             # If not done, this field is just ignored or set to current obs
            "final_observation": obs_st, 
            "final_state": info_st["state"] # The state *before* reset (terminal state)
        }
        
        # Note: If not done, final_obs == obs_st == final_info["final_observation"]
        
        return final_obs, reward, terminated, truncated, final_info

    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped environment."""
        return getattr(self.env, name)


@struct.dataclass
class FrameStackState:
    env_state: EnvState
    frames: jnp.ndarray


class FrameStackWrapper:
    """Stack the last N observations along the channel dimension."""

    def __init__(self, env, num_frames: int = 4):
        self.env = env
        self.num_frames = num_frames

    def reset(self, key: jax.Array) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(key)
        frames = jnp.repeat(obs[None, ...], self.num_frames, axis=0)
        stacked = jnp.concatenate(list(frames), axis=-1)
        info["state"] = FrameStackState(env_state=info["state"], frames=frames)
        return stacked, info

    def step(
        self, state: FrameStackState, action: jax.Array, key: Optional[jax.Array] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        if key is None:
            obs, reward, terminated, truncated, info = self.env.step(state.env_state, action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(state.env_state, action, key)

        done = terminated | truncated

        frames = jax.lax.cond(
            done,
            lambda _: jnp.repeat(obs[None, ...], self.num_frames, axis=0),
            lambda _: jnp.concatenate([state.frames[1:], obs[None, ...]], axis=0),
            operand=None,
        )
        stacked = jnp.concatenate(list(frames), axis=-1)

        info["state"] = FrameStackState(env_state=info["state"], frames=frames)
        return stacked, reward, terminated, truncated, info

    @property
    def observation_space(self):
        try:
            from gymnasium import spaces
            base = self.env.observation_space
            if base is None:
                return None
            h, w, c = base.shape
            return spaces.Box(
                low=0,
                high=255,
                shape=(h, w, c * self.num_frames),
                dtype=base.dtype,
            )
        except ImportError:
            return None

    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped environment."""
        return getattr(self.env, name)


@struct.dataclass
class RewardNormState:
    env_state: Any
    ret: jnp.ndarray
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


class RewardNormalizeWrapper:
    """Normalize rewards using running stats of discounted returns."""

    def __init__(self, env, gamma: float = 0.99, eps: float = 1e-8):
        self.env = env
        self.gamma = gamma
        self.eps = eps

    def reset(self, key: jax.Array) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(key)
        ret = jnp.array(0.0, dtype=jnp.float32)
        mean = jnp.array(0.0, dtype=jnp.float32)
        var = jnp.array(1.0, dtype=jnp.float32)
        count = jnp.array(1e-4, dtype=jnp.float32)
        info["state"] = RewardNormState(
            env_state=info["state"],
            ret=ret,
            mean=mean,
            var=var,
            count=count,
        )
        return obs, info

    def step(
        self, state: RewardNormState, action: jax.Array, key: Optional[jax.Array] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        if key is None:
            obs, reward, terminated, truncated, info = self.env.step(state.env_state, action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(state.env_state, action, key)

        done_f = (terminated | truncated).astype(jnp.float32)
        ret = state.ret * self.gamma * (1.0 - done_f) + reward

        delta = ret - state.mean
        total_count = state.count + 1.0
        new_mean = state.mean + delta / total_count
        m_a = state.var * state.count
        m_b = delta * (ret - new_mean)
        new_var = (m_a + m_b) / total_count
        new_count = total_count

        reward_norm = reward / (jnp.sqrt(new_var) + self.eps)

        info["state"] = RewardNormState(
            env_state=info["state"],
            ret=ret,
            mean=new_mean,
            var=new_var,
            count=new_count,
        )
        return obs, reward_norm, terminated, truncated, info

    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped environment."""
        return getattr(self.env, name)


@struct.dataclass
class RewardClipState:
    env_state: Any


class RewardClipWrapper:
    """Clip rewards to a fixed range."""

    def __init__(self, env, min_reward: float = -1.0, max_reward: float = 1.0):
        self.env = env
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reset(self, key: jax.Array) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(key)
        info["state"] = RewardClipState(env_state=info["state"])
        return obs, info

    def step(
        self, state: RewardClipState, action: jax.Array, key: Optional[jax.Array] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        if key is None:
            obs, reward, terminated, truncated, info = self.env.step(state.env_state, action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(state.env_state, action, key)

        reward = jnp.clip(reward, self.min_reward, self.max_reward)
        info["state"] = RewardClipState(env_state=info["state"])
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped environment."""
        return getattr(self.env, name)


@struct.dataclass
class RewardStatsState:
    env_state: Any
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray


class RewardStatsWrapper:
    """Track raw episode returns and lengths without modifying rewards."""

    def __init__(self, env):
        self.env = env

    def reset(self, key: jax.Array) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(key)
        zero_return = jnp.array(0.0, dtype=jnp.float32)
        zero_length = jnp.array(0, dtype=jnp.int32)
        info["state"] = RewardStatsState(
            env_state=info["state"],
            episode_returns=zero_return,
            episode_lengths=zero_length,
        )
        info["episode_returns"] = zero_return
        info["episode_lengths"] = zero_length
        info["returned_episode_returns"] = zero_return
        info["returned_episode_lengths"] = zero_length
        return obs, info

    def step(
        self, state: RewardStatsState, action: jax.Array, key: Optional[jax.Array] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        if key is None:
            obs, reward, terminated, truncated, info = self.env.step(state.env_state, action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(state.env_state, action, key)

        done = terminated | truncated
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1

        returned_episode_returns = jnp.where(done, new_episode_return, jnp.float32(0.0))
        returned_episode_lengths = jnp.where(done, new_episode_length, jnp.int32(0))
        returned_episode_times = jnp.where(done, new_episode_length.astype(jnp.float32), jnp.float32(0.0))
        episode_returns = jnp.where(done, jnp.float32(0.0), new_episode_return)
        episode_lengths = jnp.where(done, jnp.int32(0), new_episode_length)

        info["state"] = RewardStatsState(
            env_state=info["state"],
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
        )
        info["episode_returns"] = episode_returns
        info["episode_lengths"] = episode_lengths
        info["returned_episode_returns"] = returned_episode_returns
        info["returned_episode_lengths"] = returned_episode_lengths
        info["raw_reward"] = reward
        info["return"] = episode_returns
        info["length"] = episode_lengths
        info["episode"] = {
            "r": returned_episode_returns,
            "l": returned_episode_lengths,
            "t": returned_episode_times,
        }
        info["_episode"] = done
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped environment."""
        return getattr(self.env, name)


class LogWrapper:
    """Wrapper to log episode returns and lengths."""
    
    def __init__(self, env: KAGE_Env):
        self.env = env
        
    def reset(self, key: jax.Array) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(key)
        
        # Add logging state to info
        # We assume the user will separate this out or pass it through
        # But wait, platformer env passes 'state' in info.
        # We need to maintain separate logging state. 
        # But 'state' passed to step includes everything?
        # NO, 'state' arg to step is EnvState.
        # We can't easily attach extra state to EnvState without modifying EnvState definition or Wrapper logic.
        
        # Approach: We expect the user to manage 'log_state' externally? 
        # Or we bundle 'log_state' into the 'state' passed to step?
        # If we bundle, then 'step' signature of inner env expects EnvState, not tuple.
        # So typically LogWrapper changes the State type to (EnvState, LogState).
        
        obs, info = self.env.reset(key)
        state = info["state"]
        
        log_state = LogEnvState(
            env_state=state,
            episode_returns=jnp.float32(0.0),
            episode_lengths=jnp.int32(0),
            won_episode=jnp.bool_(False),
            timestep=jnp.int32(0)
        )
        
        info["state"] = log_state
        info["returned_episode_returns"] = jnp.float32(0.0)
        info["returned_episode_lengths"] = jnp.int32(0)
        info["timestep"] = jnp.int32(0)
        info["returned_episode"] = jnp.bool_(False)
        
        return obs, info

    def step(self, state, action, key=None):
        # Unwrap state
        env_state = state.env_state
        
        # Handle key argument for AutoResetWrapper compatibility
        if key is not None:
            # If wrapped by AutoResetWrapper, step takes key
            if hasattr(self.env, "step") and hasattr(self.env.step, "__code__") and "key" in self.env.step.__code__.co_varnames:
                 # Logic for standard python functions
                 obs, reward, terminated, truncated, info = self.env.step(env_state, action, key)
            else:
                 # Check if the step method signature accepts key (e.g. AutoResetWrapper)
                 # Simpler: Try calling with key, if fail, try without? No, JIT fails.
                 # Convention: If LogWrapper is used with AutoResetWrapper, key is required.
                 # We assume self.env handles it.
                 obs, reward, terminated, truncated, info = self.env.step(env_state, action, key)
        else:
            obs, reward, terminated, truncated, info = self.env.step(env_state, action)
            
        new_env_state = info["state"]
        done = terminated | truncated
        
        # Update logging state (accumulate reward, length)
        # Note: if done, we might want to reset accumulators.
        # But if AutoResetWrapper is *outer*, then 'env_state' is already reset?
        # If LogWrapper is *inner* (closer to env), then 'env_state' is terminal (not reset yet).
        # Usually LogWrapper is OUTSIDE AutoResetWrapper?
        # If LogWrapper is outside, then 'env_state' is reset state if done.
        
        # Let's assume LogWrapper is OUTSIDE AutoResetWrapper.
        # So 'reward' is from the last step.
        # 'done' is true.
        # 'new_env_state' is the FRESH state (t=0).
        
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        
        # Limit the return value if not done (masking)
        returned_episode_returns = jnp.where(done, new_episode_return, jnp.float32(0.0))
        returned_episode_lengths = jnp.where(done, new_episode_length, jnp.int32(0))
        
        # Reset accumulators if done
        final_episode_return = jnp.where(done, jnp.float32(0.0), new_episode_return)
        final_episode_length = jnp.where(done, jnp.int32(0), new_episode_length)
        
        new_log_state = LogEnvState(
            env_state=new_env_state,
            episode_returns=final_episode_return,
            episode_lengths=final_episode_length,
            won_episode=jnp.bool_(False), # Placeholder
            timestep=state.timestep + 1
        )
        
        info["state"] = new_log_state
        info["returned_episode_returns"] = returned_episode_returns
        info["returned_episode_lengths"] = returned_episode_lengths
        info["timestep"] = state.timestep + 1
        info["returned_episode"] = done
        
        return obs, reward, terminated, truncated, info

@struct.dataclass
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    won_episode: bool
    timestep: int

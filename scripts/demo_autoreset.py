import jax
import jax.numpy as jnp
from kage_bench import KAGE_Env, AutoResetWrapper, LogWrapper
from kage_bench.utils.config_loader import load_config_from_yaml


# Wrapper to force termination for testing
class MockTerminationWrapper:
    def __init__(self, env):
        self.env = env
        
    def reset(self, key):
        return self.env.reset(key)
        
    def step(self, state, action):
        # We hijack action 99 to mean "force terminate"
        obs, reward, terminated, truncated, info = self.env.step(state, action)
        
        # If action is 99, force termination
        # Note: 99 is outside normal range (0-7), but KAGE_Env handles actions loosely (usually just movement)
        # We assume action logic doesn't crash on 99.
        force_term = (action == 99)
        terminated = terminated | force_term
        
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)

def main():
    # 1. Load Config & Create Environment
    config = load_config_from_yaml("custom_config.yaml")
    
    # Use short episode length to trigger resets quickly for demo
    config.episode_length = 50
    
    env = KAGE_Env(config)
    env = MockTerminationWrapper(env) # Add mock termination
    env = AutoResetWrapper(env)
    env = LogWrapper(env)
    
    # 2. Vectorize and JIT
    # Note: wrapper step signature is (state, action, key)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    
    # 3. Initialize
    N_ENVS = 4
    key = jax.random.PRNGKey(42)
    key_reset, key_act = jax.random.split(key)
    keys = jax.random.split(key_reset, N_ENVS)
    
    obs, info = reset_fn(keys)
    state = info["state"]  # This is LogEnvState now
    
    print(f"Initial T: {state.env_state.t}") 
    
    # 4. Loop
    print("\nStarting loop...")
    
    # We need a stream of keys for potential resets in each step
    key_loop = key_act
    
    for i in range(120): # Run enough to trigger multiple resets (epi length 50)
        key_loop, key_step, key_action = jax.random.split(key_loop, 3)
        keys_step = jax.random.split(key_step, N_ENVS)
        
        # Dummy actions
        actions = jax.random.randint(key_action, (N_ENVS,), 0, 8).astype(jnp.int32)
        
        # Force terminate env 0 at step 20
        if i == 20:
             print("\n!!! FORCING TERMINATION OF ENV 0 !!!\n")
             actions = actions.at[0].set(99)
        
        obs, reward, terminated, truncated, info = step_fn(state, actions, keys_step)
        state = info["state"]
        
        # Check logs
        if "returned_episode_returns" in info:
             # Mask out zeros (only show finished episodes)
             valid_mask = info["returned_episode_returns"] != 0
             if jnp.any(valid_mask):
                 print(f"Step {i}: Episode Finished!")
                 print(f"  Returns: {info['returned_episode_returns'][valid_mask]}")
                 print(f"  Lengths: {info['returned_episode_lengths'][valid_mask]}")
                 
        # Check T values to confirm reset happened (T should wrap around)
        # We print T for ALL envs to see the difference
        if i % 5 == 0 or i == 21:
             print(f"Step {i}: T values = {state.env_state.t}")

    print("\nDone!")

if __name__ == "__main__":
    main()

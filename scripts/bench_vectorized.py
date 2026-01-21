import time
import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

import os
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from kage_bench import KAGE_Env, EnvConfig, load_config_from_yaml


def main(
    n_envs: int = 1024,
    n_steps: int = 1000,
    warmup_steps: int = 3,
    config_path: str = None,
    block_size: int = 100,
) -> int:
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print()

    # Create environment with config
    if config_path:
        print(f"Loading config from: {config_path}")
        config = load_config_from_yaml(config_path)
    else:
        config = EnvConfig()
    
    print(f"Config Screen: {config.H}x{config.W}")
    env = KAGE_Env(config)

    # 1. Define the vectorized step function
    step_vmap = jax.vmap(env.step, in_axes=(0, 0))

    # 2. Define a block of steps using lax.scan
    # This is the "secret sauce" for high performance in JAX
    @jax.jit
    def run_block(state, keys):
        def scan_fn(carry_state, step_key):
            # Generate random actions inside the JIT function
            # Split for each environment
            env_keys = jax.random.split(step_key, n_envs)
            actions = jax.vmap(lambda k: jax.random.randint(k, (), 0, 8))(env_keys)
            
            # Perform vectorized step
            obs, reward, terminated, truncated, info = step_vmap(carry_state, actions)
            
            # In a real training loop, you would handle auto-reset here.
            # For pure benchmark, we just return the new state.
            return info["state"], None 

        final_state, _ = jax.lax.scan(scan_fn, state, keys)
        return final_state

    # Initialize envs
    master_key = jax.random.PRNGKey(0)
    reset_keys = jax.random.split(master_key, n_envs)
    _, info = jax.jit(jax.vmap(env.reset))(reset_keys)
    states = info["state"]
    
    print(f"Reset {n_envs} envs.")

    # Prepare keys for blocks
    # We split into (n_steps // block_size, block_size) keys
    n_blocks = n_steps // block_size
    rng_key = jax.random.PRNGKey(42)
    block_rng_keys = jax.random.split(rng_key, n_steps).reshape(n_blocks, block_size, 2)

    # Compilation
    print(f"Compiling JIT block (block_size={block_size})...")
    states = run_block(states, block_rng_keys[0])
    jax.block_until_ready(states.x)
    print("Compilation complete!")

    # Warmup
    print(f"Warming up ({warmup_steps} blocks)...")
    for i in range(min(warmup_steps, n_blocks)):
        states = run_block(states, block_rng_keys[i])
    jax.block_until_ready(states.x)
    print("Warmup complete, starting benchmark...\n")

    # Benchmark loop
    t0 = time.perf_counter()
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
    )
    with progress:
        task_id = progress.add_task("Benchmark", total=n_blocks)
        for i in range(n_blocks):
            progress.update(task_id, advance=1)
            states = run_block(states, block_rng_keys[i])
    
    # Final sync
    jax.block_until_ready(states.x)
    elapsed = time.perf_counter() - t0

    # Total frames = n_envs * n_blocks * block_size
    total_steps = n_envs * n_blocks * block_size
    fps = total_steps / elapsed

    print()
    print(f"--- Results for {n_envs} parallel envs ---")
    print(f"Total steps: {total_steps}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Steps/sec (per env): {total_steps / (n_envs * elapsed):.1f}")
    print(f"FPS (Total throughput): {fps:.1f}")
    print(f"-------------------------------------------")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Vectorized JAX benchmark")
    parser.add_argument("--n_envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of benchmark steps")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--block_size", type=int, default=100, help="Number of steps per JIT call")
    args = parser.parse_args()
    
    main(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        config_path=args.config,
        block_size=args.block_size
    )

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from kage_bench import KAGE_Env, load_config_from_yaml
import os

def main():
    config_path = "src/kage_bench/configs/default_config.yaml"
    if not os.path.exists(config_path):
        print(f"Config {config_path} not found.")
        return

    print(f"Loading config: {config_path}")
    config = load_config_from_yaml(config_path)
    env = KAGE_Env(config)

    # Reset with a fixed key for reproducibility
    key = jax.random.PRNGKey(42)
    obs, info = jax.jit(env.reset)(key)
    
    # Take one step
    key, subkey = jax.random.split(key)
    action = jnp.array(1) # Move right
    obs, reward, terminated, truncated, info = jax.jit(env.step)(info["state"], action, render=True)

    # Save observation
    img = Image.fromarray(np.array(obs))
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    output_path = "tmp/verification_obs.png"
    img.save(output_path)
    print(f"Saved sample observation to {output_path}")

if __name__ == "__main__":
    main()

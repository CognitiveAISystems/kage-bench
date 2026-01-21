"""
Play JAX platformer using a YAML configuration file.

Usage:
    python play_human_config.py path/to/config.yaml
"""
import argparse
import time
import os
import pygame
import numpy as np
import jax
import jax.numpy as jnp

from kage_bench import KAGE_Env_Gymnasium, KAGE_Env
from kage_bench.utils.config_loader import load_config_from_yaml
from kage_bench.systems.generation.background import COLOR_PALETTE
from kage_bench.core import KAGE_Env as CoreEnv

def decode_action(keys) -> int:
    """
    Decode keyboard input to action.
    """
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    jump = keys[pygame.K_w] or keys[pygame.K_SPACE] or keys[pygame.K_UP]

    # Bitmask: allow combining jump + horizontal in the same frame.
    a = CoreEnv.NOOP
    a |= CoreEnv.LEFT if left else 0
    a |= CoreEnv.RIGHT if right else 0
    a |= CoreEnv.JUMP if jump else 0
    return a

def main():
    parser = argparse.ArgumentParser(description="Play JAX Platformer with YAML Config")
    parser.add_argument("config_path", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load configuration
    try:
        print(f"Loading configuration from {args.config_path}...")
        env_config = load_config_from_yaml(args.config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Initialize Environment
    # We use KAGE_Env_Gymnasium wrapper for convenience (handles resizing/rendering buffer)
    # But wait, KAGE_Env_Gymnasium constructs its own KAGE_Env internally.
    # It usually takes kwargs to build config. 
    # Let's see if we can pass the pre-loaded config directly or if we need to reconstruct.
    
    # Looking at gymnasium.py wrapper:
    # def __init__(self, config_path=None, env_config=None, ...params...)
    # It seems to support passing env_config directly or building it.
    # However, existing wrapper might expect params.
    
    # Actually, a cleaner way for this script is to use the wrapper but pass the config we just loaded.
    # Checking KAGE_Env_Gymnasium.__init__ signature in existing code... 
    # It likely builds config from kwargs.
    
    # Alternative: Instantiate low-level KAGE_Env(env_config) manually and wrap it or just use it directly.
    # Since this is a custom runner, we can control the loop.
    
    # Let's instantiate the core environment directly to ensure our custom config is respected exactly 1:1
    env = KAGE_Env(env_config)
    
    # Initialize PyGame
    pygame.init()
    
    SCALE = 6
    W, H = env_config.W * SCALE, env_config.H * SCALE
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"JAX Platformer - {os.path.basename(args.config_path)}")
    
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()
    
    # JIT-compile the environment functions for performance
    # This captures 'env' and its assets as constants in the compiled graph
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # Reset function
    def do_reset():
        # Generate random seed
        seed = int(time.time_ns()) & 0xFFFFFFFF
        key = jax.random.PRNGKey(seed)
        obs, info = jit_reset(key)
        return obs, info, seed

    # Initial Reset
    obs, info, current_seed = do_reset()
    state = info["state"]
    
    running = True
    episode_return = 0.0
    last_reward = 0.0
    t = 0
    
    print("\nControls:")
    print("  A/D/Arrow keys : Move")
    print("  Space/W/Up     : Jump")
    print("  R              : Reset")
    print("  Q/Esc          : Quit")
    
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info, current_seed = do_reset()
                    state = info["state"]
                    episode_return = 0.0
                    t = 0

        # Input
        keys = pygame.key.get_pressed()
        action_mask = decode_action(keys)
        action_jax = jnp.int32(action_mask)

        # Step
        # env.step returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = jit_step(state, action_jax, render=True)
        
        # Update state for next loop
        state = info["state"]
        
        last_reward = float(reward)
        episode_return += last_reward
        t += 1
        
        # Check Done
        done = bool(terminated or truncated)
        if done:
            obs, info, current_seed = do_reset()
            state = info["state"]
            episode_return = 0.0
            last_reward = 0.0
            t = 0
            
        # Render
        # obs is (H, W, 3)
        img = np.array(obs) # Convert to numpy for pygame
        
        # Transpose to (W, H, 3) for pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (W, H))
        screen.blit(surf, (0, 0))
        
        # HUD
        agent_x = float(state.x)
        agent_y = float(state.y)
        
        lines = [
            f"Config: {os.path.basename(args.config_path)}",
            f"t={t:5d}  seed={current_seed}",
            f"x={agent_x:8.2f} y={agent_y:8.2f}",
            f"rew={last_reward:10.3f}  ret={episode_return:10.3f}",
        ]
        
        for i, s in enumerate(lines):
            txt = font.render(s, True, (255, 255, 255))
            screen.blit(txt, (6, 6 + 18 * i))

        pygame.display.flip()
        clock.tick(60) # 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()

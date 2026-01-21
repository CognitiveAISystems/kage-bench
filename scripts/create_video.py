import argparse
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn
from kage_bench import KAGE_Env, load_config_from_yaml

def save_video_file(frames: list[np.ndarray], filename: str, fps: int = 30):
    """Save frames as both MP4 video and GIF."""
    # 1. Try saving MP4
    mp4_path = filename if filename.endswith('.mp4') else filename + '.mp4'
    try:
        import imageio
        writer = imageio.get_writer(mp4_path, fps=fps, codec='libx264', pixelformat='yuv420p')
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved to {mp4_path}")
    except ValueError as e:
        print(f"Error saving MP4: {e}")
        print("Note: To save MP4, you may need to run: uv sync --extra video")
    except ImportError:
        print("Error: imageio not installed. Cannot save MP4.")

    # 2. Try saving GIF
    gif_path = mp4_path.replace('.mp4', '.gif')
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000 // fps,
            loop=0
        )
        print(f"GIF saved to {gif_path}")
    except ImportError:
        try:
            import imageio
            imageio.mimsave(gif_path, frames, fps=fps)
            print(f"GIF saved to {gif_path} (via imageio)")
        except Exception as e:
            print(f"Error saving GIF: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run JAX Platformer from YAML and save video")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to run")
    parser.add_argument("--output", type=str, default="tmp/video.mp4", help="Output video filename")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--action-mode",
        choices=["random", "jump_forward", "forward", "idle"],
        default="jump_forward",
        help="How to pick actions for the demo video.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    print(f"Loading config from {args.config}...")
    config = load_config_from_yaml(args.config)
    env = KAGE_Env(config)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # JIT compile environment functions
    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    print(f"Initializing environment with seed {args.seed}...")
    key = jax.random.PRNGKey(args.seed)
    obs, info = reset_jit(key)
    state = info["state"]
    
    frames = []
    frames.append(np.array(obs))

    print(f"Running for {args.steps} steps...")
    
    # Build a simple action policy
    action_key = jax.random.PRNGKey(args.seed + 1)
    jump_forward_action = jnp.int32(env.JUMP | env.RIGHT)
    forward_action = jnp.int32(env.RIGHT)
    idle_action = jnp.int32(env.NOOP)
    
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
    )
    with progress:
        task_id = progress.add_task("Collecting frames", total=args.steps)
        for step in range(args.steps):
            progress.update(task_id, advance=1)
            if args.action_mode == "random":
                action_key, subkey = jax.random.split(action_key)
                action = jax.random.randint(subkey, (), 0, 8)
            elif args.action_mode == "forward":
                action = forward_action
            elif args.action_mode == "idle":
                action = idle_action
            else:  # jump_forward
                action = jump_forward_action

            # Step environment (render=True is default now)
            obs, reward, terminated, truncated, info = step_jit(state, action)
            state = info["state"]

            frames.append(np.array(obs))

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}. Resetting and continuing...")
                key, reset_key = jax.random.split(key)
                obs, info = reset_jit(reset_key)
                state = info["state"]
                # We don't append reset obs here to avoid double-rendering transitions,
                # the next step() will provide the first frame of the new episode.

    print(f"Collected {len(frames)} frames.")
    save_video_file(frames, args.output, fps=args.fps)

if __name__ == "__main__":
    main()

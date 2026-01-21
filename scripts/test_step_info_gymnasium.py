import argparse
import os

from kage_bench import KAGE_Env, load_config_from_yaml
from kage_bench.wrappers import GymnasiumWrapper


def main():
    parser = argparse.ArgumentParser(description="Verify per-step info fields via Gymnasium wrapper.")
    parser.add_argument(
        "--config",
        default="src/kage_bench/configs/default_config.yaml",
        help="Path to environment config YAML.",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = load_config_from_yaml(args.config)
    jax_env = KAGE_Env(config)
    env = GymnasiumWrapper(jax_env, render_mode="rgb_array")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    for step_idx in range(args.steps):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        for key_name in ("x", "y", "camera_x", "vx", "vy", "t", "x_max"):
            if key_name not in info:
                raise KeyError(f"Missing wrapper info field: {key_name}")

        for key_name in (
            "return",
            "timestep",
            "x",
            "y",
            "vx",
            "vy",
            "vx_mean",
            "vy_mean",
            "jump_count",
            "passed_distance",
            "progress",
            "success",
            "success_once",
        ):
            if key_name not in info:
                raise KeyError(f"Missing info field: {key_name}")

        done = bool(terminated or truncated)
        print(
            "step={step} action={action} reward={reward:.4f} return={ret:.4f} "
            "timestep={t} x={x:.2f} y={y:.2f} vx={vx:.2f} vy={vy:.2f} "
            "vx_mean={vx_mean:.2f} vy_mean={vy_mean:.2f} "
            "passed_distance={dist:.2f} progress={progress:.3f} jump_count={jumps} "
            "success={success} success_once={success_once} done={done}".format(
                step=step_idx,
                action=action,
                reward=reward,
                ret=info["return"],
                t=info["timestep"],
                x=info["x"],
                y=info["y"],
                vx=info["vx"],
                vy=info["vy"],
                vx_mean=info["vx_mean"],
                vy_mean=info["vy_mean"],
                dist=info["passed_distance"],
                progress=info["progress"],
                jumps=info["jump_count"],
                success=info["success"],
                success_once=info["success_once"],
                done=done,
            )
        )

        if done:
            env.reset()


if __name__ == "__main__":
    main()

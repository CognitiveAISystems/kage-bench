import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from kage_bench import KAGE_Env, load_config_from_yaml


def _as_scalar(value):
    return jax.device_get(value).item()


def main():
    parser = argparse.ArgumentParser(description="Verify per-step info fields.")
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
    env = KAGE_Env(config)

    key = jax.random.PRNGKey(args.seed)
    _, info = env.reset(key)
    state = info["state"]

    rng = np.random.default_rng(args.seed)

    for step_idx in range(args.steps):
        action = jnp.int32(rng.integers(0, 8))
        _, reward, terminated, truncated, info = env.step(state, action, render=False)
        state = info["state"]

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

        done = bool(_as_scalar(terminated) or _as_scalar(truncated))
        print(
            "step={step} action={action} reward={reward:.4f} return={ret:.4f} "
            "timestep={t} x={x:.2f} y={y:.2f} vx={vx:.2f} vy={vy:.2f} "
            "vx_mean={vx_mean:.2f} vy_mean={vy_mean:.2f} "
            "passed_distance={dist:.2f} progress={progress:.3f} jump_count={jumps} "
            "success={success} success_once={success_once} done={done}".format(
                step=step_idx,
                action=int(_as_scalar(action)),
                reward=_as_scalar(reward),
                ret=_as_scalar(info["return"]),
                t=int(_as_scalar(info["timestep"])),
                x=_as_scalar(info["x"]),
                y=_as_scalar(info["y"]),
                vx=_as_scalar(info["vx"]),
                vy=_as_scalar(info["vy"]),
                vx_mean=_as_scalar(info["vx_mean"]),
                vy_mean=_as_scalar(info["vy_mean"]),
                dist=_as_scalar(info["passed_distance"]),
                progress=_as_scalar(info["progress"]),
                jumps=int(_as_scalar(info["jump_count"])),
                success=bool(_as_scalar(info["success"])),
                success_once=bool(_as_scalar(info["success_once"])),
                done=done,
            )
        )

        if done:
            break


if __name__ == "__main__":
    main()

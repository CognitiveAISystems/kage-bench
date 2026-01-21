# source code: https://github.com/vwxyzjn/cleanrl/blob/35896b1fefa9898b904f7e09bcbe6e168e15d2a9/cleanrl/ppo_atari_envpool_xla_jax_scan.py

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import warnings
import random
import time
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from functools import partial
import yaml
from typing import Optional, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import imageio
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter
from termcolor import colored, cprint

from kage_bench import EnvConfig, KAGE_Env, load_config_from_yaml
from kage_bench.wrappers.jax_wrappers import (
    AutoResetWrapper,
    FrameStackWrapper,
    RewardStatsWrapper,
    RewardNormalizeWrapper,
    RewardClipWrapper,
)


os.environ["ABSL_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"

# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"
warnings.filterwarnings(
    "ignore",
    message=r"os\.fork\(\) was called.*JAX is multithreaded.*",
    category=RuntimeWarning,
)

console = Console()


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "KAGE-Benchmark"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    save_checkpoint_every: int = 50
    """save a checkpoint every N iterations (0 to disable)"""
    save_video: bool = True
    """save an .mp4 rollout when a checkpoint is saved"""
    video_fps: int = 60
    """fps for saved videos"""
    video_every_k: int = 1
    """record every k-th frame to speed up video capture"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    train_config_path: str = "configs/config_1_train.yaml"
    """path to the training environment YAML config"""
    eval_config_paths: Sequence[str] = ("configs/config_1_eval.yaml",)
    """paths to the evaluation environment YAML configs"""
    run_eval_every: int = 50
    """run evaluation every N iterations (0 to disable)"""
    eval_episodes: int = 128
    """number of eval episodes to run each evaluation"""
    eval_num_envs: int = 32
    """number of parallel eval environments"""
    total_timesteps: int = 25_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128 # * 16 is best, bigger is acceptable and more stable
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01 # 0.01
    """coefficient of the entropy"""
    ent_coef_anneal: bool = False
    """anneal entropy coefficient to 0 over training"""
    normalize_reward: bool = False # * If True, can hurt penalty-sensitive tasks
    """normalize rewards using running return statistics"""
    reward_clip_min: float = -jnp.inf #-1.0
    """minimum clipped reward (applied after normalization)"""
    reward_clip_max: float = jnp.inf #1.0
    """maximum clipped reward (applied after normalization)"""
    use_frame_stack: bool = False
    """enable 4-frame stacking for observations"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array
    completed_returns_sum: jnp.array
    completed_lengths_sum: jnp.array
    completed_count: jnp.array


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    with open(f"runs/{run_name}/train_config.yaml", "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=True)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    eval_metric_values = {}

    def record_eval_metrics(label, snapshot):
        if label not in eval_metric_values:
            eval_metric_values[label] = {}
        for key_name, value in snapshot.items():
            eval_metric_values[label].setdefault(key_name, []).append(float(value))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    cprint("Initializing env and JAX functions...", "cyan")
    # env setup
    def build_env(config_path: str):
        env_config = load_config_from_yaml(config_path) if config_path else EnvConfig()
        base_env = KAGE_Env(env_config)
        env = AutoResetWrapper(base_env)
        if args.use_frame_stack:
            env = FrameStackWrapper(env, num_frames=4)
        env = RewardStatsWrapper(env)
        if args.normalize_reward:
            env = RewardNormalizeWrapper(env, gamma=args.gamma)
        env = RewardClipWrapper(env, min_reward=args.reward_clip_min, max_reward=args.reward_clip_max)
        return env, base_env, env_config

    def make_config_label(config_path: str, fallback: str) -> str:
        if config_path:
            label = os.path.basename(config_path)
        else:
            label = fallback
        return label.replace(" ", "_")

    def ensure_yaml_suffix(path: str) -> str:
        return path if path.endswith(".yaml") else f"{path}.yaml"

    train_label = make_config_label(args.train_config_path, "train_config.yaml")
    train_env, train_base_env, train_env_config = build_env(args.train_config_path)
    if args.train_config_path:
        shutil.copyfile(args.train_config_path, f"runs/{run_name}/train_env_config.yaml")
    eval_envs = []
    for idx, config_path in enumerate(args.eval_config_paths):
        eval_label = make_config_label(config_path, f"eval_config_{idx + 1}.yaml")
        eval_env, eval_base_env, eval_env_config = build_env(config_path)
        if config_path:
            shutil.copyfile(
                config_path,
                ensure_yaml_suffix(f"runs/{run_name}/eval_env_config_{eval_label}"),
            )
        eval_envs.append(
            {
                "label": eval_label,
                "env": eval_env,
                "base_env": eval_base_env,
                "config": eval_env_config,
            }
        )
    train_reset_vec = jax.jit(jax.vmap(train_env.reset))
    train_step_vec = jax.jit(jax.vmap(lambda state, action, key: train_env.step(state, action, key)))
    cprint(f"Train observation space: {train_env.observation_space}", "cyan")
    cprint(f"Train action space: {train_env.action_space}", "cyan")
    for eval_entry in eval_envs:
        cprint(f"Eval observation space ({eval_entry['label']}): {eval_entry['env'].observation_space}", "cyan")
        cprint(f"Eval action space ({eval_entry['label']}): {eval_entry['env'].action_space}", "cyan")

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        completed_returns_sum=jnp.array(0.0, dtype=jnp.float32),
        completed_lengths_sum=jnp.array(0.0, dtype=jnp.float32),
        completed_count=jnp.array(0.0, dtype=jnp.float32),
    )

    def _split_keys(keys):
        key_pairs = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        return key_pairs[:, 0], key_pairs[:, 1]

    def make_step_env_wrapped(step_vec):
        def step_env_wrapped(episode_stats, states, action, env_keys):
            step_keys, next_env_keys = _split_keys(env_keys)
            next_obs, reward, terminated, truncated, info = step_vec(states, action, step_keys)
            done = jnp.logical_or(terminated, truncated)
            terminal = done
            episode_stats = episode_stats.replace(
                episode_returns=info["episode_returns"],
                episode_lengths=info["episode_lengths"],
                returned_episode_returns=jnp.where(
                    info["_episode"], info["episode"]["r"], episode_stats.returned_episode_returns
                ),
                returned_episode_lengths=jnp.where(
                    info["_episode"], info["episode"]["l"], episode_stats.returned_episode_lengths
                ),
                completed_returns_sum=episode_stats.completed_returns_sum + jnp.sum(info["episode"]["r"]),
                completed_lengths_sum=episode_stats.completed_lengths_sum + jnp.sum(info["episode"]["l"]),
                completed_count=episode_stats.completed_count + jnp.sum(info["_episode"]),
            )
            return episode_stats, next_obs, reward, done, terminal, info["state"], next_env_keys, info["final_state"]

        return step_env_wrapped

    def make_step_vec(env):
        return jax.jit(jax.vmap(lambda state, action, key: env.step(state, action, key)))

    train_step_env_wrapped = make_step_env_wrapped(train_step_vec)
    for eval_entry in eval_envs:
        eval_reset_vec = jax.jit(jax.vmap(eval_entry["env"].reset))
        eval_step_vec = make_step_vec(eval_entry["env"])
        eval_entry["reset_vec"] = eval_reset_vec
        eval_entry["step_env_wrapped"] = make_step_env_wrapped(eval_step_vec)

    def unwrap_env_state(state):
        while hasattr(state, "env_state"):
            state = state.env_state
        return state

    def compute_env_metrics(env_state, dist_to_success):
        passed_distance = env_state.x - env_state.initial_x
        success = passed_distance > jnp.float32(dist_to_success)
        dist_to_success = jnp.float32(dist_to_success)
        progress = jnp.where(dist_to_success > 0, passed_distance / dist_to_success, 0.0)
        return {
            "x": env_state.x,
            "y": env_state.y,
            "vx_mean": env_state.vx_mean,
            "vy_mean": env_state.vy_mean,
            "passed_distance": passed_distance,
            "progress": progress,
            "jump_count": env_state.jump_count.astype(jnp.float32),
            "success": success.astype(jnp.float32),
            "success_once": env_state.success_once.astype(jnp.float32),
        }

    METRIC_KEYS = (
        "x",
        "y",
        "vx_mean",
        "vy_mean",
        "passed_distance",
        "progress",
        "jump_count",
        "success",
        "success_once",
    )

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * frac

    cprint("Initializing network parameters...", "cyan")
    network = Network()
    actor = Actor(action_dim=8)
    critic = Critic()
    obs_shape = train_env.observation_space.shape
    dummy_obs = jnp.zeros((1, obs_shape[0], obs_shape[1], obs_shape[2]), dtype=jnp.uint8)
    network_params = network.init(network_key, dummy_obs)
    agent_params = flax.core.freeze(
        {
            "network": network_params,
            "actor": actor.init(actor_key, network.apply(network_params, dummy_obs)),
            "critic": critic.init(critic_key, network.apply(network_params, dummy_obs)),
        }
    )
    agent_state = TrainState.create(
        apply_fn=None,
        params=agent_params,
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        hidden = network.apply(agent_state.params["network"], next_obs)
        logits = actor.apply(agent_state.params["actor"], hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params["critic"], hidden)
        return action, logprob, value.squeeze(1), key

    @jax.jit
    def policy_action_greedy(params: flax.core.FrozenDict, obs: jnp.ndarray) -> jnp.ndarray:
        hidden = network.apply(params["network"], obs[None, ...])
        logits = actor.apply(params["actor"], hidden)
        return jnp.argmax(logits, axis=1)[0]

    def save_rollout_video(
        agent_state: TrainState,
        iteration: int,
        env,
        base_env,
        env_config,
        label: str,
        num_envs: int,
        greedy: bool,
    ):
        if not args.save_video:
            return
        video_dir = f"runs/{run_name}/videos"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/{label}_iter_{iteration}.mp4"
        key = jax.random.PRNGKey(args.seed + iteration)
        NUMBER_OF_VIDEO_ENVS = 9
        num_video_envs = min(num_envs, NUMBER_OF_VIDEO_ENVS)
        reset_keys = jax.random.split(key, num_video_envs)
        video_keys = reset_keys
        reset_vec_base = jax.jit(jax.vmap(base_env.reset))
        step_vec_base = jax.jit(jax.vmap(base_env.step))
        obs_batch, info_batch = reset_vec_base(reset_keys[:num_video_envs])
        states = info_batch["state"]
        frames = []
        num_frames = env.observation_space.shape[-1] // obs_batch.shape[-1]
        frame_stack = jnp.repeat(obs_batch[:, None, ...], num_frames, axis=1)
        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeRemainingColumn(),
        )
        with progress:
            total_steps = int(env_config.episode_length * 3)
            task_id = progress.add_task(f"Saving video ({label})", total=total_steps)
            for step in range(total_steps):
                progress.update(task_id, advance=1)
                if step % args.video_every_k == 0:
                    obs_np = np.asarray(jax.device_get(obs_batch))
                    h, w, c = obs_np.shape[1:]
                    count = obs_np.shape[0]
                    grid_cols = int(np.ceil(np.sqrt(count)))
                    grid_rows = int(np.ceil(count / grid_cols))
                    grid = np.zeros((grid_rows * h, grid_cols * w, c), dtype=obs_np.dtype)
                    for idx in range(count):
                        r = idx // grid_cols
                        cidx = idx % grid_cols
                        grid[r * h : (r + 1) * h, cidx * w : (cidx + 1) * w] = obs_np[idx]
                    frames.append(grid)
                stacked = frame_stack.reshape(
                    obs_batch.shape[0], obs_batch.shape[1], obs_batch.shape[2], -1
                )
                hidden = network.apply(agent_state.params["network"], stacked)
                logits = actor.apply(agent_state.params["actor"], hidden)
                if greedy:
                    actions = jnp.argmax(logits, axis=1)
                else:
                    key_pairs = jax.vmap(lambda k: jax.random.split(k, 2))(video_keys)
                    action_keys = key_pairs[:, 0]
                    video_keys = key_pairs[:, 1]
                    actions = jax.vmap(jax.random.categorical)(action_keys, logits)
                # Print action for the first environment in the batch
                # print(f"Step {step}: Action for env 0: {actions[0]}") # TODO: remove after debugging
                obs_batch, _, terminated, truncated, info_batch = step_vec_base(states, actions)
                states = info_batch["state"]
                frame_stack = jnp.concatenate([frame_stack[:, 1:], obs_batch[:, None, ...]], axis=1)
                done_mask = np.asarray(jax.device_get(jnp.logical_or(terminated, truncated)))
                if np.any(done_mask):
                    done_indices = np.where(done_mask)[0]
                    for idx in done_indices:
                        key_i, subkey = jax.random.split(reset_keys[idx])
                        reset_keys = reset_keys.at[idx].set(key_i)
                        video_keys = video_keys.at[idx].set(key_i)
                        obs_i, info_i = base_env.reset(subkey)
                        obs_batch = obs_batch.at[idx].set(obs_i)
                        states = jax.tree_util.tree_map(
                            lambda s, v: s.at[idx].set(v), states, info_i["state"]
                        )
                        frame_stack = frame_stack.at[idx].set(
                            jnp.repeat(obs_i[None, ...], num_frames, axis=0)
                        )
        imageio.mimsave(
            video_path,
            frames,
            fps=args.video_fps,
            codec="libx264",
            quality=8,
        )
        cprint(f"video saved to {video_path}", "yellow")

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        hidden = network.apply(params["network"], x)
        logits = actor.apply(params["actor"], hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params["critic"], hidden).squeeze()
        return logprob, entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_terminal: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(
            agent_state.params["critic"], network.apply(agent_state.params["network"], next_obs)
        ).squeeze()

        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_terminal[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def entropy_coef_schedule(count):
        if not args.ent_coef_anneal:
            return jnp.array(args.ent_coef, dtype=jnp.float32)
        frac = 1.0 - (count / (args.num_iterations * args.update_epochs * args.num_minibatches))
        return args.ent_coef * jnp.clip(frac, 0.0, 1.0)

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, mb_values, ent_coef):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > args.clip_coef)

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        if args.clip_vloss:
            v_clipped = mb_values + jnp.clip(newvalue - mb_values, -args.clip_coef, args.clip_coef)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl), clipfrac)

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

            ent_coef = entropy_coef_schedule(agent_state.step)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl, clipfrac)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                    ent_coef,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    video_time_accum = 0.0
    eval_time_accum = 0.0
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    next_obs, reset_info = train_reset_vec(reset_keys)
    cprint(f"First reset obs shape: {next_obs.shape}", "cyan")
    states = reset_info["state"]
    _, env_keys = _split_keys(reset_keys)
    next_terminal = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    cprint("JAX warmup: first reset complete.", "cyan")

    # based on https://github.dev/google/evojax/blob/0625d875262011d8e1b6aa32566b236f44b4da66/evojax/sim_mgr.py
    def step_once(carry, step, env_step_fn, dist_to_success):
        agent_state, episode_stats, obs, terminal, key, states, env_keys, metrics_sum, metrics_count = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        (
            episode_stats,
            next_obs,
            reward,
            next_done,
            next_terminal,
            next_states,
            next_env_keys,
            final_state,
        ) = env_step_fn(
            episode_stats, states, action, env_keys
        )
        done_mask = next_terminal.astype(jnp.float32)
        base_state = unwrap_env_state(final_state)
        step_metrics = compute_env_metrics(base_state, dist_to_success)
        metrics_sum = {
            key_name: metrics_sum[key_name] + jnp.sum(step_metrics[key_name] * done_mask)
            for key_name in METRIC_KEYS
        }
        metrics_count = metrics_count + jnp.sum(done_mask)
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=terminal,
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return (
            (
                agent_state,
                episode_stats,
                next_obs,
                next_terminal,
                key,
                next_states,
                next_env_keys,
                metrics_sum,
                metrics_count,
            ),
            storage,
        )

    def rollout(
        agent_state,
        episode_stats,
        next_obs,
        next_terminal,
        key,
        states,
        env_keys,
        metrics_sum,
        metrics_count,
        step_once_fn,
        max_steps,
    ):
        (
            agent_state,
            episode_stats,
            next_obs,
            next_terminal,
            key,
            states,
            env_keys,
            metrics_sum,
            metrics_count,
        ), storage = jax.lax.scan(
            step_once_fn,
            (
                agent_state,
                episode_stats,
                next_obs,
                next_terminal,
                key,
                states,
                env_keys,
                metrics_sum,
                metrics_count,
            ),
            (),
            max_steps,
        )
        return (
            agent_state,
            episode_stats,
            next_obs,
            next_terminal,
            storage,
            key,
            states,
            env_keys,
            metrics_sum,
            metrics_count,
        )

    rollout = partial(
        rollout,
        step_once_fn=partial(
            step_once,
            env_step_fn=train_step_env_wrapped,
            dist_to_success=train_env_config.dist_to_success,
        ),
        max_steps=args.num_steps,
    )

    def run_eval(agent_state: TrainState, iteration: int, global_step: int):
        if args.run_eval_every <= 0 or args.eval_episodes <= 0 or args.eval_num_envs <= 0:
            return

        def eval_on_env(reset_vec, step_env_wrapped, num_envs, seed_offset, label: str, dist_to_success: float):
            eval_key = jax.random.PRNGKey(args.seed + seed_offset + iteration)
            eval_reset_keys = jax.random.split(eval_key, num_envs)
            eval_obs, eval_reset_info = reset_vec(eval_reset_keys)
            eval_states = eval_reset_info["state"]
            action_keys, eval_env_keys = _split_keys(eval_reset_keys)
            eval_terminal = jnp.zeros(num_envs, dtype=jax.numpy.bool_)
            eval_episode_stats = EpisodeStatistics(
                episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
                episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
                returned_episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
                returned_episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
                completed_returns_sum=jnp.array(0.0, dtype=jnp.float32),
                completed_lengths_sum=jnp.array(0.0, dtype=jnp.float32),
                completed_count=jnp.array(0.0, dtype=jnp.float32),
            )

            eval_returns = []
            eval_lengths = []
            last_metrics_sum = {key_name: 0.0 for key_name in METRIC_KEYS}
            metrics_done = 0
            progress = Progress(
                "[progress.description]{task.description}",
                TextColumn("{task.completed}/{task.total}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
            )
            with progress:
                task_id = progress.add_task(f"Evaluating ({label})", total=args.eval_episodes)
                while len(eval_returns) < args.eval_episodes:
                    hidden = network.apply(agent_state.params["network"], eval_obs)
                    logits = actor.apply(agent_state.params["actor"], hidden)
                    key_pairs = jax.vmap(lambda k: jax.random.split(k, 2))(action_keys)
                    action_keys = key_pairs[:, 0]
                    action_subkeys = key_pairs[:, 1]
                    actions = jax.vmap(jax.random.categorical)(action_subkeys, logits)
                    (
                        eval_episode_stats,
                        eval_obs,
                        _,
                        eval_done,
                        eval_terminal,
                        eval_states,
                        eval_env_keys,
                        final_state,
                    ) = (
                        step_env_wrapped(eval_episode_stats, eval_states, actions, eval_env_keys)
                    )
                    done_mask = np.asarray(jax.device_get(eval_done))
                    if np.any(done_mask):
                        base_state = unwrap_env_state(final_state)
                        step_metrics = compute_env_metrics(base_state, dist_to_success)
                        for key_name in METRIC_KEYS:
                            last_metrics_sum[key_name] += float(
                                np.sum(np.asarray(jax.device_get(step_metrics[key_name]))[done_mask])
                            )
                        metrics_done += int(np.sum(done_mask))
                    returned_returns = np.asarray(jax.device_get(eval_episode_stats.returned_episode_returns))
                    returned_lengths = np.asarray(jax.device_get(eval_episode_stats.returned_episode_lengths))
                    for idx, done in enumerate(done_mask):
                        if done:
                            eval_returns.append(float(returned_returns[idx]))
                            eval_lengths.append(float(returned_lengths[idx]))
                            progress.update(task_id, advance=1)
                            if len(eval_returns) >= args.eval_episodes:
                                break
            returns_np = np.asarray(eval_returns)
            lengths_np = np.asarray(eval_lengths)
            valid_mask = returns_np != 0
            avg_metrics = {k: v / max(1, metrics_done) for k, v in last_metrics_sum.items()}
            return float(np.mean(returns_np[valid_mask])), float(np.mean(lengths_np[valid_mask])), avg_metrics

        def log_eval_bundle(label, avg_return, avg_length, metrics):
            writer.add_scalar(f"{label}/avg_episodic_return", avg_return, global_step)
            writer.add_scalar(f"{label}/avg_episodic_length", avg_length, global_step)
            writer.add_scalar(f"main_metrics/{label}/avg_episodic_return", avg_return, global_step)
            for key_name, value in metrics.items():
                writer.add_scalar(f"{label}/avg_config_{key_name}", value, global_step)
                if key_name in ("passed_distance", "jump_count"):
                    writer.add_scalar(f"main_metrics/{label}/avg_config_{key_name}", value, global_step)

        def build_eval_snapshot(label, avg_return, avg_length, metrics):
            snapshot = {
                f"{label}/avg_episodic_return": avg_return,
                f"{label}/avg_episodic_length": avg_length,
            }
            for key_name, value in metrics.items():
                snapshot[f"{label}/avg_config_{key_name}"] = value
            return snapshot

        avg_train_return, avg_train_length, train_metrics = eval_on_env(
            train_reset_vec,
            train_step_env_wrapped,
            args.eval_num_envs,
            seed_offset=20_000,
            label=train_label,
            dist_to_success=train_env_config.dist_to_success,
        )
        log_eval_bundle(train_label, avg_train_return, avg_train_length, train_metrics)
        record_eval_metrics(
            train_label,
            build_eval_snapshot(train_label, avg_train_return, avg_train_length, train_metrics),
        )

        train_table = Table(show_header=True, header_style="bold")
        train_table.add_column("global_step", justify="right")
        train_table.add_column(f"{train_label}/return", justify="right")
        train_table.add_column(f"{train_label}/length", justify="right")
        train_table.add_row(
            str(global_step),
            f"{avg_train_return:.6f}",
            f"{avg_train_length:.6f}",
        )
        console.print(train_table, style="cyan")

        if eval_envs:
            eval_table = Table(show_header=True, header_style="bold")
            eval_table.add_column("global_step", justify="right")
            eval_table.add_column("eval_label", justify="right")
            eval_table.add_column("eval/return", justify="right")
            eval_table.add_column("eval/length", justify="right")
            eval_table.add_column("|train-eval|", justify="right")
            for idx, eval_entry in enumerate(eval_envs):
                avg_eval_return, avg_eval_length, eval_metrics = eval_on_env(
                    eval_entry["reset_vec"],
                    eval_entry["step_env_wrapped"],
                    args.eval_num_envs,
                    seed_offset=10_000 + idx * 1_000,
                    label=eval_entry["label"],
                    dist_to_success=eval_entry["config"].dist_to_success,
                )
                eval_label = eval_entry["label"]
                log_eval_bundle(eval_label, avg_eval_return, avg_eval_length, eval_metrics)

                return_diff = abs(avg_train_return - avg_eval_return)
                diff_snapshot = {
                    f"{eval_label}/train_eval_return_diff": return_diff,
                    f"{eval_label}/train_eval_success_diff": abs(
                        train_metrics["success"] - eval_metrics["success"]
                    ),
                    f"{eval_label}/train_eval_progress_diff": abs(
                        train_metrics["progress"] - eval_metrics["progress"]
                    ),
                    f"{eval_label}/train_eval_success_once_diff": abs(
                        train_metrics["success_once"] - eval_metrics["success_once"]
                    ),
                    f"{eval_label}/train_eval_passed_distance_diff": abs(
                        train_metrics["passed_distance"] - eval_metrics["passed_distance"]
                    ),
                    f"{eval_label}/train_eval_jump_count_diff": abs(
                        train_metrics["jump_count"] - eval_metrics["jump_count"]
                    ),
                }
                writer.add_scalar(
                    f"{eval_label}/train_eval_return_diff", return_diff, global_step
                )
                writer.add_scalar(
                    f"main_metrics/{eval_label}/train_eval_return_diff", return_diff, global_step
                )
                writer.add_scalar(
                    f"{eval_label}/train_eval_success_diff",
                    diff_snapshot[f"{eval_label}/train_eval_success_diff"],
                    global_step,
                )
                writer.add_scalar(
                    f"{eval_label}/train_eval_progress_diff",
                    diff_snapshot[f"{eval_label}/train_eval_progress_diff"],
                    global_step,
                )
                writer.add_scalar(
                    f"main_metrics/{eval_label}/train_eval_progress_diff",
                    diff_snapshot[f"{eval_label}/train_eval_progress_diff"],
                    global_step,
                )
                writer.add_scalar(
                    f"{eval_label}/train_eval_success_once_diff",
                    diff_snapshot[f"{eval_label}/train_eval_success_once_diff"],
                    global_step,
                )
                writer.add_scalar(
                    f"main_metrics/{eval_label}/train_eval_success_once_diff",
                    diff_snapshot[f"{eval_label}/train_eval_success_once_diff"],
                    global_step,
                )
                writer.add_scalar(
                    f"{eval_label}/train_eval_passed_distance_diff",
                    diff_snapshot[f"{eval_label}/train_eval_passed_distance_diff"],
                    global_step,
                )
                writer.add_scalar(
                    f"main_metrics/{eval_label}/train_eval_passed_distance_diff",
                    diff_snapshot[f"{eval_label}/train_eval_passed_distance_diff"],
                    global_step,
                )
                writer.add_scalar(
                    f"{eval_label}/train_eval_jump_count_diff",
                    diff_snapshot[f"{eval_label}/train_eval_jump_count_diff"],
                    global_step,
                )
                writer.add_scalar(
                    f"main_metrics/{eval_label}/train_eval_jump_count_diff",
                    diff_snapshot[f"{eval_label}/train_eval_jump_count_diff"],
                    global_step,
                )
                record_eval_metrics(
                    eval_label,
                    {
                        **build_eval_snapshot(eval_label, avg_eval_return, avg_eval_length, eval_metrics),
                        **diff_snapshot,
                    },
                )
                eval_table.add_row(
                    str(global_step),
                    eval_label,
                    f"{avg_eval_return:.6f}",
                    f"{avg_eval_length:.6f}",
                    f"{return_diff:.6f}",
                )
            console.print(eval_table, style="cyan")

        save_rollout_video(
            agent_state,
            iteration,
            train_env,
            train_base_env,
            train_env_config,
            train_label,
            args.num_envs,
            greedy=False,
        )
        for eval_entry in eval_envs:
            save_rollout_video(
                agent_state,
                iteration,
                eval_entry["env"],
                eval_entry["base_env"],
                eval_entry["config"],
                eval_entry["label"],
                args.eval_num_envs,
                greedy=False,
            )

    progress = Progress(
        "[progress.description]{task.description}",
        TextColumn("{task.completed}/{task.total}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    )
    if args.run_eval_every > 0:
        eval_start = time.time()
        run_eval(agent_state, 0, global_step)
        eval_time_accum += time.time() - eval_start
    with progress:
        task_id = progress.add_task("Training", total=args.num_iterations)
        for iteration in range(1, args.num_iterations + 1):
            progress.update(task_id, advance=1)
            iteration_time_start = time.time()
            metrics_sum = {key_name: jnp.array(0.0, dtype=jnp.float32) for key_name in METRIC_KEYS}
            metrics_count = jnp.array(0.0, dtype=jnp.float32)
            (
                agent_state,
                episode_stats,
                next_obs,
                next_terminal,
                storage,
                key,
                states,
                env_keys,
                metrics_sum,
                metrics_count,
            ) = rollout(
                agent_state,
                episode_stats,
                next_obs,
                next_terminal,
                key,
                states,
                env_keys,
                metrics_sum,
                metrics_count,
            )
            global_step += args.num_steps * args.num_envs
            storage = compute_gae(agent_state, next_obs, next_terminal, storage)
            agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, key = update_ppo(
                agent_state,
                storage,
                key,
            )
            returned_returns = np.asarray(jax.device_get(episode_stats.returned_episode_returns))
            returned_lengths = np.asarray(jax.device_get(episode_stats.returned_episode_lengths))
            valid_mask = returned_returns != 0
            avg_episodic_return = None
            avg_episodic_length = None
            if np.any(valid_mask):
                avg_episodic_return = float(np.mean(returned_returns[valid_mask]))
                avg_episodic_length = float(np.mean(returned_lengths[valid_mask]))

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar("train/avg_completed_episodic_return", avg_episodic_return, global_step)
                writer.add_scalar("train/avg_completed_episodic_length", avg_episodic_length, global_step)
            avg_step_returns = float(np.mean(np.asarray(jax.device_get(episode_stats.episode_returns))))
            # writer.add_scalar("train/avg_episodic_return", avg_step_returns, global_step)
            writer.add_scalar(
                "charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step
            )
            writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
            writer.add_scalar("losses/clipfrac", clipfrac[-1, -1].item(), global_step)
            writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)
            writer.add_scalar("charts/entropy_coef", float(entropy_coef_schedule(agent_state.step)), global_step)
            metrics_count_host = float(np.asarray(jax.device_get(metrics_count)))
            if metrics_count_host > 0:
                metrics_sum_host = jax.device_get(metrics_sum)
                for key_name in METRIC_KEYS:
                    writer.add_scalar(
                        f"train/avg_{key_name}",
                        float(np.asarray(metrics_sum_host[key_name]) / metrics_count_host),
                        global_step,
                    )
                    if key_name == "progress":
                        writer.add_scalar(
                            "main_metrics/avg_train_progress",
                            float(np.asarray(metrics_sum_host[key_name]) / metrics_count_host),
                            global_step,
                        )
                    if key_name == "success_once":
                        writer.add_scalar(
                            "main_metrics/avg_success_once",
                            float(np.asarray(metrics_sum_host[key_name]) / metrics_count_host),
                            global_step,
                        )
            returns_np = np.asarray(jax.device_get(storage.returns))
            values_np = np.asarray(jax.device_get(storage.values))
            if returns_np.var() > 0:
                explained_variance = 1.0 - np.var(returns_np - values_np) / (np.var(returns_np) + 1e-8)
                writer.add_scalar("train/explained_variance", explained_variance, global_step)
            elapsed = max(1e-6, time.time() - start_time - video_time_accum - eval_time_accum)
            sps = int(global_step / elapsed)
            table = Table(show_header=True, header_style="bold")
            table.add_column("global_step", justify="right")
            table.add_column("avg_episodic_return", justify="right")
            table.add_column("avg_episodic_length", justify="right")
            table.add_column("SPS", justify="right")
            table.add_row(
                str(global_step),
                f"{avg_episodic_return:.6f}" if avg_episodic_return is not None else "-",
                f"{avg_episodic_length:.6f}" if avg_episodic_length is not None else "-",
                str(sps),
            )
            console.print(table)
            writer.add_scalar("charts/global_step", global_step, global_step)
            writer.add_scalar("charts/SPS", int(global_step / elapsed), global_step)
            writer.add_scalar(
                "charts/SPS_update",
                int(args.num_envs * args.num_steps / (time.time() - iteration_time_start)),
                global_step,
            )
            ran_eval = False
            if args.run_eval_every > 0 and iteration % args.run_eval_every == 0:
                eval_start = time.time()
                run_eval(agent_state, iteration, global_step)
                eval_time_accum += time.time() - eval_start
                ran_eval = True
            if args.save_checkpoint_every > 0 and iteration % args.save_checkpoint_every == 0:
                ckpt_dir = f"runs/{run_name}/checkpoints"
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = f"{ckpt_dir}/iter_{iteration}.ckpt"
                with open(ckpt_path, "wb") as f:
                    f.write(
                        flax.serialization.to_bytes(
                            [
                                vars(args),
                                [
                                    agent_state.params["network"],
                                    agent_state.params["actor"],
                                    agent_state.params["critic"],
                                ],
                            ]
                    )
                )
                cprint(f"checkpoint saved to {ckpt_path}", "yellow")
                if not ran_eval:
                    video_start = time.time()
                    save_rollout_video(
                        agent_state,
                        iteration,
                        train_env,
                        train_base_env,
                        train_env_config,
                        train_label,
                        args.num_envs,
                        greedy=False,
                    )
                    video_time_accum += time.time() - video_start

    final_iteration = args.num_iterations
    final_eval_ran = False
    if args.run_eval_every > 0 and final_iteration % args.run_eval_every != 0:
        eval_start = time.time()
        run_eval(agent_state, final_iteration, global_step)
        eval_time_accum += time.time() - eval_start
        final_eval_ran = True
    if args.save_checkpoint_every > 0 and final_iteration % args.save_checkpoint_every != 0:
        ckpt_dir = f"runs/{run_name}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f"{ckpt_dir}/iter_{final_iteration}.ckpt"
        with open(ckpt_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params["network"],
                            agent_state.params["actor"],
                            agent_state.params["critic"],
                        ],
                    ]
                )
            )
        cprint(f"checkpoint saved to {ckpt_path}", "yellow")
        if not final_eval_ran:
            video_start = time.time()
            save_rollout_video(
                agent_state,
                final_iteration,
                train_env,
                train_base_env,
                train_env_config,
                train_label,
                args.num_envs,
                greedy=False,
            )
            video_time_accum += time.time() - video_start

    eval_labels = [train_label] + [entry["label"] for entry in eval_envs]
    for label in eval_labels:
        label_metrics = eval_metric_values.get(label, {})
        eval_metrics_summary = {}
        for key_name, values in label_metrics.items():
            if not values:
                eval_metrics_summary[key_name] = {
                    "min": None,
                    "mean": None,
                    "median": None,
                    "max": None,
                }
                continue
            values_np = np.asarray(values, dtype=np.float32)
            eval_metrics_summary[key_name] = {
                "min": float(np.min(values_np)),
                "mean": float(np.mean(values_np)),
                "median": float(np.median(values_np)),
                "max": float(np.max(values_np)),
            }
        summary_path = ensure_yaml_suffix(f"runs/{run_name}/eval_metrics_summary_{label}")
        with open(summary_path, "w") as f:
            yaml.safe_dump(eval_metrics_summary, f, sort_keys=True)

    writer.close()

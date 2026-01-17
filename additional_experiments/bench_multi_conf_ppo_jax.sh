#!/usr/bin/env bash
set -euo pipefail

# Runs PPO training on one train config and evaluates on multiple eval configs.
#
# Usage:
#   bash additional_experiments/bench_multi_conf_ppo_jax.sh <train_config> <seed_start> <seed_end> <eval_configs...>
#
# Args:
#   train_config: path to the train config yaml.
#   seed_start: first seed to run (inclusive).
#   seed_end: last seed to run (inclusive).
#   eval_configs: list of eval config yaml paths.
#
# Env vars:
#   WANDB_PROJECT_NAME: project name for wandb (default: kage-benchmark-multi).
#   EXP_PREFIX: experiment name prefix (default: test_multi).
#   EXP_NAME: full experiment name override (default: <EXP_PREFIX>/<train_base>).
#   TOTAL_TIMESTEPS: total timesteps (default: 10000000).
#   TRACK: set to 0 to disable wandb tracking (default: 1).
#   EXTRA_ARGS: extra args for ppo_jax_multi_conf.py (space-delimited).
#
# Examples:
  # bash additional_experiments/bench_multi_conf_ppo_jax.sh \\
  #   additional_experiments/num_of_train_bg_colors/config_2_train.yaml 1 3 \\
  #   additional_experiments/num_of_train_bg_colors/config_2_val_black.yaml \\
  #   additional_experiments/num_of_train_bg_colors/config_2_val_black_white.yaml

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <train_config> <seed_start> <seed_end> <eval_configs...>" >&2
  exit 1
fi

TRAIN_CONFIG="$1"
SEED_START="$2"
SEED_END="$3"
shift 3
EVAL_CONFIGS=("$@")

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-kage-benchmark-multi}"
EXP_PREFIX="${EXP_PREFIX:-test_multi}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
TRACK="${TRACK:-1}"

if [ ! -f "$TRAIN_CONFIG" ]; then
  echo "Missing train config: ${TRAIN_CONFIG}" >&2
  exit 1
fi

if [ "${#EVAL_CONFIGS[@]}" -eq 0 ]; then
  echo "Provide at least one eval config." >&2
  exit 1
fi

for eval_config in "${EVAL_CONFIGS[@]}"; do
  if [ ! -f "$eval_config" ]; then
    echo "Missing eval config: ${eval_config}" >&2
    exit 1
  fi
done

train_base="$(basename "$TRAIN_CONFIG")"
train_base="${train_base%.yaml}"
EXP_NAME="${EXP_NAME:-${EXP_PREFIX}/${train_base}}"


EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS}"
fi

for ((seed=SEED_START; seed<=SEED_END; seed++)); do
  uv run python rl_algorithms/jax/ppo_jax_multi_conf.py \
    --exp-name "$EXP_NAME" \
    --train-config-path "$TRAIN_CONFIG" \
    --eval-config-paths "${EVAL_CONFIGS[@]}" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --wandb-project-name "$WANDB_PROJECT_NAME" \
    --seed "${seed}" \
    --track \
    --no-save-video \
    --no-save-model \
    "${EXTRA_ARGS_ARR[@]}"
done

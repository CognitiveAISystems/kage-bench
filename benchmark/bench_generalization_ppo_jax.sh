#!/usr/bin/env bash
set -euo pipefail

# Runs PPO across a config group with multiple seeds.
#
# Config tree:
#   benchmark/kage_bench_configs/
#     ├── agent_appearance/
#     ├── background/
#     ├── background_and_agent/
#     ├── distractors/
#     ├── effects/
#     ├── filters/
#     └── layout/
#
# Usage:
#   bash benchmark/bench_generalization_ppo_jax.sh <config_group> [seed_end] [config_ids...]
#   bash benchmark/bench_generalization_ppo_jax.sh <config_group> <seed_start> <seed_end> [config_ids...]
#
# Args:
#   config_group: subdirectory under benchmark/kage_bench_configs (e.g., agent_appearance).
#   seed_end: last seed to run (defaults to 10). Seeds run as 1..seed_end.
#   seed_start: optional first seed to run (defaults to 1).
#   config_ids: optional list of config IDs to run (e.g., 1 4 7).
#               If omitted, runs all configs in the group.
#
# Env vars:
#   WANDB_PROJECT_NAME: project name for wandb (default: kage-generalization).
#   EXP_PREFIX: experiment name prefix (default: generalization/<config_group>).
#
# Examples:
#   bash benchmark/bench_generalization_ppo_jax.sh filters
#     - runs all configs in filters with seeds 1..10
#   bash benchmark/bench_generalization_ppo_jax.sh filters 3
#     - runs all configs in filters with seeds 1..3
#   bash benchmark/bench_generalization_ppo_jax.sh filters 3 3
#     - runs all configs in filters with seed 3 only
#   bash benchmark/bench_generalization_ppo_jax.sh filters 1 5 1 4 7
#     - runs configs 1, 4, 7 with seeds 1..5

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <config_group> [seed_end] [config_ids...]" >&2
  echo "       $0 <config_group> <seed_start> <seed_end> [config_ids...]" >&2
  exit 1
fi

CONFIG_GROUP="$1"
SEED_START=1
SEED_END=10
SELECTED_CONFIG_IDS=()
CONFIG_DIR="benchmark/kage_bench_configs/${CONFIG_GROUP}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-kage-benchmark}"
EXP_PREFIX="${EXP_PREFIX:-generalization_benchmark/${CONFIG_GROUP}}"

if [ "$#" -ge 3 ]; then
  SEED_START="$2"
  SEED_END="$3"
  SELECTED_CONFIG_IDS=("${@:4}")
elif [ "$#" -ge 2 ]; then
  SEED_END="$2"
  SELECTED_CONFIG_IDS=("${@:3}")
fi

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Config group not found: $CONFIG_DIR" >&2
  exit 1
fi

shopt -s nullglob
if [ "${#SELECTED_CONFIG_IDS[@]}" -gt 0 ]; then
  TRAIN_CONFIGS=()
  for config_id in "${SELECTED_CONFIG_IDS[@]}"; do
    train_config="${CONFIG_DIR}/config_${config_id}_train.yaml"
    if [ ! -f "$train_config" ]; then
      echo "Missing train config: ${train_config}" >&2
      exit 1
    fi
    TRAIN_CONFIGS+=("$train_config")
  done
else
  TRAIN_CONFIGS=("${CONFIG_DIR}"/config_*_train.yaml)
  if [ "${#TRAIN_CONFIGS[@]}" -eq 0 ]; then
    echo "No train configs found in: $CONFIG_DIR" >&2
    exit 1
  fi
fi

for train_config in "${TRAIN_CONFIGS[@]}"; do
  base_name="$(basename "$train_config")"
  config_id="${base_name#config_}"
  config_id="${config_id%_train.yaml}"
  eval_config="${CONFIG_DIR}/config_${config_id}_val.yaml"
  if [ ! -f "$eval_config" ]; then
    echo "Missing eval config for ${config_id}: ${eval_config}" >&2
    exit 1
  fi
  for ((seed=SEED_START; seed<=SEED_END; seed++)); do
    uv run rl_algorithms/jax/ppo_jax.py \
      --exp-name "${EXP_PREFIX}/config_${config_id}" \
      --train-config-path "$train_config" \
      --eval-config-path "$eval_config" \
      --track \
      --wandb-project-name "$WANDB_PROJECT_NAME" \
      --seed "${seed}" \
      --no-save-video \
      --no-save-model
  done
done

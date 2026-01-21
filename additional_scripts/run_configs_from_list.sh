#!/usr/bin/env bash
# usage: ./run_configs_from_list.sh 20 40
# usage: ./run_configs_from_list.sh 77 78 79 80 81 82 83 84 85 86
# additional_scripts/run_configs_from_list.sh

# list:
# 2 3 4 5 6 7 9 14 15 16 19 21 24 44 50 51 54 56 57 60 61 64 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94

# 2 3 4 5 6 7 9 14 15 16 19 21 24 44 50 51 54 56
# 57 60 61 64 72 73 74 75 76
# 77 78 79 80 81 82 83 84 85 86
# 87 88 89 90 91 92 93 94

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <config_id> [<config_id> ...]" >&2
  exit 1
fi

for i in "$@"; do
  uv run rl_algorithms/jax/ppo_jax.py \
    --exp-name "configs_analysis_v2/config_${i}" \
    --train-config-path "configs/config_${i}_train.yaml" \
    --eval-config-path "configs/config_${i}_val.yaml" \
    --track \
    --wandb-project-name kage-configs-search-v2
done

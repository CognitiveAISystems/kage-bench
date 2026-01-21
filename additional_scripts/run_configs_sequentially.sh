#!/usr/bin/env bash
# usage: ./run_range.sh 20 40
# bash additional_scripts/run_configs_sequentially.sh 1 87

start="${1:?start}"
end="${2:?end}"

# 1..87
for ((i=start; i<=end; i++)); do
  uv run rl_algorithms/jax/ppo_jax.py \
    --exp-name "configs_analysis/config_${i}" \
    --train-config-path "configs/config_${i}_train.yaml" \
    --eval-config-path "configs/config_${i}_val.yaml" \
    --track \
    --wandb-project-name kage-configs-search
done

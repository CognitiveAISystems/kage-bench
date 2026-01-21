#!/usr/bin/env bash

base="runs/configs_analysis"

for dir in "$base"/*/; do
  [ -d "$dir" ] || continue
  if [ ! -f "${dir}videos/eval_iter_1500.mp4" ]; then
    echo "${dir%/}"
  fi
done

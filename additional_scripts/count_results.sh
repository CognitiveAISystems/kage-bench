#!/usr/bin/env bash

for ((i=1; i<=87; i++)); do
  # match any file/dir containing "config_$i" in name
  if compgen -G "runs/configs_analysis/*config_${i}*" > /dev/null; then
    continue
  else
    echo "config_${i}"
  fi
done

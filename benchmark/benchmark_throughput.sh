#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {easy|hard}"
  exit 1
fi

case "$1" in
  easy)
    config_path="scripts/fps_bench_configs/simplest_config.yaml"
    out_dir="tmp/throughput_results_easy"
    ;;
  hard)
    config_path="scripts/fps_bench_configs/hardest_config.yaml"
    out_dir="tmp/throughput_results_hard"
    ;;
  *)
    echo "Unknown mode: $1"
    echo "Usage: $0 {easy|hard}"
    exit 1
    ;;
esac

mkdir -p "$out_dir"
results_file="$out_dir/throughput_summary.tsv"
yaml_file="$out_dir/throughput_summary.yaml"
echo -e "n_envs\tmean_fps\tstd_fps" > "$results_file"
if [[ ! -f "$yaml_file" ]]; then
  echo "results:" > "$yaml_file"
fi

for p in $(seq 0 20); do
  n_envs=$((1 << p))
  echo
  echo "=== Benchmarking $n_envs envs ($1 config) ==="

  run_fps=()
  for run_idx in 1 2 3; do
    echo "--- Run $run_idx/3 ---"
    run_log="$(mktemp)"
    uv run python3 scripts/bench_vectorized.py --config "$config_path" --n_envs "$n_envs" | tee "$run_log"
    fps="$(awk -F': ' '/FPS \(Total throughput\)/{print $2}' "$run_log" | tail -n1)"
    rm -f "$run_log"

    if [[ -z "$fps" ]]; then
      echo "Failed to parse FPS for n_envs=$n_envs (run $run_idx)." >&2
      exit 1
    fi

    run_fps+=("$fps")
  done

  stats="$(printf "%s\n" "${run_fps[@]}" | awk '{
      sum+=$1; sumsq+=$1*$1; n+=1
    } END {
      if (n > 0) {
        mean=sum/n
        var=sumsq/n - mean*mean
        if (var < 0) var=0
        std=sqrt(var)
        printf "%.2f\t%.2f", mean, std
      }
    }')"

  mean_fps="$(cut -f1 <<< "$stats")"
  std_fps="$(cut -f2 <<< "$stats")"

  echo "Mean ± Std FPS: ${mean_fps} ± ${std_fps}"
  echo -e "${n_envs}\t${mean_fps}\t${std_fps}" >> "$results_file"
  {
    echo "  - n_envs: ${n_envs}"
    echo "    run_1_fps: ${run_fps[0]}"
    echo "    run_2_fps: ${run_fps[1]}"
    echo "    run_3_fps: ${run_fps[2]}"
    echo "    mean_fps: ${mean_fps}"
    echo "    std_fps: ${std_fps}"
  } >> "$yaml_file"
done

echo
echo "Saved summary to: $results_file"

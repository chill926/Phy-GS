#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${1:-0}"
RUN_DIR="${2:-output/RoadBlock_learnable_$(date +%Y%m%d_%H%M%S)/all}"
PYTHON=/home/liupengyu/miniconda3/envs/3dgs/bin/python
if [[ -e "$RUN_DIR" ]]; then
    echo "Output already exists; choose a new directory: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u train.py \
  -s /home/liupengyu/physir-splat/data/RGBT-Scenes/RoadBlock \
  -m "$RUN_DIR" --data_branch rgbt --eval --iterations 30000 \
  --warm_up 3000 --densify_until_iter 20000 --densify_grad_threshold 0.0007 \
  --phys_probe_mode learnable_linear_emissivity \
  --phys_probe_temp_ref 300 --phys_probe_emissivity_slope 0 \
  --phys_probe_lambda_slope 0.001 \
  --phys_probe_slope_warmup_fraction 0.25 --phys_probe_slope_lr_scale 0.25 \
  --phys_probe_every 1000 --phys_decouple_steps 128 \
  --save_iterations 7000 10000 20000 30000 \
  2>&1 | tee "$RUN_DIR/train.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u thesis_scripts/render_probe_maps.py \
  -m "$RUN_DIR" --iteration 30000 --split test \
  2>&1 | tee "$RUN_DIR/export_probe_maps.log"

echo "Completed: $RUN_DIR"

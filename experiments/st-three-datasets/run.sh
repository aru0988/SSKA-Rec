#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-/opt/conda/envs/pytorch-2.1.1/bin/python}" -m sska_rec.st_runner \
  --config configs/st_three_datasets.json \
  --dataset-root "${DATASET_ROOT:-$HOME/shared/dataset}" \
  --cache-root "${CACHE_ROOT:-$HOME/shared/cache}" \
  --comparison-root "${COMPARISON_ROOT:-$HOME/experiments/tri-three-datasets-20261112/outputs/tri}" \
  --legacy-kuairand "${LEGACY_ST_ROOT:-$HOME/experiments/kuairand-st-20261031-v2}" \
  --output outputs/st --device cuda

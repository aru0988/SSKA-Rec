#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-/opt/conda/envs/pytorch-2.1.1/bin/python}" -m sska_rec.tri_runner \
  --config configs/tri_three_datasets.json \
  --dataset-root "${DATASET_ROOT:-$HOME/shared/dataset}" \
  --cache-root "${CACHE_ROOT:-$HOME/shared/cache}" \
  --reuse-map experiments/tri-three-datasets/reuse-map.json \
  --output outputs/tri --device cuda

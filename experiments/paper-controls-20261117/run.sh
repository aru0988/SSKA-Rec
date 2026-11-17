#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON_BIN:-/opt/conda/envs/pytorch-2.1.1/bin/python}" -m sska_rec.paper_controls \
  --parallel --source-root "${SOURCE_ROOT:-$HOME/experiments/tri-three-datasets-20261112/outputs/tri}" \
  --readout-root "${READOUT_ROOT:-$HOME/experiments/schemes-1-2-20261116/outputs/scheme2}" \
  --dataset-root "${DATASET_ROOT:-$HOME/shared/dataset}" --cache-root "${CACHE_ROOT:-$HOME/shared/cache}" \
  --output "${OUTPUT_ROOT:-outputs/paper-controls}" \
  --workers "${WORKERS:-4}" --threads "${THREADS_PER_WORKER:-2}" "$@"

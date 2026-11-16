#!/usr/bin/env bash
set -euo pipefail
scheme="${1:?Usage: bash experiments/schemes-20261116/run.sh 1|2 [extra options]}"
shift
case "$scheme" in 1|2) ;; *) echo 'Scheme must be 1 or 2' >&2; exit 2;; esac
cd "$(dirname "$0")/../.."
exec "${PYTHON_BIN:-/opt/conda/envs/pytorch-2.1.1/bin/python}" -m sska_rec.scheme_parallel \
  --scheme "$scheme" \
  --source-root "${SOURCE_ROOT:-$HOME/experiments/tri-three-datasets-20261112/outputs/tri}" \
  --dataset-root "${DATASET_ROOT:-$HOME/shared/dataset}" \
  --cache-root "${CACHE_ROOT:-$HOME/shared/cache}" \
  --output "${OUTPUT_ROOT:-outputs/scheme$scheme}" \
  --workers "${WORKERS:-2}" --threads-per-worker "${THREADS_PER_WORKER:-2}" "$@"

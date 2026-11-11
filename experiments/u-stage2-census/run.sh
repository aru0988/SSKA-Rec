#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" -m sska_rec.stage2_runner \
  --source-root "${SOURCE_ROOT:-$HOME/experiments/u-access-census-20261111/outputs/u-access}" \
  --dataset-root "${DATASET_ROOT:-$HOME/shared/dataset}" \
  --output outputs/u-stage2

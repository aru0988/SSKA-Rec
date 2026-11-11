#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" -m sska_rec.u_access \
  --dataset-root "${DATASET_ROOT:-$HOME/shared/dataset}" \
  --output outputs/u-access

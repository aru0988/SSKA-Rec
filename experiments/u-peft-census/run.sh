#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
exec /opt/conda/envs/pytorch-2.1.1/bin/python -m sska_rec.peft_runner \
  --source-root "$HOME/experiments/u-access-census-20261111/outputs/u-access" \
  --baseline-root "$HOME/experiments/u-stage2-census-20261111/outputs/u-stage2" \
  --dataset-root "$HOME/shared/dataset" \
  --output outputs/u-peft

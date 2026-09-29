#!/bin/bash
# Download the RealPDEBench cylinder scenario (metadata + real + numerical Arrow shards).
# Run on the LOGIN node only (compute nodes may have no outbound network).
set -euo pipefail

source /work/$USER/RealPDEBench/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /work/$USER/conda/envs/realpdebench

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1

WHAT="${1:-metadata}"          # metadata | hf_dataset
TYPE="${2:-}"                  # real | numerical  (only for hf_dataset)

ARGS=(download --dataset-root "$DATASET_ROOT" --scenario cylinder --what "$WHAT")
[ -n "$TYPE" ] && ARGS+=(--dataset-type "$TYPE")

echo "### realpdebench ${ARGS[*]}"
date
realpdebench "${ARGS[@]}"
date
du -sh "$DATASET_ROOT"

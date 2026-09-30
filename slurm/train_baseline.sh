#!/bin/bash
#SBATCH --account=MST114566
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=25a-hgpn003,25a-hgpn146
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=/work/b314513067/pi-lfm/results/slurm/%x-%j.out
#SBATCH --error=/work/b314513067/pi-lfm/results/slurm/%x-%j.err
# Official realpdebench.train, unchanged. Only dataset_root, results_path, seed (and checkpoint_path for finetune)
# differ from the official config; each run gets its own generated config (no shared/overwritten files).
#   sbatch --job-name=<n> --export=ALL,MODEL=<unet|deeponet|trainsolver>,TDT=<numerical|real>,SEED=<s>[,INIT=<ckpt>] train_baseline.sh
# Two GPUs only raise the cgroup memory ceiling (200 GB/GPU); training runs on cuda:0 (see scripts/legacy).
set -euo pipefail
module purge
source /work/$USER/RealPDEBench/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /work/$USER/conda/envs/realpdebench
source /work/$USER/pi-lfm-code/slurm/gpu_guard.sh
gpu_guard "$@"

ROOT=/work/$USER/pi-lfm/data_v2.0.1
OUT=/work/$USER/pi-lfm/results/v2.0.1/baselines
FT=""; if [ -n "${INIT:-}" ]; then FT=_ft; fi
TAG=${MODEL}_${TDT}${FT}_s${SEED}
CFG=$OUT/configs/${TAG}.yaml
mkdir -p $OUT/configs
[ -e "$CFG" ] && { echo "config exists: $CFG"; exit 1; }
sed -e "s#^dataset_root:.*#dataset_root: \"$ROOT\"#" \
    -e "s#^results_path:.*#results_path: \"$OUT/runs\"#" \
    -e "s#^seed:.*#seed: $SEED#" \
    -e "s#^exp_name: *\"\?\([a-z_]*\)\"\?.*#exp_name: \"\1_s${SEED}\"#" \
    -e "s#^num_workers:.*#num_workers: 6#" \
    /work/$USER/pi-lfm-code/configs/cylinder/${MODEL}.yaml > $CFG
EXTRA=""
if [ -n "${INIT:-}" ]; then
  # checkpoint_path may span two lines in deeponet.yaml; drop the continuation line before rewriting
  sed -i -e '/^ *\.\/results\//d' -e "s#^checkpoint_path:.*#checkpoint_path: \"$INIT\"#" $CFG
  EXTRA=--is_finetune
fi
echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID tag=$TAG ==="
diff /work/$USER/pi-lfm-code/configs/cylinder/${MODEL}.yaml $CFG || true
cd /work/$USER/RealPDEBench
srun python -m realpdebench.train --config "$CFG" --train_data_type "$TDT" --use_hf_dataset $EXTRA
echo "=== done $(date) ==="

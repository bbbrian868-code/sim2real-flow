#!/bin/bash
#SBATCH --account=MST114566
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=/work/b314513067/pi-lfm/runs/slurm/%x-%j.out
#SBATCH --error=/work/b314513067/pi-lfm/runs/slurm/%x-%j.err
#
# Table 1 / Table 2 are SINGLE-STEP results, so eval runs with N_autoregressive=1
# (Tables 3 and 4 are the 2- and 3-round autoregressive variants).
#
# sbatch --job-name=<n> --export=ALL,MODEL=<cfg>,TDT=<numerical|real>,CKPT=<path>,NAR=1 eval_cylinder.sh

set -euo pipefail
module purge

source /work/$USER/RealPDEBench/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /work/$USER/conda/envs/realpdebench

NAR="${NAR:-1}"
SRC=/work/$USER/pi-lfm/configs/cylinder/${MODEL}.yaml
CFG=/work/$USER/pi-lfm/configs/cylinder/eval_${MODEL}_nar${NAR}.yaml
sed "s#^N_autoregressive:.*#N_autoregressive: ${NAR}#" "$SRC" > "$CFG"

echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
echo "model=$MODEL tdt=$TDT N_autoregressive=$NAR"
echo "ckpt=$CKPT"

cd /work/$USER/RealPDEBench
srun python -m realpdebench.eval \
  --config "$CFG" \
  --train_data_type "$TDT" \
  --checkpoint_path "$CKPT" \
  --use_hf_dataset

echo "=== done $(date) ==="

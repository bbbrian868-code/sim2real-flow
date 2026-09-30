#!/bin/bash
#SBATCH --account=MST114566
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --exclude=25a-hgpn003,25a-hgpn146
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/work/b314513067/pi-lfm/results/slurm/%x-%j.out
#SBATCH --error=/work/b314513067/pi-lfm/results/slurm/%x-%j.err
# sbatch --gres=gpu:G --job-name=<n> slurm/fm.sh <train_fm args...>   (torchrun with G processes)
set -euo pipefail
module purge
source /work/$USER/RealPDEBench/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /work/$USER/conda/envs/realpdebench
G=$(nvidia-smi -L | wc -l)
echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID gpus=$G ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /work/$USER/pi-lfm-code
if [ "$G" -gt 1 ]; then
  srun torchrun --standalone --nproc_per_node=$G fm/train_fm.py "$@"
else
  srun python fm/train_fm.py "$@"
fi
echo "=== done $(date) ==="

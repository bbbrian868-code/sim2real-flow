#!/bin/bash
#SBATCH --account=mst114566
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/work/b314513067/pi-lfm/results/slurm/%x-%j.out
#SBATCH --error=/work/b314513067/pi-lfm/results/slurm/%x-%j.err
# Nano5 (H100) version of slurm/fm.sh (dit_xpred_width Step 5); fm.sh is kept as it was for Nano4.
# Differences: account / partition, no Nano4 node excludes, --mem, the env's python by full path (no module/conda
# activate), gpu_guard_nano5.sh. Nano5 allows at most 12 CPUs per GPU.
# sbatch --gres=gpu:G --cpus-per-task=12G --job-name=<n> slurm/fm_nano5.sh <train_fm args...>   (torchrun if G > 1)
set -euo pipefail
source /work/$USER/RealPDEBench/env.sh
P=/work/$USER/conda/envs/realpdebench/bin/python
source /work/$USER/pi-lfm-code/slurm/gpu_guard_nano5.sh
gpu_guard "$@"
G=$(nvidia-smi -L | wc -l)
echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID gpus=$G ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /work/$USER/pi-lfm-code
if [ "$G" -gt 1 ]; then
  srun $(dirname $P)/torchrun --standalone --nproc_per_node=$G fm/train_fm.py "$@"
else
  srun $P fm/train_fm.py "$@"
fi
echo "=== done $(date) ==="

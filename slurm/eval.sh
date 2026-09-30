#!/bin/bash
#SBATCH --account=MST114566
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/b314513067/pi-lfm/results/slurm/%x-%j.out
#SBATCH --error=/work/b314513067/pi-lfm/results/slurm/%x-%j.err
# Generic runner: sbatch --job-name=<n> slurm/eval.sh <python script> <args...>
set -euo pipefail
module purge
source /work/$USER/RealPDEBench/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /work/$USER/conda/envs/realpdebench
echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /work/$USER/pi-lfm-code
srun python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else \"no CUDA in srun step\")" && srun python "$@"
echo "=== done $(date) ==="

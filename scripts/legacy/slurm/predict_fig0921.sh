#!/bin/bash
#SBATCH --account=MST114566
#SBATCH --job-name=pred-fig0921
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/work/b314513067/pi-lfm/runs/slurm/%x-%j.out
#SBATCH --error=/work/b314513067/pi-lfm/runs/slurm/%x-%j.err
set -euo pipefail
module purge
source /work/$USER/RealPDEBench/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /work/$USER/conda/envs/realpdebench
cd /work/$USER/RealPDEBench
srun python scripts/make_predictions_0921.py

#!/bin/bash
#SBATCH --account=MST114566
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclude=25a-hgpn003,25a-hgpn146
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/work/b314513067/pi-lfm/results/slurm/%x-%j.out
#SBATCH --error=/work/b314513067/pi-lfm/results/slurm/%x-%j.err
# Run independent single-GPU python commands side by side on one node (one per GPU, 12 CPUs each); avoids the
# per-user job limit for screening sweeps.  sbatch --gres=gpu:G --cpus-per-task=12G --job-name=<n> slurm/pack.sh LIST
# LIST: one "python args..." line per run (at most G lines); each line's stdout/stderr goes to <LIST>.<i>.log
set -euo pipefail
module purge
source /work/$USER/RealPDEBench/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /work/$USER/conda/envs/realpdebench
source /work/$USER/pi-lfm-code/slurm/gpu_guard.sh
gpu_guard "$@"
LIST=$1
G=$(nvidia-smi -L | wc -l)
echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID gpus=$G list=$LIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /work/$USER/pi-lfm-code
i=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  [[ "$line" == \#* ]] && continue
  if [ $i -ge $G ]; then echo "more lines than GPUs; skipping: $line" >&2; continue; fi
  echo "[gpu $i] $line"
  CUDA_VISIBLE_DEVICES=$i bash -c "$line" > "$LIST.$i.log" 2>&1 &
  i=$((i + 1))
done < "$LIST"
wait
echo "=== done $(date) ==="

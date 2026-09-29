#!/bin/bash
#SBATCH --account=MST114566
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=/work/b314513067/pi-lfm/runs/slurm/%x-%j.out
#SBATCH --error=/work/b314513067/pi-lfm/runs/slurm/%x-%j.err
#
# Submit with:
#   sbatch --job-name=<name> --export=ALL,MODEL=<cfg>,TDT=<numerical|real>,FT=<0|1>,CKPT=<path> train_cylinder.sh
#
# MODEL is the config basename under configs/cylinder/: deeponet | trainsolver | unet
# TDT   = --train_data_type
# FT=1  adds --is_finetune, which makes train.py load `checkpoint_path` from the yaml,
#         so CKPT must point at the corresponding Simulated-Training checkpoint.

set -euo pipefail
module purge

source /work/$USER/RealPDEBench/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /work/$USER/conda/envs/realpdebench

CFG=/work/$USER/pi-lfm/configs/cylinder/${MODEL}.yaml

# Two GPUs are requested only to raise the cgroup memory ceiling to 400 GB (the cap is
# 200 GB per GPU). Training itself still runs on cuda:0 -- the loader's full-trajectory
# decode drives page cache toward the limit, and a validation dataloader spawning its
# workers at that point dies with SIGBUS on a shm allocation.
# NW overrides num_workers to cut the peak; it does not affect batch composition.
if [ -n "${NW:-}" ]; then
  CFG_NW=/work/$USER/pi-lfm/configs/cylinder/${MODEL}_nw${NW}.yaml
  sed "s/^num_workers:.*/num_workers: ${NW}/" "$CFG" > "$CFG_NW"
  CFG="$CFG_NW"
fi

# Finetune runs need their own config carrying the pretrained checkpoint path.
if [ "${FT:-0}" = "1" ]; then
  MODEL_NAME=$(sed -n 's/^model_name: *"\?\([a-z]*\)"\?.*/\1/p' "$CFG")
  EXP_NAME=$(sed -n 's/^exp_name: *"\?\([a-z_]*\)"\?.*/\1/p' "$CFG")
  # CKPT=AUTO -> resolve the best-val checkpoint of this model's Simulated Training run
  if [ "${CKPT:-AUTO}" = "AUTO" ]; then
    CKPT=$(python /work/$USER/RealPDEBench/scripts/pick_ckpt.py \
             --run-root /work/$USER/pi-lfm/runs \
             --model-name "$MODEL_NAME" --exp-name "$EXP_NAME" \
             --train-data-type numerical --is-finetune False)
  fi
  echo "finetuning from: $CKPT"
  CFG_FT=/work/$USER/pi-lfm/configs/cylinder/${MODEL}_finetune.yaml
  sed "s#^checkpoint_path:.*#checkpoint_path: \"${CKPT}\"#" "$CFG" \
    | sed '/^ *\.\/results\//d' > "$CFG_FT"
  CFG="$CFG_FT"
  EXTRA="--is_finetune"
else
  EXTRA=""
fi

echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
echo "config=$CFG train_data_type=$TDT extra=$EXTRA"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
df -h /dev/shm | tail -1
python -c "import torch;print('torch',torch.__version__,'cap',torch.cuda.get_device_capability(0),'tf32',torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)"

cd /work/$USER/RealPDEBench
srun python -m realpdebench.train \
  --config "$CFG" \
  --train_data_type "$TDT" \
  --use_hf_dataset \
  $EXTRA

echo "=== done $(date) ==="

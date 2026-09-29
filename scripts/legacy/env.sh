# RealPDEBench environment for NCHC Nano4 (H200, sm_90)
# usage: source /work/$USER/RealPDEBench/env.sh

export PROJECT_ID=MST114566
export PILFM_ROOT=/work/$USER/pi-lfm
export REPO_ROOT=/work/$USER/RealPDEBench

# --- keep every cache off /home (100 GB quota) ---
export HF_HOME=/work/$USER/cache/hf
export TORCH_HOME=/work/$USER/cache/torch
export PIP_CACHE_DIR=/work/$USER/cache/pip
export XDG_CACHE_HOME=/work/$USER/cache/xdg
export CONDA_PKGS_DIRS=/work/$USER/conda/pkgs
export CONDA_ENVS_DIRS=/work/$USER/conda/envs
export TRITON_CACHE_DIR=/work/$USER/cache/triton
export HF_HUB_DISABLE_XET=1

# --- data / outputs ---
export DATASET_ROOT=$PILFM_ROOT/data
export RESULTS_PATH=$PILFM_ROOT/runs

# TF32: left at torch defaults (matmul=False, cudnn=True) to match the reference runs.
# repo `set_seed()` already pins cudnn.deterministic=True, cudnn.benchmark=False.

# --- offline experiment tracking (compute nodes may have no outbound network) ---
export WANDB_MODE=offline

# --- conda ---
export CONDA_ROOT=/work/envstack/apps/miniconda3/26.1.1

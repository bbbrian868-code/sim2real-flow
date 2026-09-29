#!/bin/bash
# One-shot conda env build for RealPDEBench on Nano4 login node.
set -euo pipefail

source /work/$USER/RealPDEBench/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh

ENV_PREFIX=/work/$USER/conda/envs/realpdebench

if [ ! -d "$ENV_PREFIX" ]; then
  # --override-channels -c conda-forge avoids the anaconda.com ToS gate on the shared miniconda
  conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge python=3.10
fi
conda activate "$ENV_PREFIX"

python -V
# H200 is sm_90; cu128 wheels run fine on driver 580.65.06 (CUDA 13.0 runtime present)
pip install --no-input torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --no-input -e /work/$USER/RealPDEBench

# environment.yml lists a bogus `yaml` pip package (pyyaml is already listed) -> skipped.
pip install --no-input huggingface_hub[hf_transfer] hf_xet

echo "=== verify ==="
python - <<'EOF'
import torch, sys
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__)
print("torch.cuda  ", torch.version.cuda)
print("cudnn       ", torch.backends.cudnn.version())
print("is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      ", torch.cuda.get_device_name(0))
    print("capability  ", torch.cuda.get_device_capability(0))
print("tf32 matmul ", torch.backends.cuda.matmul.allow_tf32)
print("tf32 cudnn  ", torch.backends.cudnn.allow_tf32)
EOF

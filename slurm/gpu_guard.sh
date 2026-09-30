# Sourced by the job scripts right after conda activation.
# Some nodes intermittently hand a job step GPUs that CUDA cannot initialize ("no accelerator" / "CUDA unknown
# error"); code would then silently fall back to CPU or crash. If this step cannot see a working GPU, resubmit
# the same script with the same arguments, excluding this node (and previously bad ones), then exit cleanly.
gpu_guard() {
  local ngpu; ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
  if srun --ntasks=1 python -c "import torch,sys; n=torch.cuda.device_count(); torch.zeros(1,device='cuda'); sys.exit(0 if n==$ngpu and n>0 else 1)" 2>/dev/null; then
    return 0
  fi
  local self; self=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
  export EXCL="${EXCL:-25a-hgpn003,25a-hgpn146},$(hostname -s)"
  local new; new=$(cd /work/$USER/pi-lfm-code && sbatch --parsable --exclude="$EXCL" --gres=gpu:$ngpu \
      --job-name="$SLURM_JOB_NAME" "$self" "$@")
  echo "GPU_GUARD: no working CUDA on $(hostname -s); resubmitted as $new (exclude=$EXCL)"
  exit 0
}

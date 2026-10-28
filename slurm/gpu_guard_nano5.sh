# Nano5 version of slurm/gpu_guard.sh, sourced by fm_nano5.sh after $P is set. Same logic; the exclude list starts
# empty (gpu_guard.sh starts from Nano4 node names) and the GPU check uses $P instead of the activated env's python.
# If this step cannot see a working GPU, resubmit the same script with the same arguments, excluding this node (and
# previously bad ones), then exit cleanly.
gpu_guard() {
  local ngpu; ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
  if srun --ntasks=1 $P -c "import torch,sys; n=torch.cuda.device_count(); ok=n==$ngpu and n>0 and all(f > 0.9*t for f,t in (torch.cuda.mem_get_info(i) for i in range(n))); sys.exit(0 if ok else 1)" 2>/dev/null; then
    return 0
  fi
  local self; self=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
  export EXCL="${EXCL:+$EXCL,}$(hostname -s)"
  # The resubmission can hit the per-user submit limit (QOSMaxSubmitJobPerUserLimit); wait for a slot instead of
  # losing the run.
  local new tries=0
  until new=$(cd /work/$USER/pi-lfm-code && sbatch --parsable --exclude="$EXCL" --gres=gpu:$ngpu --cpus-per-task=$((12 * ngpu)) \
      --job-name="$SLURM_JOB_NAME" "$self" "$@" 2>&1) && [[ "$new" =~ ^[0-9]+$ ]]; do
    tries=$((tries + 1))
    if [[ "$new" != *QOSMaxSubmitJobPerUserLimit* || $tries -ge 120 ]]; then
      echo "GPU_GUARD: resubmission failed: $new" >&2; exit 1
    fi
    sleep 60
  done
  echo "GPU_GUARD: no working CUDA on $(hostname -s); resubmitted as $new (exclude=$EXCL)"
  exit 0
}

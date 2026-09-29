#!/bin/bash
# Submit a Table1/2 eval (N_autoregressive=1) for every finished training run
# that does not already have one. Safe to re-run; it skips what is already done.
set -uo pipefail
P=/work/$USER/conda/envs/realpdebench/bin/python
RUNS=/work/$USER/pi-lfm/runs
S=/work/$USER/RealPDEBench/scripts

# model_name : config basename : exp_name
ROWS="deeponet:deeponet:deeponet_cylinder unet:unet:unet_cylinder transolver:trainsolver:transolver_cylinder"

for row in $ROWS; do
  IFS=: read -r mname cfg ename <<<"$row"
  for cond in "numerical False sim" "real False real" "real True ft"; do
    IFS=' ' read -r tdt ft tag <<<"$cond"
    d="$RUNS/$mname/${ename}_${tdt}_${ft}"
    [ -d "$d" ] || continue
    ls "$d"/*/model_*.pth >/dev/null 2>&1 || continue

    marker="$RUNS/.eval_submitted_${mname}_${tag}"
    [ -f "$marker" ] && continue

    # only evaluate a run whose training job already finished
    jn="cyl-${cfg}-${tag}"; [ "$mname" = deeponet ] && [ "$tag" = sim ] && jn="cyl-canary"
    # An empty squeue is legitimate once everything has finished, so key off the exit
    # status instead: only a real slurmctld failure should abort the pass.
    q=$(squeue -u $USER -h -o "%j" 2>/dev/null); rc=$?
    if [ $rc -ne 0 ]; then echo "squeue failed (rc=$rc); skipping this pass"; exit 0; fi
    if [ -n "$q" ] && grep -qx "$jn" <<<"$q"; then continue; fi

    ck=$($P $S/pick_ckpt.py --run-root "$RUNS" --model-name "$mname" \
           --exp-name "$ename" --train-data-type "$tdt" --is-finetune "$ft" 2>/dev/null)
    [ -n "$ck" ] || continue

    jid=$(sbatch --parsable --job-name=ev-${mname}-${tag} \
      --export=ALL,MODEL=$cfg,TDT=$tdt,CKPT=$ck,NAR=1 $S/slurm/eval_cylinder.sh)
    # sbatch can come back empty when slurmctld times out; do not record a marker then,
    # otherwise the empty marker permanently blocks this eval from ever being submitted.
    if [ -z "$jid" ]; then echo "sbatch failed for ${mname}/${tag}; will retry"; continue; fi
    echo "$jid" > "$marker"
    echo "eval ${mname}/${tag} -> $jid   ckpt=$(basename $ck)"
  done
done

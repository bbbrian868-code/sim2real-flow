#!/bin/bash
# Submit the 9 cylinder baseline runs: {unet, deeponet, transolver} x {sim, real, finetune}.
#   Simulated Training : --train_data_type numerical
#   Real-world Training: --train_data_type real
#   Real-world Finetune: --train_data_type real --is_finetune (loads the Simulated ckpt)
# The finetune jobs are chained with --dependency=afterok on their Simulated counterpart
# and resolve the best-val checkpoint at runtime (CKPT=AUTO).
set -euo pipefail
S=/work/$USER/RealPDEBench/scripts/slurm

declare -A CFG=( [unet]=unet [deeponet]=deeponet [transolver]=trainsolver )

for m in unet deeponet transolver; do
  c=${CFG[$m]}

  jid_sim=$(sbatch --parsable --job-name=cyl-$m-sim \
    --export=ALL,MODEL=$c,TDT=numerical,FT=0 $S/train_cylinder.sh)
  echo "$m sim      -> $jid_sim"

  jid_real=$(sbatch --parsable --job-name=cyl-$m-real \
    --export=ALL,MODEL=$c,TDT=real,FT=0 $S/train_cylinder.sh)
  echo "$m real     -> $jid_real"

  jid_ft=$(sbatch --parsable --dependency=afterok:$jid_sim --job-name=cyl-$m-ft \
    --export=ALL,MODEL=$c,TDT=real,FT=1,CKPT=AUTO $S/train_cylinder.sh)
  echo "$m finetune -> $jid_ft (after $jid_sim)"
done

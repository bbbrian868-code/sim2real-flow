#!/bin/bash
# One-line progress per cylinder job.
R=/work/b314513067/pi-lfm/runs/slurm
printf "%-8s %-22s %-9s %-7s %s\n" JOBID NAME STATE TIME PROGRESS
for j in $(squeue -u $USER -h -o "%i" | sort); do
  read -r n st tm <<<"$(squeue -j $j -h -o '%j %T %M')"
  p=$(tr "\r" "\n" < $R/${n}-${j}.err 2>/dev/null | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:?]+" | tail -1)
  printf "%-8s %-22s %-9s %-7s %s\n" "$j" "$n" "$st" "$tm" "${p:-...}"
done
echo "--- finished ---"
sacct -u $USER -X -n --format=JobID,JobName%22,State,Elapsed -S today 2>/dev/null | grep -vE "RUNNING|PENDING" | tail -12

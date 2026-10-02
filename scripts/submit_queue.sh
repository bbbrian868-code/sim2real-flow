#!/bin/bash
# Submit sbatch command lines from a queue file, waiting while the per-user submit limit is hit.
# A line may start with "WAIT:<path> " -- it is only submitted once <path> exists (e.g. a run's done.json);
# until then it stays in the queue and later lines are still processed.
# Submitted lines are appended to <queue>.done with their job id.
#   bash scripts/submit_queue.sh QUEUE_FILE
Q=$1
cd /work/$USER/pi-lfm-code
while [ -s "$Q" ]; do
  progressed=0
  n=$(wc -l < "$Q")
  for ((i = 1; i <= n; i++)); do
    line=$(sed -n "${i}p" "$Q")
    [ -z "$line" ] && continue
    cmd=$line
    if [[ "$line" == WAIT:* ]]; then
      f=${line#WAIT:}; f=${f%% *}; cmd=${line#WAIT:* }
      [ -e "$f" ] || continue
    fi
    out=$(eval "$cmd" 2>&1)
    if [[ "$out" =~ ^[0-9]+$ ]]; then
      echo "$out $cmd" >> "$Q.done"; echo "submitted $out: $cmd"
      sed -i "${i}s/.*//" "$Q"; progressed=1
    elif [[ "$out" == *QOSMaxSubmitJobPerUserLimit* ]]; then
      break
    else
      echo "RETRY later: $cmd :: $out"
    fi
  done
  sed -i '/^$/d' "$Q"
  [ -s "$Q" ] && sleep 60
done
echo "queue empty"

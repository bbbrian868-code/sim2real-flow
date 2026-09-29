#!/bin/bash
# Submit sbatch command lines from a queue file one by one, waiting while the per-user submit limit is hit.
# Lines are consumed from the top; submitted ones are appended to <queue>.done with their job id.
#   bash scripts/submit_queue.sh QUEUE_FILE
Q=$1
cd /work/$USER/pi-lfm-code
while [ -s "$Q" ]; do
  line=$(head -n 1 "$Q")
  out=$(eval "$line" 2>&1)
  if [[ "$out" =~ ^[0-9]+$ ]]; then
    echo "$out $line" >> "$Q.done"; sed -i '1d' "$Q"; echo "submitted $out: $line"
  elif [[ "$out" == *QOSMaxSubmitJobPerUserLimit* ]]; then
    sleep 60
  else
    echo "FAILED: $line :: $out"; echo "FAILED $line :: $out" >> "$Q.done"; sed -i '1d' "$Q"
  fi
done
echo "queue empty"

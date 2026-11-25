#!/usr/bin/env bash
# Run every line of one or more experiment lists, one after another:
#
#     scripts/run.sh experiments/e1.txt [more lists ...]
#
# Each line of a list is the arguments for one `python train.py` call. Empty lines and lines
# starting with # are skipped. Re-running a list is always safe: train.py skips runs that are
# finished and continues interrupted ones from their last saved epoch. A failing run is reported
# and the list continues with the next line.

for list in "$@"; do
  while read -r line; do
    if [[ -z "$line" || "$line" == \#* ]]; then
      continue
    fi
    echo "=== $(date '+%F %T')  $line"
    # < /dev/null: train.py must not read from stdin, which here is the list file itself
    python3 train.py $line < /dev/null || echo "!!! FAILED (details in its train.log): $line"
  done < "$list"
done

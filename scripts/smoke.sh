#!/usr/bin/env bash
# Quick end-to-end check on CPU (a few minutes), before spending GPU money:
# unit tests, every method and noise type for 2 epochs on small data, and resume after an
# interruption. Uses data/cifar-10-batches-py if present, otherwise a generated fake dataset.

set -euo pipefail
cd "$(dirname "$0")/.."

# The project needs nothing from PYTHONPATH. Clearing it keeps packages from other setups
unset PYTHONPATH

OUT=results/smoke
rm -rf "$OUT"
mkdir -p "$OUT"

if python3 -c "import pytest" 2>/dev/null; then
  echo "== unit tests"
  python3 -m pytest -q tests
else
  echo "== pytest not installed: skipping unit tests (pip install pytest)"
fi

DATA=data
if [[ ! -d data/cifar-10-batches-py ]]; then
  echo "== no real CIFAR-10 in data/: using a fake dataset"
  python3 tests/fake_cifar.py "$OUT/fake-data" > /dev/null
  DATA="$OUT/fake-data"
fi

# Small and fast: CPU, small CNN, 1000 training images, 2 epochs.
SMALL="--data-dir $DATA --out-dir $OUT --device cpu --arch cnn --subset 1000 --val-size 500 --test-limit 500 --batch-size 64 --epochs 2"

echo "== one short run per method and noise type"
python3 train.py $SMALL --name ce        --noise sym    --eta 0.4 --method ce        > /dev/null
python3 train.py $SMALL --name gce       --noise pair   --eta 0.4 --method gce       > /dev/null
python3 train.py $SMALL --name smallloss --noise sym    --eta 0.4 --method smallloss > /dev/null
python3 train.py $SMALL --name coteach   --noise sym    --eta 0.4 --method coteach   > /dev/null
python3 train.py $SMALL --name random    --noise random           --method ce --augment > /dev/null
python3 train.py $SMALL --name resnet    --noise sym    --eta 0.2 --arch resnet18 --subset 200 --epochs 1 > /dev/null

echo "== interrupted run: train 1 epoch, then the same command with 2 epochs must continue"
python3 train.py $SMALL --name resume --noise pair --eta 0.4 --method coteach --epochs 1 > /dev/null
out=$(python3 train.py $SMALL --name resume --noise pair --eta 0.4 --method coteach --epochs 2)
grep -q "resuming resume from epoch 2" <<< "$out" || { echo "FAIL: did not resume"; exit 1; }
[[ $(cut -d, -f1 "$OUT/resume/epochs.csv" | tail -n +2 | tr '\n' ' ') == "0 1 2 " ]] \
  || { echo "FAIL: epochs.csv should contain epochs 0 1 2 exactly once"; exit 1; }

echo "== finished run: running it again must do nothing"
out=$(python3 train.py $SMALL --name resume --noise pair --eta 0.4 --method coteach)
grep -q "nothing to do" <<< "$out" || { echo "FAIL: finished run was trained again"; exit 1; }

echo "== same name with different settings must be refused"
if python3 train.py $SMALL --name resume --noise sym --eta 0.4 --method coteach > /dev/null 2>&1; then
  echo "FAIL: a run folder was reused for different settings"; exit 1
fi

echo "== run.sh: every line below is already finished, so nothing may be trained again"
printf '# comment line\n\n%s --name ce --noise sym --eta 0.4 --method ce\n%s --name coteach --noise sym --eta 0.4 --method coteach\n' \
  "$SMALL" "$SMALL" > "$OUT/list.txt"
out=$(scripts/run.sh "$OUT/list.txt")
[[ $(grep -c "nothing to do" <<< "$out") == 2 ]] || { echo "FAIL: run.sh"; echo "$out"; exit 1; }

echo "== Co-teaching log has all columns"
for col in train_loss sel_precision sel_precision_net2 test_acc_net2 auroc_loss memorized; do
  head -1 "$OUT/coteach/epochs.csv" | tr ',' '\n' | grep -qx "$col" || { echo "FAIL: missing column $col"; exit 1; }
done

echo
tail -n 3 "$OUT/coteach/train.log"
echo
echo "SMOKE TEST PASSED"

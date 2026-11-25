# Running the experiments

## How it works (three pieces)

1. **`train.py`** trains one run (one noise setting × one method × one seed) and writes
   everything to `results/raw/<name>/`.
2. **Experiment lists** (`experiments/*.txt`): each line holds the arguments for one
   `train.py` call.
3. **`scripts/run.sh`** goes through a list line by line and calls `train.py` for each.

**Safety against crashes and Spot preemptions** is one rule inside `train.py`. After every epoch
it saves `ckpt.pt` (the network weights, optimizer state and all metrics so far). When the same
command runs again, it loads `ckpt.pt` and continues with the next epoch. If the run is already
complete, it prints "nothing to do". So after any interruption you simply **run the same
command again**.

| List | Runs |
|---|---|
| `pilot.txt` | 2 runs: choose the learning rate and epoch budget (HYPOTHESES.md §6) |
| `e1.txt` | noise-rate sweep with cross-entropy (7) |
| `e1b.txt` | true vs. random labels, 10k-image subset (2) |
| `e3_sym.txt` | small-loss and Co-teaching, symmetric 0.4, 3 seeds (6); cross-entropy baselines come from E1 |
| `e3_pair.txt` | cross-entropy, small-loss and Co-teaching, pair-flip 0.4, 3 seeds (9) |
| `optional_a.txt`, `optional_b.txt` | exploratory runs for Kaggle's two GPUs |

The learning rate and epoch budget for all main runs are the defaults of `--lr` and `--epochs`
in `train.py`. **After the pilot,** change those two defaults, add a line to the deviations log in
`HYPOTHESES.md`, and commit before starting E1.

## Setup (once per machine)

```bash
pip install -r requirements.txt
pip install -e .          # makes the `noisylabels` package in src/ importable
scripts/smoke.sh          # a few minutes on CPU; must end with "SMOKE TEST PASSED"
```

On the VM, create the venv with `--system-site-packages` first (see `infra/README.md`).

## On the VM

Always inside `tmux` (`tmux new -s runs`; detach with `Ctrl-b d`, reattach with `tmux attach -t runs`).

```bash
# Pilot: the two runs one after the other
scripts/run.sh experiments/pilot.txt
```

Main runs: two lists at the same time on the one GPU, then switch the VM off when both are done:

```bash
(scripts/run.sh experiments/e1.txt  experiments/e3_sym.txt  &
 scripts/run.sh experiments/e1b.txt experiments/e3_pair.txt &
 wait); sudo shutdown -h now
```

`&` starts a list in the background, `wait` waits for both, and the shutdown runs once both have
finished. The two groups of lists never contain the same run, so they cannot get in each other's way.

**After a preemption or crash:** start the VM and run the same command again.

**Progress:** `tail -n 3 results/raw/*/train.log`, or watch one run with
`tail -f results/raw/<name>/train.log`.

## On Kaggle (optional runs, 2 GPUs)

```bash
pip install -e .
CUDA_VISIBLE_DEVICES=0 scripts/run.sh experiments/optional_a.txt &
CUDA_VISIBLE_DEVICES=1 scripts/run.sh experiments/optional_b.txt &
wait
```

Add `--data-dir /kaggle/input/<dataset>` to the lines (or copy the data to `./data`).

## Outputs: `results/raw/<name>/`

| File | Contents |
|---|---|
| `config.json` | all arguments, forget rate used, realized noise rate (train/val), git commit, versions, GPU |
| `labels.npz` | `y_true`, `y_noisy`, `is_corrupted` (all 50,000 training images, original order), `train_idx`, `val_idx` |
| `epochs.csv` | one row per epoch (epoch 0 = before training); columns below |
| `per_sample/epoch_XXX.npz` | for the training split, in `train_idx` order: `loss` (cross-entropy w.r.t. the noisy label), `pred`, `margin` (noisy-label logit minus the largest other logit) |
| `train.log` | one line per epoch; errors are written here too |
| `ckpt.pt` | weights + optimizer + metric history; used to resume, and the final model at the end |

`epochs.csv` columns: `epoch, forget_rate, train_loss, [sel_precision, sel_fraction,
sel_precision_net2], train_acc_noisy, fit_clean, recovered, memorized, loss_clean,
loss_corrupted, auroc_loss, val_acc_noisy, val_acc_clean, test_acc, [test_acc_net2], seconds`.
Definitions follow HYPOTHESES.md §2. Everything is measured on network 1, in eval mode, without
augmentation.

Pull results to a laptop without the large checkpoints:

```bash
rsync -avP --exclude 'ckpt.pt' "$HOST":~/noisy-labels/results/ ./results/
```

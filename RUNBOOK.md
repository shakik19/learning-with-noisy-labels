# Runbook: from a working laptop to finished experiments

Every step in order, with the exact commands. Each step says where it runs:

- **(laptop)**: your machine, in the repo folder
- **(VM)**: the GCP machine, after `gcloud compute ssh`

Only the laptop ever commits or pushes. The VM only pulls code and produces results.

---

## 0. Commit everything the VM needs (laptop)

The VM gets the code from GitHub, so anything not pushed does not exist there.

```bash
git status                      # nothing important may be untracked or modified
git add scripts/run.sh scripts/plot_pilot.py uv.lock infra/.terraform.lock.hcl .gitignore \
        RUNBOOK.md README.md infra/README.md experiments/README.md
git commit -m "add: runner, runbook and lock files"
git push
```

Then merge the branch into `main` (a pull request on GitHub, or `git switch main && git merge
feat/experiment-pipeline && git push`). The VM clones `main`.

## 1. Load the environment (laptop)

```bash
cp .env.example .env            # only once; then fill in PROJECT_ID, REGION, ZONE
set -a; source .env; set +a     # in every new terminal
gcloud config set project "$PROJECT_ID"
```

## 2. One-time GCP setup (laptop; skip what you have already done)

```bash
gcloud services enable compute.googleapis.com

# Spot L4 quota must be >= 1 in your region (Spot has its own quota: PREEMPTIBLE_NVIDIA_L4_GPUS)
gcloud compute regions describe "$REGION" --flatten=quotas \
  --format="table(quotas.metric,quotas.limit)" | grep -E "L4"

# Service account for Terraform, with only the two roles it needs, and its key
gcloud iam service-accounts create "$TF_SA_NAME"
for role in roles/compute.instanceAdmin.v1 roles/compute.networkUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$TF_SA" --role="$role" --condition=None
done
gcloud iam service-accounts keys create "$TF_CREDENTIALS_FILE" --iam-account="$TF_SA"
chmod 600 "$TF_CREDENTIALS_FILE"
```

Also set a **budget alert** in the console (Billing → Budgets & alerts), e.g. €15 with alerts at
50 / 75 / 90 %.

## 3. Create the VM (laptop)

```bash
terraform -chdir=infra init
terraform -chdir=infra apply                    # type "yes"
gcloud compute ssh "$VM_NAME" --zone "$ZONE"    # first login; creates your SSH key if needed
```

If apply fails with `ZONE_RESOURCE_POOL_EXHAUSTED`, there is no Spot L4 free in that zone right
now: change `ZONE` (and `REGION` if needed) in `.env`, re-source it, apply again.

On the VM, check the GPU. The driver installs on first boot; if this fails, wait 2–3 minutes:

```bash
nvidia-smi                                      # must list an NVIDIA L4
exit
```

Back on the laptop, create an SSH alias so `rsync` works. Re-run this after every VM restart,
because the external IP changes:

```bash
gcloud compute config-ssh
ssh "$HOST" hostname                            # should print the VM's name
```

## 4. Get the code and the Python environment (VM)

`uv sync --frozen` installs exactly the versions in `uv.lock`, including the CUDA build of torch.

```bash
gcloud compute ssh "$VM_NAME" --zone "$ZONE"        # from the laptop
git clone https://github.com/shakik19/learning-with-noisy-labels.git ~/noisy-labels
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv (official installer)
source ~/.local/bin/env                             # or open a new shell
cd ~/noisy-labels
uv sync --frozen
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.9.1+cu128 True NVIDIA L4` (the version suffix may differ slightly).

From here on, every command is prefixed with `uv run`, which runs it inside the project's
environment (including the `python3` calls inside the scripts). No activation needed.

## 5. Upload the data (laptop)

```bash
rsync -avP --exclude '*.tar.gz' data/ "$HOST:$REMOTE_DIR/data/"
```

This sends `data/cifar-10-batches-py/` and `data/CIFAR-10_human.pt`, about 180 MB.

## 6. Check everything (VM)

```bash
cd ~/noisy-labels
uv run scripts/smoke.sh                             # now with the real CIFAR-10; must end with SMOKE TEST PASSED
uv run python train.py --name gpu_check --out-dir results/smoke --epochs 2   # full-size run on the GPU
```

The last command prints a line per epoch ending in `…s`, which is **the real time per epoch**.
Write it down; you need it for the time estimate in step 9.

## 7. Run the pilot (VM)

Always start long jobs inside `tmux`, so they keep running after you disconnect
(if `tmux` is missing: `sudo apt-get install -y tmux`).

```bash
tmux new -s runs
cd ~/noisy-labels
uv run scripts/run.sh experiments/pilot.txt         # 2 runs x 100 epochs, one after the other
```

Detach with `Ctrl-b d`. You can now close the laptop. Later, reattach with `tmux attach -t runs`
or check progress with:

```bash
tail -n 2 results/raw/pilot_*/train.log
```

**Never start the same list twice at the same time.** Two copies would train the same run folder
at once.

## 8. Read the pilot and choose learning rate and epochs (laptop)

```bash
rsync -avP --exclude 'ckpt.pt' --exclude 'per_sample' "$HOST:$REMOTE_DIR/results/raw/" results/raw/
uv run python scripts/plot_pilot.py
```

It prints one line per learning rate and saves `results/pilot.png`. Choose using the rule in
HYPOTHESES.md §6:

1. **Learning rate:** the one whose training is stable and where you can see both phases:
   `recovered` above `memorized` early on, then a **crossover** (memorized overtakes recovered).
   If both qualify, take the one with the clearer peak in test accuracy, and write down why.
2. **Epochs:** memorization should have levelled off before the end. Take the `mem>=90% ep`
   column for the chosen rate, add about 20 % margin, and round up to a multiple of 10 (at most 100).

If `crossover ep` is `None` for both rates, memorization did not start within 100 epochs. Raise
`--epochs` in `pilot.txt` (e.g. to 150), commit, pull on the VM and run step 7 again: the pilot
runs continue from epoch 100 rather than starting over. Record this in the deviations log.

## 9. Freeze the choice and send it to the VM (laptop, then VM)

1. In `train.py`, change the two defaults:
   `g.add_argument("--epochs", type=int, default=…)` and `g.add_argument("--lr", type=float, default=…)`.
2. In `HYPOTHESES.md`, fill in "LR / epochs" in §1 and add a row to the deviations log (§7),
   for example: `2026-11-27 | lr 0.02, epochs 70 | pilot: crossover at epoch …, plateau at … | all main runs`.
3. Commit and push:

```bash
git commit -am "freeze: lr and epoch budget after pilot"
git push
```

Then on the VM:

```bash
cd ~/noisy-labels && git pull
grep -nE '"--(epochs|lr)"' train.py                 # shows the new defaults
```

**Time check before launching:** the main runs are about 28 full-size run-equivalents (E1: 7,
E3: 21 with each Co-teaching run counted twice) plus the two small E1b runs. Multiply 28 × epochs
× seconds per epoch (step 6) for the sequential time. Running two lists in parallel should finish
sooner, since one run alone doesn't use the whole L4.

## 10. Launch the main runs (VM)

```bash
tmux attach -t runs           # or: tmux new -s runs
cd ~/noisy-labels
(uv run scripts/run.sh experiments/e1.txt  experiments/e3_sym.txt  &
 uv run scripts/run.sh experiments/e1b.txt experiments/e3_pair.txt &
 wait); sudo shutdown -h now
```

The two groups of lists run at the same time on the one GPU, and the VM switches itself off when
both are finished. Detach with `Ctrl-b d`.

**Progress:** `tail -n 1 results/raw/*/train.log`

**A run failed** (`!!! FAILED` in tmux, `RUN FAILED` in its `train.log`): the other runs continue.
Read the error. If the fix needs a code change, make it on the laptop, push, `git pull` on the
VM, **delete that run's folder** (so it restarts cleanly with the fixed code), and run the same
command again.

## 11. After a Spot preemption (laptop, then VM)

The VM is stopped, not deleted: the disk and all results are still there.

```bash
gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --format="value(status)"  # TERMINATED = stopped
gcloud compute instances start "$VM_NAME" --zone "$ZONE"
gcloud compute config-ssh                                # new IP
gcloud compute ssh "$VM_NAME" --zone "$ZONE"
```

On the VM, repeat step 10 exactly: `tmux new -s runs`, `cd ~/noisy-labels`, the same command. Finished runs
print "nothing to do", and interrupted runs continue from their last completed epoch.

If `start` fails because no Spot capacity is free, wait and retry. Do **not** change the zone:
that creates a new VM with an empty disk. Pull results first (step 12) if you must move.

## 12. Get the results (laptop)

While running (any time) or after the VM has switched itself off (start it first, step 11):

```bash
rsync -avP --exclude 'ckpt.pt' "$HOST:$REMOTE_DIR/results/raw/" results/raw/
```

This includes `per_sample/` (about 20 MB per run), which the separability analysis (E2) needs.
`results/` is git-ignored, so results are shared via this folder (or a drive), not via git.

## 13. Optional runs on Kaggle

1. Upload `data/cifar-10-batches-py/` and `data/CIFAR-10_human.pt` as a private Kaggle dataset.
2. New notebook, accelerator **GPU T4 x2**, internet on, add the dataset, then:

```python
!git clone https://github.com/shakik19/learning-with-noisy-labels.git repo
%cd repo
!pip install --no-deps -e .
!ln -s /kaggle/input/<your-dataset>/cifar-10-batches-py data/ && ln -s /kaggle/input/<your-dataset>/CIFAR-10_human.pt data/
!CUDA_VISIBLE_DEVICES=0 scripts/run.sh experiments/optional_a.txt > a.log 2>&1 & CUDA_VISIBLE_DEVICES=1 scripts/run.sh experiments/optional_b.txt > b.log 2>&1 & wait
```

`--no-deps` keeps Kaggle's own torch; its version is recorded in each run's `config.json`. Run
the notebook with **Save Version → Save & Run All** so it continues in the background, and
download `results/kaggle/` from the notebook's output when it is done.

## 14. Tear down at the end (laptop)

```bash
rsync -avP --exclude 'ckpt.pt' "$HOST:$REMOTE_DIR/results/raw/" results/raw/   # last pull
terraform -chdir=infra destroy
gcloud compute instances list && gcloud compute disks list                    # both must be empty
gcloud iam service-accounts delete "$TF_SA" --quiet                            # also invalidates the key
rm "$TF_CREDENTIALS_FILE"
```

---

## Quick reference

| I want to… | Where | Command |
|---|---|---|
| pause the VM (keeps the disk, stops GPU billing) | laptop | `gcloud compute instances stop "$VM_NAME" --zone "$ZONE"` |
| resume it | laptop | `gcloud compute instances start "$VM_NAME" --zone "$ZONE"` then `gcloud compute config-ssh` |
| see progress | VM | `tail -n 1 results/raw/*/train.log` |
| reattach to the runs | VM | `tmux attach -t runs` |
| continue after any interruption | VM | run the same `uv run scripts/run.sh …` command again |
| pull results | laptop | `rsync -avP --exclude 'ckpt.pt' "$HOST:$REMOTE_DIR/results/raw/" results/raw/` |
| delete everything | laptop | `terraform -chdir=infra destroy` |

#!/usr/bin/env python
"""Train one run (one noise setting x one method x one seed) and log what the analysis needs.

    python train.py --name sym0.4_coteach_s0 --noise sym --eta 0.4 --method coteach --seed 0

Everything goes to <out-dir>/<name>/ (file formats: experiments/README.md).

Crash / preemption safety
-------------------------
After every epoch the network(s), optimizer(s) and the metric history are saved to ckpt.pt.
Running the same command again loads ckpt.pt and continues with the next epoch. If the
checkpoint has already reached --epochs, there is nothing left to do and the script exits.
The data order depends only on (seed, epoch), so a resumed run sees the same minibatches it would
have seen without the interruption.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from noisylabels.data import (NUM_CLASSES, DeviceImages, load_cifar10, load_cifar10n,
                              stratified_split, stratified_subset)
from noisylabels.evaluate import evaluate
from noisylabels.methods import SELECTION_METHODS, Learner, forget_rate
from noisylabels.noise import corrupt_labels

REPO = Path(__file__).resolve().parent

# Settings that may change when a run is resumed. Everything else must match the saved config,
# so one run folder always belongs to exactly one experimental setting.
MAY_CHANGE_ON_RESUME = {"epochs", "data_dir", "out_dir", "device"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--name", required=True, help="run folder name, e.g. sym0.4_coteach_s0")

    g = p.add_argument_group("label noise")
    g.add_argument("--noise", choices=["none", "sym", "pair", "random", "cifar10n"], default="none")
    g.add_argument("--eta", type=float, default=0.0, help="flip probability (sym / pair)")
    g.add_argument("--cifar10n-key", default="worse_label", help="CIFAR-10N label set")

    g = p.add_argument_group("method")
    g.add_argument("--method", choices=["ce", "gce", "smallloss", "coteach"], default="ce")
    g.add_argument("--gce-q", type=float, default=0.7)
    g.add_argument("--tau", type=float, default=None,
                   help="final forget rate (default: eta; CIFAR-10N: its real noise rate)")
    g.add_argument("--tk", type=int, default=10, help="epochs until the full forget rate is reached")

    # --lr and --epochs: frozen after the pilot (HYPOTHESES.md §6)
    g = p.add_argument_group("model and optimisation")
    g.add_argument("--arch", choices=["resnet18", "cnn"], default="resnet18")
    g.add_argument("--epochs", type=int, default=80)
    g.add_argument("--lr", type=float, default=0.02, help="constant learning rate")
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--wd", type=float, default=5e-4, help="weight decay")
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--augment", action="store_true", help="random crop + horizontal flip")

    g = p.add_argument_group("data and seed")
    g.add_argument("--seed", type=int, default=0, help="controls init, data order and noise draw")
    g.add_argument("--val-size", type=int, default=5000)
    g.add_argument("--subset", type=int, default=0, help="use N training images (0 = all)")
    g.add_argument("--test-limit", type=int, default=0, help="use N test images (0 = all)")

    g = p.add_argument_group("locations")
    g.add_argument("--data-dir", default="data")
    g.add_argument("--out-dir", default="results/raw")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args(argv)
    if args.noise in ("sym", "pair") and not 0 < args.eta < 1:
        p.error("--eta must be between 0 and 1 for sym / pair noise")
    if args.tau is None and args.noise in ("sym", "pair"):
        args.tau = args.eta
    if args.method in SELECTION_METHODS and args.tau is None and args.noise != "cifar10n":
        p.error(f"--tau is required for --method {args.method} with --noise {args.noise}")
    return args


def make_logger(path: Path):
    def log(msg: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def mixed_precision(device: torch.device):
    """bf16 on newer GPUs (L4, A100), fp16 on older ones (T4), off on the CPU."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def git_commit() -> dict:
    def git(*cmd):
        return subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


def save_checkpoint(state: dict, path: Path):
    # Write to a temporary file, then rename. Renaming is atomic, so a crash leaves either the old
    # or the new checkpoint on disk, never a half-written one.
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    run_dir = Path(args.out_dir) / args.name
    (run_dir / "per_sample").mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    try:
        run(args, run_dir, log)
    except Exception:
        log("RUN FAILED\n" + traceback.format_exc())
        raise


def run(args, run_dir: Path, log):
    device = torch.device(args.device)
    amp = mixed_precision(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- 1. data, fixed train/validation split, noisy labels ----------------------------------
    x_tr, y_tr, x_te, y_te = load_cifar10(args.data_dir)
    if args.test_limit:
        x_te, y_te = x_te[:args.test_limit], y_te[:args.test_limit]
    train_idx, val_idx = stratified_split(y_tr, args.val_size, seed=0)  # same split for every run
    if args.subset:
        train_idx = stratified_subset(train_idx, y_tr, args.subset, seed=0)

    if args.noise == "cifar10n":
        y_noisy, clean = load_cifar10n(args.data_dir, args.cifar10n_key)
        assert np.array_equal(clean, y_tr), "CIFAR-10N labels do not match the CIFAR-10 order"
        corrupted = y_noisy != y_tr
    else:  # deterministic given the seed, so a resumed run regenerates exactly the same labels
        rng = np.random.default_rng([args.seed, 2024])
        y_noisy, corrupted = corrupt_labels(y_tr, args.noise, args.eta, NUM_CLASSES, rng)
    np.savez(run_dir / "labels.npz", y_true=y_tr, y_noisy=y_noisy, is_corrupted=corrupted,
             train_idx=train_idx, val_idx=val_idx)
    noise_rate = float(corrupted[train_idx].mean())
    tau = args.tau if args.tau is not None else noise_rate

    # ---- 2. config.json: written on the first start; a resume must use the same settings -------
    cfg_file = run_dir / "config.json"
    if cfg_file.exists():
        saved = json.loads(cfg_file.read_text())["args"]
        changed = sorted(k for k, v in vars(args).items()
                         if k not in MAY_CHANGE_ON_RESUME and saved.get(k) != v)
        if changed:
            raise SystemExit(f"error: {run_dir} was started with different settings {changed}. "
                             "Use another --name.")
    else:
        cfg_file.write_text(json.dumps({
            "args": vars(args), "tau_used": tau,
            "noise_rate_train": noise_rate, "noise_rate_val": float(corrupted[val_idx].mean()),
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(y_te),
            "git": git_commit(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "mixed_precision": str(amp),
        }, indent=2))

    # ---- 3. everything on the device -----------------------------------------------------------
    train_images = DeviceImages(x_tr, device, channels_last=device.type == "cuda")
    test_images = DeviceImages(x_te, device, channels_last=device.type == "cuda")
    t = {
        "train_idx": torch.from_numpy(train_idx).to(device),
        "val_idx": torch.from_numpy(val_idx).to(device),
        "test_idx": torch.arange(len(y_te), device=device),
        "y_true": torch.from_numpy(y_tr).to(device),
        "y_noisy": torch.from_numpy(y_noisy).to(device),
        "corrupted": torch.from_numpy(corrupted).to(device),
        "y_test": torch.from_numpy(y_te).to(device),
    }
    learner = Learner(args, device, amp, tau)
    ckpt_file = run_dir / "ckpt.pt"

    def record(epoch: int, row: dict, per_sample: dict, history: list):
        """Save this epoch's per-sample arrays, the checkpoint and the CSV (in that order)."""
        np.savez(run_dir / "per_sample" / f"epoch_{epoch:03d}.npz", **per_sample)
        history.append(row)
        save_checkpoint({"epoch": epoch, "learner": learner.state_dict(), "history": history}, ckpt_file)
        write_csv(run_dir / "epochs.csv", history)
        sel = f" | sel_prec {row['sel_precision']:.3f}" if "sel_precision" in row else ""
        log(f"epoch {epoch:3d}/{args.epochs} | test {row['test_acc']:.4f} | val(noisy) "
            f"{row['val_acc_noisy']:.4f} | fit_clean {row['fit_clean']:.3f} | recovered "
            f"{row['recovered']:.3f} | memorized {row['memorized']:.3f} | auroc {row['auroc_loss']:.3f}"
            f"{sel} | {row['seconds']:.1f}s")

    # ---- 4. start fresh, or resume from the checkpoint -----------------------------------------
    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        learner.load_state_dict(ckpt["learner"])
        history = ckpt["history"]
        write_csv(run_dir / "epochs.csv", history)  # in case the crash hit between ckpt and CSV
        start = ckpt["epoch"] + 1
        if start > args.epochs:
            log(f"nothing to do: {args.name} is already trained for {ckpt['epoch']} epochs")
            return
        log(f"resuming {args.name} from epoch {start}")
    else:
        log(f"starting {args.name} | noise {args.noise} (realized rate {noise_rate:.4f}) | method "
            f"{args.method} | tau {tau} | device {device} | mixed precision {amp} | "
            f"{len(train_idx)} training images")
        t0 = time.time()
        per_sample, metrics = evaluate(learner, train_images, test_images, t, 1024)
        history = []
        record(0, {"epoch": 0, "forget_rate": 0.0, **metrics, "seconds": time.time() - t0},
               per_sample, history)
        start = 1

    # ---- 5. training loop ----------------------------------------------------------------------
    n = len(train_idx)
    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        fr = forget_rate(epoch, tau, args.tk) if args.method in SELECTION_METHODS else 0.0
        order = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        perm = t["train_idx"][torch.randperm(n, generator=order).to(device)]
        learner.train()
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            x = train_images.batch(idx, augment=args.augment)
            learner.step(x, t["y_noisy"][idx], t["corrupted"][idx], fr)
        stats = learner.pop_epoch_stats()
        if not np.isfinite(stats["train_loss"]):
            raise FloatingPointError(f"training loss became {stats['train_loss']} in epoch {epoch}")

        per_sample, metrics = evaluate(learner, train_images, test_images, t, 1024)
        record(epoch, {"epoch": epoch, "forget_rate": fr, **stats, **metrics,
                       "seconds": time.time() - t0}, per_sample, history)

    log(f"finished {args.name}")


if __name__ == "__main__":
    main()

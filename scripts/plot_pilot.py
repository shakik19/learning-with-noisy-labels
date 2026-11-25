#!/usr/bin/env python
"""Summarise the pilot runs to choose the learning rate and epoch budget (HYPOTHESES.md §6).

    python3 scripts/plot_pilot.py [results/raw]

Prints one line per pilot run and saves results/pilot.png with three panels per run:
test / noisy-validation accuracy, corrupted samples recovered vs. memorized, and clean-sample fit.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/raw")
runs = sorted(root.glob("pilot_*/epochs.csv"))
if not runs:
    sys.exit(f"no pilot runs found in {root}")

fig, axes = plt.subplots(len(runs), 3, figsize=(15, 4 * len(runs)), squeeze=False)
print(f"{'run':16} {'epochs':>6} {'best val ep':>11} {'crossover ep':>12} "
      f"{'mem>=90% ep':>11} {'final mem':>9} {'s/epoch':>7}")
for row, csv in zip(axes, runs):
    d = pd.read_csv(csv)
    name = csv.parent.name
    trained = d[d.epoch > 0]

    best_val = int(d.loc[d.val_acc_noisy.idxmax(), "epoch"])
    # memorization onset: first epoch with memorized > recovered after the reverse held
    early = d.index[d.recovered > d.memorized]
    after = d[(d.index > early[0]) & (d.memorized > d.recovered)] if len(early) else d.iloc[0:0]
    crossover = int(after.epoch.iloc[0]) if len(after) else None
    # plateau: first epoch at which memorized reaches 90% of its final value
    final_mem = d.memorized.iloc[-1]
    plateau = int(d.epoch[d.memorized >= 0.9 * final_mem].iloc[0])

    print(f"{name:16} {int(d.epoch.max()):6d} {best_val:11d} {str(crossover):>12} "
          f"{plateau:11d} {final_mem:9.3f} {trained.seconds.mean():7.1f}")

    row[0].plot(d.epoch, d.test_acc, label="test (clean)")
    row[0].plot(d.epoch, d.val_acc_noisy, label="validation (noisy)")
    row[0].axvline(best_val, ls=":", c="grey")
    row[1].plot(d.epoch, d.recovered, label="recovered (pred = true label)")
    row[1].plot(d.epoch, d.memorized, label="memorized (pred = noisy label)")
    if crossover:
        row[1].axvline(crossover, ls=":", c="grey")
    row[2].plot(d.epoch, d.fit_clean, label="clean samples fitted")
    for ax, title in zip(row, ["accuracy", "corrupted training samples", "clean training samples"]):
        ax.set_title(f"{name}: {title}")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.legend(loc="best")

out = root.parent / "pilot.png"
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"\nsaved {out}")

# CIFAR-10 results

## Figures

| ID | Figure | Files |
| --- | --- | --- |
| F1 | Clean fit, recovery and wrong-label fit over training | [PNG](F1.png) · [PDF](F1.pdf) |
| F2 | Clean test trajectories and noisy-validation early stopping | [PNG](F2.png) · [PDF](F2.pdf) |
| F3 | Peak-to-final test accuracy drop by noise rate | [PNG](F3.png) · [PDF](F3.pdf) |
| F4 | Training and test accuracy with true versus random labels | [PNG](F4.png) · [PDF](F4.pdf) |
| F5 | Loss-based detection AUROC over training | [PNG](F5.png) · [PDF](F5.pdf) |
| F6 | Clean versus corrupted per-image loss histograms | [PNG](F6.png) · [PDF](F6.pdf) |
| F7 | Clean test accuracy for the three training methods | [PNG](F7.png) · [PDF](F7.pdf) |
| F8 | Selection precision for small-loss and Co-teaching | [PNG](F8.png) · [PDF](F8.pdf) |
| X1 | Test accuracy dip around crossover | [PNG](X1.png) · [PDF](X1.pdf) |
| X2 | Matched 50% thresholds and fit gap over training | [PNG](X2.png) · [PDF](X2.pdf) |
| X3 | Noisy versus clean validation and test trajectories | [PNG](X3.png) · [PDF](X3.pdf) |
| S1 | Realized training-label transition matrices | [PNG](S1.png) · [PDF](S1.pdf) |
| S4 | Existing pilot figure | [PNG](S4.png) |

## Tables

- [T1](T1.csv) · Noise sweep and per-run diagnostics
- [T2](T2.csv) · Random-label fit thresholds
- [T3](T3.csv) · Per-seed and summary method scores
- [T4](T4.csv) · Pre-specified method decision rule
- [T5](T5.csv) · Selection precision against realized clean share
- [S2](S2.csv) · Run inventory and log events
- [S3](S3.csv) · Clean test drops strictly greater than 10 percentage points after epoch 20

## Optional Kaggle figures

- O1 requires these missing run folders: `sym0.4_ce_s0`, `sym0.4_ce_aug_s0`.
- O2 requires these missing run folders: `sym0.4_coteach_tau0.3_s0`, `sym0.4_coteach_s0`, `sym0.4_coteach_tau0.5_s0`.
- O3 requires these missing run folders: `sym0.4_ce_s0`, `sym0.4_gce_s0`, `pair0.4_gce_s0`.
- O4 requires these missing run folders: `c10n-worst_ce_s0`, `c10n-worst_smallloss_s0`, `c10n-worst_coteach_s0`.

## Reading the results

- Figures X1–X3 and O1–O4 are exploratory. S4 is the pre-existing pilot image.
- Epoch 0 appears in trajectories when a measurement exists. Calculated milestones use trained epochs.
- Early stopping (ES) uses the highest noisy validation accuracy in epochs 1–60, with ties going to the earliest epoch. Test peaks are descriptive. Test labels are clean.
- A blank milestone means the threshold was not reached or the measurement did not apply. F1 shows any distinct seed-specific milestone as a vertical line. F2 and F5 show individual seed markers even where the line is a seed mean.
- Shaded bands are minimum to maximum across seeds. SD in the tables and F3 is sample standard deviation. The method comparison rule in T4 is a rule from the analysis plan, with no significance claim.
- X1 uses the highest score up to and including crossover and the lowest score strictly after it. F6 reports a share of each subset per bin, and the final column rescales the loss axis by 1,000.
- F8 shows the mean realized clean share as a visual reference. T5 tests each seed against its own realized clean share over every post-warm-up epoch.
- T4 covers the final-accuracy part of H3.3. T5 contains the second-half selection precision for its other condition.

## Regenerate

From the repository root, using the NumPy, pandas and Matplotlib dependencies in `pyproject.toml`:

```sh
python scripts/plot_results.py --raw ../raw --output results/figures
```

Use `--kaggle PATH` if the optional runs are stored outside `results/kaggle`. No GPU or training is needed.

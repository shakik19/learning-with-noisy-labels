# Hypotheses and Analysis Plan: Learning with Noisy Labels

## 0. Purpose

This note fixes, *in advance*, what we expect to observe, how we will measure it, and what would count as support or contradiction. It separates:

- **Descriptive questions** (E1, E1b, E2): demonstrate and characterize a phenomenon reported in the literature (Zhang et al. 2017; Arpit et al. 2017). Expectations come from the literature; our job is to measure them under our own controlled setup.
- **Confirmatory questions** (E3): comparisons between methods with a pre-stated decision rule.
- **Exploratory questions**: no directional prediction is made; anything found is reported as exploratory and not as a confirmed result.

---

## 1. Frozen design

| Item | Value |
|---|---|
| Dataset | CIFAR-10 (python version), original training-set order preserved |
| Splits | 45,000 train / 5,000 validation (fixed seed, stratified), 10,000 test |
| Noise applied to | train **and** validation split (validation is noisy, as in practice) |
| Test labels | always clean; used **only** for reporting, never for selection |
| Model | ResNet-18 (CIFAR variant, 3×3 stem, no max-pool) |
| Optimizer | SGD, momentum 0.9, weight decay 5e-4, batch 128, constant LR |
| LR / epochs | **set by the pilot (see §6), then frozen** |
| Augmentation | off in all main runs (on only in the optional ablation) |
| Precision | mixed precision (bf16) |
| Seeds | 0, 1, 2 (seed controls init, data order and noise draw) |

### Noise models

- **Symmetric**, rate η: each label flipped with probability η to a class drawn uniformly from the
  *other* K−1 classes. Realized flip rate is reported for every run.
- **Pair-flip**, rate η: each label flipped with probability η to (y + 1) mod K (Han et al. 2018).

Theoretical identifiability limits (K = 10): symmetric η < (K−1)/K = 0.9; pair-flip η < 0.5.
All tested rates lie inside these limits.

---

## 2. Operational definitions

All quantities are computed per epoch from an **eval-mode pass** over the full training split
(no augmentation), using the saved per-sample loss, prediction and margin.

| Symbol | Definition |
|---|---|
| C, N | clean (label unchanged) and corrupted (label flipped) training samples |
| `fit_clean(t)` | fraction of C predicted as their (correct) label |
| `recovered(t)` | fraction of N predicted as their **true** label |
| `memorized(t)` | fraction of N predicted as their **noisy** label |
| `test_acc(t)` | accuracy on the clean test set |
| `val_acc(t)` | accuracy on the **noisy** validation set |
| `t_cross` | first epoch with `memorized(t) > recovered(t)` after an epoch where the reverse held (**memorization onset**) |
| `t_clean90` | first epoch with `fit_clean(t) ≥ 0.90` |
| `t_mem50` | first epoch with `memorized(t) ≥ 0.50` |
| `peak_drop` | `max_t test_acc(t) − test_acc(final)` |
| `AUROC_loss(t)` | AUROC of per-sample loss as a detector of N vs. C (higher loss ⇒ "corrupted") |
| `precision_sel(t)` | fraction of samples selected for the update (small-loss / Co-teaching) that are in C |
| `ES` (early stopping) | the epoch maximizing `val_acc(t)`; its `test_acc` is the "early-stopped" score |

Thresholds (0.90, 0.50) are arbitrary but fixed here before seeing results.

---

## 3. Descriptive hypotheses

### E1: Noise sweep (symmetric η ∈ {0, 0.2, 0.4, 0.6, 0.8}; extra seeds at 0.4)

**H1.1: Early learning precedes memorization.**
For η ∈ {0.2, 0.4, 0.6}: early in training `recovered(t) > memorized(t)`, and later a crossover
`t_cross` occurs.
- *Supported if* both phases are observed within the epoch budget.
- *Contradicted if* `memorized(t) ≥ recovered(t)` from the first logged epoch (no early-learning phase).
- *Inconclusive if* no crossover occurs within the budget → reported as "memorization not reached in
  budget", not as a refutation.

**H1.2: Clean samples are fitted before corrupted ones.**
For η ∈ {0.2, 0.4, 0.6}: `t_clean90 < t_mem50`.
- *Contradicted if* `t_mem50 ≤ t_clean90`.

**H1.3: Memorization harms test performance.**
For every η > 0: `peak_drop > 0`, and `peak_drop(η) > peak_drop(0)`.
- *Contradicted if* test accuracy is non-decreasing to the end of training for η > 0.

**H1.4: Damage grows with noise rate.**
`peak_drop` increases monotonically over η ∈ {0.2, 0.4, 0.6}.
- *Contradicted if* the ordering is violated by more than the seed spread measured at η = 0.4.

**E1-X (exploratory, no prediction).** Does `t_cross` move earlier or later as η increases?
Arguments exist both ways (weaker clean signal vs. more samples to memorize); we only report it.

**E1-X2 (exploratory).** Behaviour at η = 0.8, which is learnable in principle (< 0.9) but has a very small clean margin. We expect a weak or absent early-learning phase, but make no formal prediction.

### E1b: Capacity to memorize random labels (10k subset)

**H1b.1.** ResNet-18 reaches ≥ 99 % training accuracy on fully random labels within the budget, while test accuracy stays near chance (10 %).

**H1b.2.** Fitting random labels takes more epochs than fitting true labels
(epochs to reach 99 % training accuracy).
- *Contradicted if* random labels are fitted as fast as or faster than true labels.
- *Inconclusive if* 99 % is not reached → reported as a budget limit.

### E2: Clean vs. corrupted separability (analysis of E1 and E3-CE logs)

**H2.1: Loss separates clean from corrupted samples during early learning.**
For η ∈ {0.2, 0.4, 0.6}: `max_t AUROC_loss(t)` is well above 0.5.

**H2.2: Separability is lost as memorization proceeds.**
`AUROC_loss` at the final epoch is lower than its maximum, and the maximum occurs before `t_cross`.
- *Contradicted if* AUROC is maximal at or after `t_cross`, or does not decrease.

**E2-X (exploratory).** Is the epoch of maximal `AUROC_loss` close to the epoch of maximal
`test_acc`? Is separability lower under pair-flip than under symmetric noise at the same η?

---

## 4. Confirmatory hypotheses: E3 (methods)

Conditions: {symmetric 0.4, pair-flip 0.4} × {CE, small-loss (single net), Co-teaching} × 3 seeds.
Co-teaching uses τ = η (known noise rate), T_k = 10. Small-loss is identical to Co-teaching except that each network trains on **its own** small-loss selection (no cross-update).

**Decision rule.** With 3 seeds we do not claim statistical significance. Method A is said to *outperform* B if the difference in mean final `test_acc` exceeds **2 × the larger of the two across-seed standard deviations** and A ≥ B holds for every seed. Otherwise: "no clear difference".

**H3.1: Co-teaching beats standard training at the end of training.**
Co-teaching outperforms CE on final-epoch `test_acc`, under both noise types.

**H3.2: Co-teaching vs. early stopping (two-sided, no directional prediction).**
Compare Co-teaching (final epoch) with CE at the ES epoch. We consider this the most informative comparison and state no expected direction in advance.

**H3.3: The cross-update matters.**
Co-teaching outperforms small-loss single-network on final `test_acc`, and its `precision_sel(t)` is higher in the second half of training.
- *Contradicted if* the two are indistinguishable under the decision rule reported as "no benefit of cross-update observed at this scale".

**H3.4: Selection works (mechanism check).**
For both selection methods, `precision_sel(t) > 1 − η` (the base rate of clean labels) for all epochs after the warm-up (t > T_k).
- *If contradicted,* selection is not filtering noise; E3 results are then interpreted as failure of the small-loss assumption in our setup.

**E3-X (exploratory).** Pair-flip vs. symmetric: are accuracies and `precision_sel` lower under pair-flip at the same η?

---

## 5. Analysis rules (fixed in advance)

1. **No test-set selection.** Any "best epoch" is chosen on the noisy validation set.
Test-set curves are shown for illustration; test-set peaks are never reported as a method's score.
2. All numbers and figures are generated by scripts from the saved logs; nothing is computed by hand.
3. Report mean ± std over seeds where seeds exist; state explicitly where only one seed was run.
4. Every run reports its **realized** noise rate.
5. Failed or crashed runs are rerun with the same seed and noted in the deviations log; runs are never dropped because of their outcome.
6. Optional experiments (augmentation ablation, CIFAR-10N, τ mis-specification, GCE) are **exploratory** and labelled as such.

---

## 6. Pilot protocol (the only allowed tuning)

Before E1, one pilot run at symmetric η = 0.4, seed 0. It may change only:

- the constant **learning rate** (candidates 0.01, 0.02), and
- the **epoch budget**,

with the goal that `t_cross` is reached and `memorized(t)` approaches a plateau before the final epoch.
The chosen values are written into §1 and the deviations log before E1 starts.
The pilot run itself is not used as an E1 result unless its configuration exactly matches the frozen one.

---

## 7. Deviations log

| Date | Change | Reason | Affects |
|---|---|---|---|
| | | | |

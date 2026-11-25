# Learning with Noisy Labels

AAI2 practical, topic A2. A controlled study of how a deep network trained on partly wrong labels
first learns the clean structure and later **memorizes the wrong labels**, whether clean and
corrupted training examples can be told apart by their loss, and whether **Co-teaching**
(Han et al., 2018) limits the damage.

We corrupt CIFAR-10 training labels on purpose (symmetric and pair-flip noise at known rates).
Because we know exactly which labels are wrong, we can follow every training example through
training. All hypotheses and decision rules were fixed before the experiments; see
[HYPOTHESES.md](HYPOTHESES.md).

## Experiments

| | Question | Runs |
|---|---|---|
| **E1** | When does memorization start, and how does it depend on the noise rate? | cross-entropy, symmetric noise 0 / 20 / 40 / 60 / 80 % |
| **E1b** | Can the network fit completely random labels? (Zhang et al., 2017) | true vs. random labels, 10k images |
| **E2** | Do clean and corrupted examples have different losses, and when? | analysis of the E1 logs, no extra training |
| **E3** | Does Co-teaching help, and is the exchange between its two networks what helps? | cross-entropy vs. small-loss vs. Co-teaching, symmetric and pair-flip 40 %, 3 seeds |
| optional | augmentation, GCE loss, wrong forget rate, real human label noise (CIFAR-10N) | Kaggle |

Setup: ResNet-18, SGD with a constant learning rate, no augmentation in the main runs, a fixed
45k / 5k train / validation split with the **same noise on the validation set** (as in practice),
and the clean test set used only for reporting.

## Repository layout

```
HYPOTHESES.md        hypotheses, definitions and analysis rules, frozen before the experiments
RUNBOOK.md           step-by-step guide: cloud VM -> pilot -> main runs -> results
train.py             trains one run and logs everything (resumes automatically after a crash)
src/noisylabels/     data.py  noise.py  model.py  methods.py  evaluate.py
experiments/         one text file per experiment, one run per line (+ README: output formats)
scripts/run.sh       runs every line of experiment lists
scripts/smoke.sh     few-minute end-to-end test on CPU
scripts/plot_pilot.py  summary used to choose learning rate and epochs
tests/               unit tests (noise matrices, data split, fake CIFAR-10 for tests)
data/                datasets (not in git; see data/README.md)
infra/               Terraform for the GCP training VM (see infra/README.md)
results/             run outputs (not in git)
```

## Quick start (local, CPU)

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # creates .venv with the locked dependencies
uv run scripts/smoke.sh          # a few minutes; must end with "SMOKE TEST PASSED"
```

The smoke test does not need the real dataset (it generates a small fake CIFAR-10 if `data/` is
empty). For the real experiments, download the data as described in [data/README.md](data/README.md).

## Reproducing the experiments

Follow [RUNBOOK.md](RUNBOOK.md). In short: create the GPU VM with Terraform, clone the repo,
`uv sync --frozen`, upload the data, run the pilot, freeze the learning rate and epoch budget, then:

```bash
(uv run scripts/run.sh experiments/e1.txt  experiments/e3_sym.txt  &
 uv run scripts/run.sh experiments/e1b.txt experiments/e3_pair.txt &
 wait)
```

A single run can also be started directly, e.g.

```bash
uv run python train.py --name sym0.4_coteach_s0 --noise sym --eta 0.4 --method coteach --seed 0
```

and `uv run python train.py --help` lists all options. Re-running an interrupted command continues from
the last completed epoch.

## Outputs

Each run writes to `results/raw/<name>/`: its settings and the **git commit** of the code that
produced it (`config.json`), the clean and noisy labels (`labels.npz`), one row of metrics per epoch
(`epochs.csv`), and the loss, prediction and margin of every training example at every epoch
(`per_sample/`). The formats are listed in [experiments/README.md](experiments/README.md).

## References

- Zhang, Bengio, Hardt, Recht, Vinyals (2017). *Understanding deep learning requires rethinking
  generalization.* ICLR. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
- Arpit et al. (2017). *A Closer Look at Memorization in Deep Networks.* ICML.
  [arXiv:1706.05394](https://arxiv.org/abs/1706.05394)
- Han et al. (2018). *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy
  Labels.* NeurIPS. [arXiv:1804.06872](https://arxiv.org/abs/1804.06872)
- Wei et al. (2022). *Learning with Noisy Labels Revisited: A Study Using Real-World Human
  Annotations* (CIFAR-10N). ICLR. [arXiv:2110.12088](https://arxiv.org/abs/2110.12088)

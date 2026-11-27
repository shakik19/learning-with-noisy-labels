#!/usr/bin/env python3
"""Generate FIGURES.md outputs from saved logs without training or loading checkpoints."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RATES = (0, 0.2, 0.4, 0.6, 0.8)
NOISES = {"sym": "Symmetric", "pair": "Pair-flip"}
METHODS = {"ce": "Cross-entropy", "smallloss": "Small-loss", "coteach": "Co-teaching"}
RATE_COLORS = ("#586279", "#006A72", "#3058A9", "#B24221", "#7B4087")
METHOD_COLORS = dict(zip(METHODS, ("#596378", "#B24221", "#006A72")))
CLEAN, TRUE, WRONG = "#596378", "#006A72", "#B24221"
INK = "#172337"
MUTED = "#516078"
DIAGNOSTICS = (
    ("t_cross", "Crossover", "#88479E", ":"),
    ("t_clean90", "90% clean fit", "#596378", "-."),
    ("t_mem50", "50% wrong-label fit", "#A66B20", "--"),
)
CLASSES = ("Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#A1ADC0",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": "#E0E6EF",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


class Run:
    def __init__(self, path):
        self.path = path
        self.name = path.name
        self.config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        self.args = self.config["args"]
        self.epochs = pd.read_csv(path / "epochs.csv", index_col="epoch").sort_index()
        expected = np.arange(self.args["epochs"] + 1)
        if not np.array_equal(self.epochs.index, expected):
            raise ValueError(f"{path}: expected one row for each epoch 0–{expected[-1]}")
        self.trained = self.epochs.loc[1:]
        self.final = self.trained.iloc[-1]
        self.best_test = peak(self.trained.test_acc)
        self.best_auc = peak(self.trained.auroc_loss)
        self.es = peak(self.trained.loc[:60, "val_acc_noisy"])
        if self.es is None:
            raise ValueError(f"{path}: missing noisy validation scores")
        recovery = self.trained.recovered
        wrong = self.trained.memorized
        self.had_recovery_lead = bool((recovery > wrong).any())
        self.t_cross = first((wrong > recovery) & (recovery > wrong).cummax().shift(fill_value=False))
        self.t_clean90 = first(self.trained.fit_clean >= 0.90)
        self.t_clean50 = first(self.trained.fit_clean >= 0.50)
        self.t_mem50 = first(wrong >= 0.50)

    @property
    def drop(self):
        return self.trained.at[self.best_test, "test_acc"] - self.final.test_acc

    @property
    def base_clean(self):
        return 1 - self.config["noise_rate_train"]

    def training_labels(self):
        with np.load(self.path / "labels.npz", allow_pickle=False) as data:
            train = data["train_idx"]
            return tuple(data[key][train] for key in ("y_true", "y_noisy", "is_corrupted"))


def first(condition):
    hits = condition.index[condition]
    return int(hits[0]) if len(hits) else None


def peak(series):
    valid = series.dropna()
    return int(valid.idxmax()) if len(valid) else None


def load_runs(root):
    return {path.parent.name: Run(path.parent) for path in sorted(root.glob("*/epochs.csv"))}


def seed_group(runs, prefix):
    return [runs[f"{prefix}_s{seed}"] for seed in range(3)]


def new_figure(title, subtitle, shape=(1, 1), size=(10.2, 5.5), **kwargs):
    fig, axes = plt.subplots(*shape, figsize=size, squeeze=False, **kwargs)
    fig.subplots_adjust(left=.10, right=.965, top=.79, bottom=.22, hspace=.60, wspace=.27)
    fig.text(.10, .955, title, size=20, weight="bold", va="top")
    fig.text(.10, .894, subtitle, size=10.6, color=MUTED, va="top")
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y")
    return fig, axes


def legend(fig, handles, columns):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .023), ncol=columns,
               fontsize=10.5, columnspacing=1.25)


def swatch(label, color, style="-", marker=None):
    return Line2D([], [], color=color, ls=style, label=label, lw=2.2, marker=marker)


def epoch_axis(ax, end, label=None, ylim=None, percentage=True):
    ax.set(xlim=(0, end), xlabel="Epoch", ylabel=label, ylim=ylim)
    ax.set_xticks(np.arange(0, end + 1, 10 if end <= 60 else 25))
    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def series_plot(ax, runs, key, color, *, band=False, style="-"):
    index = runs[0].epochs.index
    if any(not index.equals(run.epochs.index) for run in runs):
        raise ValueError(f"Mismatched epoch budgets for {runs[0].name}")
    values = np.stack([run.epochs[key].to_numpy() for run in runs])
    present = np.isfinite(values).all(axis=0)
    epochs, values = index[present], values[:, present]
    if band and len(runs) > 1:
        ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=color, alpha=.15, lw=0)
    ax.plot(epochs, values.mean(axis=0), color=color, ls=style, lw=2.15)


def save(fig, out, name):
    for fmt in ("png", "pdf"):
        fig.savefig(out / f"{name}.{fmt}", dpi=250)
    plt.close(fig)


def main_figures(out, sweep, pair, subset, methods):
    metrics = (
        ("fit_clean", "Clean labels fitted", CLEAN),
        ("recovered", "Original label on corrupted images", TRUE),
        ("memorized", "Wrong label on corrupted images", WRONG),
    )
    fig, axes = new_figure(
        "When does the network start fitting wrong labels?",
        "Symmetric label flips. Shading at 40% spans the seed range. Vertical lines show each seed’s milestones.",
        (2, 2), (12, 8.4), sharex=True, sharey=True,
    )
    fig.subplots_adjust(top=.81, bottom=.19, hspace=.44, wspace=.20)
    for ax, rate in zip(axes.flat, RATES[1:]):
        group = sweep[rate]
        ax.set_title(f"{rate:.0%} noise  ·  {seed_label(group)}", loc="left")
        for field, _, color, style in DIAGNOSTICS:
            times = {getattr(run, field) for run in group} - {None}
            for epoch in sorted(times):
                ax.axvline(epoch, color=color, ls=style, lw=1.05, alpha=.70, zorder=1)
        for key, _, color in reversed(metrics):
            series_plot(ax, group, key, color, band=True, style="--" if key == "fit_clean" else "-")
        epoch_axis(ax, 60, "Share of subset" if ax in axes[:, 0] else None, (0, 1.03))
    for ax in axes[0]:
        ax.set_xlabel("")
    legend(fig, [swatch(label, color, "--" if key == "fit_clean" else "-") for key, label, color in metrics]
           + [swatch(label, color, style) for _, label, color, style in DIAGNOSTICS], 3)
    save(fig, out, "F1")

    fig, ((ax,),) = new_figure(
        "Clean test accuracy across noise rates",
        "Symmetric label flips. Lines show seed means. Each dot shows a run’s score at its noisy-validation choice.",
    )
    for rate, color in zip(RATES, RATE_COLORS):
        group = sweep[rate]
        series_plot(ax, group, "test_acc", color)
        ax.scatter([run.es for run in group], [run.epochs.at[run.es, "test_acc"] for run in group],
                   s=48, facecolors="white", edgecolors=color, linewidths=1.6, zorder=4, clip_on=False)
    epoch_axis(ax, 60, "Clean test accuracy", (0, .9))
    legend(fig, [swatch(f"{rate:.0%} ({seed_label(sweep[rate])})", color)
                 for rate, color in zip(RATES, RATE_COLORS)], 5)
    save(fig, out, "F2")

    fig, ((ax,),) = new_figure(
        "How far did test accuracy finish below its peak?",
        "Symmetric label flips. The 40% error bar is one sample standard deviation across three seeds.",
        size=(9, 4.9),
    )
    drops = [[run.drop for run in sweep[rate]] for rate in RATES]
    averages = np.array([np.mean(group) for group in drops]) * 100
    ax.plot(np.array(RATES) * 100, averages, color="#A1ADC0", lw=1.7, zorder=1)
    for rate, values, mean, color in zip(RATES, drops, averages, RATE_COLORS):
        error = np.std(values, ddof=1) * 100 if len(values) > 1 else 0
        ax.errorbar(rate * 100, mean, yerr=error if len(values) > 1 else None,
                    fmt="o", color=color, markersize=8, capsize=5, zorder=3)
        ax.annotate(f"{mean:.1f}", (rate * 100, mean + error), xytext=(0, 8), textcoords="offset points",
                    ha="center", weight="bold", color=color)
    ax.set(xlim=(-6, 86), ylim=(0, max(averages) * 1.25), xlabel="Nominal label noise",
           ylabel="Peak-to-final drop (percentage points)",
           xticks=np.array(RATES) * 100, xticklabels=[f"{rate:.0%}" for rate in RATES])
    save(fig, out, "F3")

    random_rate = subset[1].config["noise_rate_train"]
    fig, ((ax,),) = new_figure(
        "The network can fit random training labels",
        f"10,000 training images. One seed per condition. Random replacement changed {random_rate:.1%} of labels.",
    )
    for run, color in zip(subset, (TRUE, WRONG)):
        series_plot(ax, [run], "train_acc_noisy", color)
        series_plot(ax, [run], "test_acc", color, style="--")
    for y, label in ((.99, "99% fit threshold"), (.10, "10% chance")):
        ax.axhline(y, color="#9DA9BF", ls=":", lw=1.3)
        ax.text(3, y + .018, label, color=MUTED, size=9.5)
    epoch_axis(ax, 150, "Accuracy", (0, 1.07))
    legend(fig, [swatch("Original labels", TRUE), swatch("Random labels", WRONG),
                 swatch("Training accuracy", INK), swatch("Clean test accuracy", INK, "--")], 4)
    save(fig, out, "F4")

    groups = [(f"Sym. {rate:.0%}", sweep[rate], RATE_COLORS[RATES.index(rate)], "-")
              for rate in (0.2, 0.4, 0.6)]
    groups.append(("Pair-flip 40%", pair, RATE_COLORS[4], "--"))
    fig, ((ax,),) = new_figure(
        "When does loss identify corrupted labels?",
        "A higher AUROC means better separation. Dots mark individual seed crossovers. The dotted line marks chance.",
    )
    ax.axhline(.5, color="#8491A7", ls=":", lw=1.4)
    for _, group, color, style in groups:
        series_plot(ax, group, "auroc_loss", color, style=style)
        for run in group:
            if run.t_cross is not None:
                ax.scatter(run.t_cross, run.epochs.at[run.t_cross, "auroc_loss"],
                           s=30, color=color, edgecolors="white", linewidths=.5, zorder=4)
    epoch_axis(ax, 60, "AUROC for detecting corrupted labels", (.45, 1.015), percentage=False)
    legend(fig, [swatch(f"{name} ({seed_label(group)})", color, style)
                 for name, group, color, style in groups], 4)
    save(fig, out, "F5")

    loss_histograms(out, (sweep[.4][0], pair[0]))
    method_curves(out, methods)


def loss_histograms(out, runs):
    fig, axes = new_figure(
        "How training loss changes for clean and corrupted labels",
        "40% noise, seed 0. Each column shares its bins across noise types. Loss scales differ by column.",
        (2, 3), (13.3, 8.2),
    )
    fig.subplots_adjust(top=.79, bottom=.16, wspace=.43, hspace=.61)
    snapshots = []
    for run in runs:
        _, _, mask = run.training_labels()
        mask = mask.astype(bool)
        row = []
        for epoch in (1, run.best_auc, 60):
            with np.load(run.path / "per_sample" / f"epoch_{epoch:03d}.npz", allow_pickle=False) as data:
                loss = data["loss"].astype(float)
            if len(loss) != len(mask) or not np.isfinite(loss).all() or np.any(loss < 0):
                raise ValueError(f"Invalid loss array for {run.name} epoch {epoch}")
            row.append((epoch, loss, mask))
        snapshots.append(row)
    for col, heading in enumerate(("First trained epoch", "Peak AUROC", "Final epoch")):
        scale = 1000 if col == 2 else 1
        right = max(row[col][1].max() for row in snapshots) * scale
        bins = np.linspace(0, right, 39)
        for row_id, (row, run) in enumerate(zip(snapshots, runs)):
            ax = axes[row_id, col]
            epoch, loss, mask = row[col]
            for segment, color in ((~mask, TRUE), (mask, WRONG)):
                histogram, _ = np.histogram(loss[segment] * scale, bins=bins)
                ax.stairs(histogram / segment.sum(), bins, color=color, lw=1.5, fill=True, alpha=.28)
                ax.stairs(histogram / segment.sum(), bins, color=color, lw=1.35)
            ax.set(xlim=(0, right), xlabel="Cross-entropy loss × 1,000" if scale == 1000 else "Cross-entropy loss")
            ax.set_title(f"{heading}\n{NOISES[run.args['noise']]} · epoch {epoch}", loc="left", fontsize=10.8)
            if col == 0:
                ax.set_ylabel("Share of subset per bin")
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ceiling = max(ax.get_ylim()[1] for ax in axes[:, col])
        for ax in axes[:, col]:
            ax.set_ylim(0, ceiling)
    legend(fig, [swatch("Unchanged label", TRUE), swatch("Corrupted label", WRONG)], 2)
    save(fig, out, "F6")


def warmup(group):
    values = {run.args["tk"] for run in group}
    if len(values) != 1:
        raise ValueError("Compared methods have different warm-up lengths")
    return values.pop()


def method_curves(out, methods):
    for name, key, title, ylabel, ylim in (
        ("F7", "test_acc", "Robust training improves clean test accuracy", "Clean test accuracy", (0, .8)),
        ("F8", "sel_precision", "How many selected examples have clean labels?", "Correct labels among selections", (.5, 1)),
    ):
        selection = key == "sel_precision"
        subtitle = "40% nominal noise. Means and seed ranges across three runs. Vertical dashes mark the warm-up limit."
        fig, axes = new_figure(title, subtitle, (1, 2), (12, 5.6), sharey=True)
        for ax, (noise, noise_label) in zip(axes[0], NOISES.items()):
            ax.set_title(noise_label, loc="left")
            ax.axvline(warmup(methods[noise, "coteach"]), color="#8491A7", ls="--", lw=1.25)
            if selection:
                bases = [run.base_clean for method in ("smallloss", "coteach") for run in methods[noise, method]]
                ax.axhline(np.mean(bases), color="#8491A7", ls=":", lw=1.6)
            for method, method_label in METHODS.items():
                if selection and method == "ce":
                    continue
                series_plot(ax, methods[noise, method], key, METHOD_COLORS[method], band=True)
            epoch_axis(ax, 60, ylabel if noise == "sym" else None, ylim)
        handles = [swatch(label, METHOD_COLORS[method]) for method, label in METHODS.items()
                   if not selection or method != "ce"]
        handles.append(swatch("End of warm-up", "#8491A7", "--"))
        if selection:
            handles.append(swatch("Clean share before selection (seed mean)", "#8491A7", ":"))
        legend(fig, handles, len(handles))
        save(fig, out, name)


def seed_label(group):
    return f"{len(group)} seed" + ("s" if len(group) != 1 else "")


def csv_table(out, name, rows, columns=None):
    frame = pd.DataFrame(rows, columns=columns)
    for column in frame.columns:
        if column.startswith("es_epoch_s"):
            frame[column] = frame[column].astype("Int64")
    frame.to_csv(out / f"{name}.csv", index=False, lineterminator="\n", float_format="%.5f")


def accuracy_at(run, epoch):
    return run.epochs.at[epoch, "test_acc"]


def summarize(values, digits=2):
    values = [value for value in values if value is not None and np.isfinite(value)]
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.{digits}f} (1 seed)"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def ce_row(run, case):
    h11 = h12 = "Not pre-specified for this condition"
    if case == "Clean":
        h11 = h12 = "Not applicable"
    elif run.args["noise"] == "sym" and run.args["eta"] <= .6:
        if run.t_cross is not None:
            h11 = "Supported"
        elif not run.had_recovery_lead:
            h11 = "Contradicted: no recovery lead"
        else:
            h11 = "Inconclusive: crossover not reached"
        if run.t_mem50 is None:
            h12 = "Inconclusive: 50% wrong-label fit not reached"
        elif run.t_clean90 is None or run.t_mem50 <= run.t_clean90:
            h12 = "Contradicted"
        else:
            h12 = "Supported"
    return {
        "condition": case, "run": run.name, "seeds": "1 seed",
        "realized_train_noise_pct": run.config["noise_rate_train"] * 100,
        "realized_val_noise_pct": run.config["noise_rate_val"] * 100,
        "t_clean90": run.t_clean90, "t_mem50": run.t_mem50, "t_cross": run.t_cross,
        "test_peak_pct": accuracy_at(run, run.best_test) * 100,
        "test_peak_epoch": run.best_test, "final_test_pct": run.final.test_acc * 100,
        "peak_drop_pp": run.drop * 100, "es_epoch": run.es,
        "test_at_es_pct": accuracy_at(run, run.es) * 100,
        "max_auroc": run.epochs.at[run.best_auc, "auroc_loss"] if run.best_auc is not None else None,
        "max_auroc_epoch": run.best_auc, "final_auroc": run.final.auroc_loss,
        "h1_1_verdict": h11, "h1_2_verdict": h12,
    }


def main_tables(out, sweep, pair, subset, methods):
    ce_groups = [("Clean" if rate == 0 else f"Symmetric {rate:.0%}", sweep[rate]) for rate in RATES]
    ce_groups.append(("Pair-flip 40%", pair))
    ce_rows = []
    for case, group in ce_groups:
        rows = [ce_row(run, case) for run in group]
        ce_rows.extend(rows)
        if len(rows) > 1:
            summary = {"condition": case, "run": "Mean ± SD", "seeds": seed_label(group)}
            for key in rows[0]:
                values = [row[key] for row in rows]
                if key.startswith("h1_"):
                    summary[key] = "; ".join(f"s{i} {value}" for i, value in enumerate(values))
                elif key not in summary:
                    digits = 4 if key in ("max_auroc", "final_auroc") else 2
                    text = summarize(values, digits)
                    n = sum(value is not None and np.isfinite(value) for value in values)
                    summary[key] = f"{text} ({n}/{len(rows)} reached)" if 1 < n < len(rows) else text
            ce_rows.append(summary)
    csv_table(out, "T1", [{key: round(value, 5) if isinstance(value, (float, np.floating)) else value
                           for key, value in row.items()} for row in ce_rows])
    csv_table(out, "T2", [{
        "run": run.name, "seeds": "1 seed", "realized_train_noise_pct": run.config["noise_rate_train"] * 100,
        "first_epoch_at_99pct": first(run.trained.train_acc_noisy >= .99),
        "final_accuracy_on_training_labels_pct": run.final.train_acc_noisy * 100,
        "final_clean_test_accuracy_pct": run.final.test_acc * 100,
    } for run in subset])

    method_rows, comparisons, precision_rows = [], [], []
    for noise, noise_label in NOISES.items():
        final = {method: np.array([run.final.test_acc * 100 for run in methods[noise, method]])
                 for method in METHODS}
        ce = methods[noise, "ce"]
        es = np.array([accuracy_at(run, run.es) * 100 for run in ce])
        for method, method_label in METHODS.items():
            runs = methods[noise, method]
            row = {"noise": noise_label, "method": method_label, "seeds": seed_label(runs),
                   "realized_train_noise_pct": summarize([run.config["noise_rate_train"] * 100 for run in runs]),
                   "final_test_mean_pct": final[method].mean(), "final_test_sd_pp": final[method].std(ddof=1)}
            row.update({f"final_s{i}_pct": x for i, x in enumerate(final[method])})
            if method == "ce":
                row.update(es_test_mean_pct=es.mean(), es_test_sd_pp=es.std(ddof=1))
                row.update({f"es_s{i}_pct": x for i, x in enumerate(es)})
                row.update({f"es_epoch_s{i}": run.es for i, run in enumerate(ce)})
            method_rows.append(row)

            if method != "ce":
                second_half = [run.epochs.loc[31:60, "sel_precision"].mean() * 100 for run in runs]
                post_warmup = [run.epochs.loc[11:60, "sel_precision"] for run in runs]
                if any(len(values) != 50 or values.isna().any() for values in post_warmup):
                    raise ValueError(f"Missing selection precision for {noise} {method}")
                floors = [values.min() * 100 for values in post_warmup]
                bases = [run.base_clean * 100 for run in runs]
                row = {"noise": noise_label, "method": method_label, "seeds": seed_label(runs),
                       "mean_precision_epochs_31_to_60_pct": np.mean(second_half),
                       "sd_of_seed_means_pp": np.std(second_half, ddof=1),
                       "min_precision_epochs_11_to_60_pct": min(floors),
                       "mean_base_clean_pct": np.mean(bases),
                       "h3_4_all_post_warmup_epochs_all_seeds": bool(np.all(np.array(floors) > bases))}
                for i in range(len(runs)):
                    row[f"base_clean_s{i}_pct"] = bases[i]
                    row[f"min_post_warmup_precision_s{i}_pct"] = floors[i]
                precision_rows.append(row)

        a = final["coteach"]
        for hypothesis, b_label, b in (
            ("H3.1", "Cross-entropy final", final["ce"]),
            ("H3.2", "Cross-entropy at early stopping", es),
            ("H3.3", "Small-loss final", final["smallloss"]),
        ):
            margin = (a - b).mean()
            threshold = 2 * max(a.std(ddof=1), b.std(ddof=1))
            all_ge = bool(np.all(a >= b))
            reverse_all_ge = bool(np.all(b >= a))
            verdict = "No clear difference"
            if margin > threshold and all_ge:
                verdict = "Co-teaching final outperforms"
            elif hypothesis == "H3.2" and -margin > threshold and reverse_all_ge:
                verdict = "Cross-entropy at early stopping outperforms"
            comparisons.append({"noise": noise_label, "hypothesis": hypothesis, "method_a": "Co-teaching final",
                                "method_b": b_label, "mean_a_minus_b_pp": margin, "twice_larger_sd_pp": threshold,
                                "a_ge_b_all_seeds": all_ge, "b_ge_a_all_seeds": reverse_all_ge, "verdict": verdict})
    csv_table(out, "T3", method_rows)
    csv_table(out, "T4", comparisons)
    csv_table(out, "T5", precision_rows)

def descriptive_diagnostics(out, sweep, pair):
    groups = [(f"Sym. {rate:.0%}", sweep[rate], color)
              for rate, color in zip(RATES[1:], RATE_COLORS[1:])]
    groups.append(("Pair 40%", pair, "#252F43"))
    fig, ((ax,),) = new_figure(
        "The test-accuracy dip around memorization",
        "Exploratory. Early high minus post-crossover low. Bars show seed means and dots show individual seeds.",
        size=(9.9, 5.1),
    )
    for i, (_, group, color) in enumerate(groups):
        values = []
        for run in group:
            if run.t_cross is not None and run.t_cross < run.trained.index[-1]:
                high = run.trained.loc[:run.t_cross, "test_acc"].max()
                low = run.trained.loc[run.t_cross + 1:, "test_acc"].min()
                values.append((high - low) * 100)
        if values:
            ax.bar(i, np.mean(values), width=.55, color=color, alpha=.23)
            offsets = np.linspace(-.15, .15, len(values)) if len(values) > 1 else [0]
            ax.scatter(i + np.array(offsets), values, color=color, s=45, zorder=3)
        else:
            ax.text(i, .02, "No crossover", transform=ax.get_xaxis_transform(), ha="center", color=MUTED)
    ax.axhline(0, color="#A1ADC0", lw=1)
    ax.set(xticks=range(len(groups)), xticklabels=[f"{name}\n{seed_label(group)}" for name, group, _ in groups],
           ylabel="Drop (percentage points)", xlim=(-.6, len(groups) - .4))
    fig.subplots_adjust(bottom=.18)
    save(fig, out, "X1")

    fig, axes = new_figure(
        "Clean and corrupted fitting at the same threshold",
        "Exploratory. Overlapping dots are nudged sideways. Curves show seed means.",
        (1, 2), (12.5, 5.6),
    )
    ax, trend = axes[0]
    ax.set_title("First epoch reaching 50%", loc="left")
    ax.plot([0, 60], [0, 60], color="#8491A7", ls=":", lw=1.3)
    points = defaultdict(list)
    for group_index, (label, group, color) in enumerate(groups):
        reached = [(run.t_clean50, run.t_mem50) for run in group
                   if run.t_clean50 is not None and run.t_mem50 is not None]
        for point in reached:
            points[point].append(color)
        missed = len(group) - len(reached)
        if missed:
            ax.text(.03, .96 - .06 * group_index, f"{label}: {missed} without both thresholds", transform=ax.transAxes,
                    color=color, va="top")
        gap = np.stack([(run.epochs.fit_clean - run.epochs.memorized).to_numpy() for run in group])
        trend.plot(group[0].epochs.index, gap.mean(axis=0), color=color, lw=1.9,
                   ls="--" if label.startswith("Pair") else "-")
    for (x, y), colors in points.items():
        offsets = np.linspace(-.27, .27, len(colors)) if len(colors) > 1 else [0]
        ax.scatter(x + np.array(offsets), [y] * len(colors), c=colors, s=42,
                   edgecolors="white", linewidths=.6, zorder=3)
    reached_max = max(max(run.t_clean50 or 0, run.t_mem50 or 0) for _, group, _ in groups for run in group)
    limit = max(10, reached_max + 2)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Epoch when clean fit reaches 50%",
           ylabel="Epoch when wrong-label fit reaches 50%")
    trend.set_title("Gap over training", loc="left")
    trend.axhline(0, color="#8491A7", ls=":", lw=1.3)
    epoch_axis(trend, 60, "Clean fit − wrong-label fit")
    legend(fig, [swatch(f"{label} ({seed_label(group)})", color, "--" if label.startswith("Pair") else "-")
                 for label, group, color in groups], 5)
    save(fig, out, "X2")

    fig, axes = new_figure(
        "The validation scores used to choose an epoch",
        "Exploratory. 40% nominal noise. The dotted marker uses noisy validation in every panel.",
        (2, 3), (13.5, 7.8), sharex=True, sharey=True,
    )
    fig.subplots_adjust(top=.79, bottom=.17, hspace=.43, wspace=.19)
    metrics = (("val_acc_noisy", "Noisy validation", WRONG, "-"),
               ("val_acc_clean", "Clean validation", TRUE, "--"),
               ("test_acc", "Clean test", CLEAN, "-"))
    for row, group in enumerate((sweep[.4], pair)):
        for col, run in enumerate(group):
            ax = axes[row, col]
            name = NOISES[run.args["noise"]]
            ax.set_title(f"{name} · seed {run.args['seed']} · ES {run.es}", loc="left", fontsize=10.8)
            ax.axvline(run.es, color="#8491A7", ls=":", lw=1.4)
            for key, _, color, style in metrics:
                series_plot(ax, [run], key, color, style=style)
            epoch_axis(ax, 60, "Accuracy" if col == 0 else None, (0, .75))
            if row == 0:
                ax.set_xlabel("")
    legend(fig, [swatch(label, color, style) for _, label, color, style in metrics]
           + [swatch("Early-stopping epoch", "#8491A7", ":")], 4)
    save(fig, out, "X3")


def transition_heatmaps(out, runs):
    fig, axes = new_figure(
        "Where the flipped labels go",
        "Realized CIFAR-10 training transitions at 40% nominal noise, seed 0. Each row sums to 100%.",
        (1, 2), (12.2, 6.3),
    )
    fig.subplots_adjust(top=.79, left=.10, bottom=.22, right=.86, wspace=.24)
    for ax, run in zip(axes[0], runs):
        true, noisy, mask = run.training_labels()
        if not np.array_equal(true != noisy, mask) or not np.isclose(mask.mean(), 1 - run.base_clean):
            raise ValueError(f"Label and configuration mismatch for {run.name}")
        counts = np.zeros((10, 10))
        np.add.at(counts, (true, noisy), 1)
        transition = counts / counts.sum(axis=1, keepdims=True) * 100
        image = ax.imshow(transition, cmap="Blues", vmin=0, vmax=65)
        ax.grid(False)
        ax.set_title(f"{NOISES[run.args['noise']]} · {mask.mean():.2%} realized", loc="left")
        ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Assigned label", ylabel="Original label")
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for i, j in np.argwhere(transition >= 0.05):
            value = transition[i, j]
            ax.text(j, i, f"{value:.0f}" if value >= 10 else f"{value:.1f}",
                    ha="center", va="center", fontsize=7.3, color="white" if value > 38 else INK)
    bar = fig.colorbar(image, cax=fig.add_axes((.91, .26, .017, .48)))
    bar.set_label("Percent of original class")
    save(fig, out, "S1")


def supporting_tables(out, main, optional):
    inventory, drops = [], []
    for source, runs in (("Main raw", main), ("Optional Kaggle", optional)):
        for run in runs.values():
            log = (run.path / "train.log").read_text(encoding="utf-8", errors="replace")
            failures = [line.strip() for line in log.splitlines() if "RUN FAILED" in line]
            inventory.append({
                "source": source, "run": run.name, "epochs": len(run.trained), "seed": run.args["seed"],
                "realized_train_noise_pct": (1 - run.base_clean) * 100,
                "realized_val_noise_pct": run.config["noise_rate_val"] * 100,
                "git_commit": run.config["git"]["commit"],
                "uncommitted_changes": run.config["git"]["uncommitted_changes"],
                "gpu": run.config["gpu"], "torch": run.config["torch"],
                "mixed_precision": run.config["mixed_precision"],
                "mean_seconds_per_trained_epoch": run.trained.seconds.mean(),
                "resumes": sum("resuming " in line for line in log.splitlines()),
                "failures": len(failures), "failure_log_entries": " | ".join(failures),
            })
            accuracy = run.epochs.test_acc
            for epoch in accuracy.index[(accuracy.index > 20) & (accuracy.diff() < -.10)]:
                drops.append({"source": source, "run": run.name, "epoch": epoch,
                              "previous_test_pct": accuracy.at[epoch - 1] * 100,
                              "test_pct": accuracy.at[epoch] * 100,
                              "drop_pp": (accuracy.at[epoch - 1] - accuracy.at[epoch]) * 100})
    csv_table(out, "S2", inventory)
    csv_table(out, "S3", drops, ["source", "run", "epoch", "previous_test_pct", "test_pct", "drop_pp"])


OPTIONAL_INPUTS = {
    "O1": ("sym0.4_ce_s0", "sym0.4_ce_aug_s0"),
    "O2": ("sym0.4_coteach_tau0.3_s0", "sym0.4_coteach_s0", "sym0.4_coteach_tau0.5_s0"),
    "O3": ("sym0.4_ce_s0", "sym0.4_gce_s0", "pair0.4_gce_s0"),
    "O4": ("c10n-worst_ce_s0", "c10n-worst_smallloss_s0", "c10n-worst_coteach_s0"),
}


def optional_figures(out, optional, main_pair):
    missing = {}
    for name, required in OPTIONAL_INPUTS.items():
        absent = [run for run in required if run not in optional]
        if absent:
            missing[name] = absent
            continue
        runs = [optional[run] for run in required]
        if name == "O1":
            fig, axes = new_figure("Augmentation and memorization",
                                   "Exploratory · Kaggle · symmetric 40% noise · one seed",
                                   (1, 2), (11.5, 5.5))
            for ax, key, label in zip(axes[0], ("memorized", "test_acc"),
                                      ("Wrong-label fit", "Clean test accuracy")):
                for run, color in zip(runs, (CLEAN, TRUE)):
                    series_plot(ax, [run], key, color)
                epoch_axis(ax, 60, label, (0, 1.02))
            legend(fig, [swatch("Augmentation off", CLEAN), swatch("Augmentation on", TRUE)], 2)
        elif name == "O2":
            fig, ((ax,),) = new_figure("How sensitive is Co-teaching to its assumed noise rate?",
                                       "Exploratory · Kaggle · symmetric 40% noise · one seed", size=(11, 5.4))
            tau = [run.config["tau_used"] for run in runs]
            ax.plot(tau, [run.final.test_acc for run in runs], "-o", color=TRUE)
            ax.plot(tau, [accuracy_at(run, run.es) for run in runs], "--s", color=WRONG)
            ax.set(xticks=tau, xlabel="Assumed noise rate (τ)", ylabel="Clean test accuracy")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            legend(fig, [swatch("Final", TRUE, marker="o"),
                         swatch("At noisy-validation choice", WRONG, "--", "s")], 2)
        elif name == "O3":
            fig, axes = new_figure("Generalized cross-entropy and clean test accuracy",
                                   "Exploratory · 40% nominal noise · one seed · the pair-flip hardware differs",
                                   (1, 2), (12, 5.5), sharey=True)
            for ax, group, noise in zip(axes[0], (runs[:2], (main_pair, runs[2])), NOISES.values()):
                for run, label, color in zip(group, ("Cross-entropy", "Generalized cross-entropy"), (CLEAN, TRUE)):
                    series_plot(ax, [run], "test_acc", color)
                    ax.plot([], [], color=color, label=f"{label} ({run.config['gpu']})")
                ax.set_title(noise, loc="left")
                epoch_axis(ax, 60, "Clean test accuracy" if ax is axes[0, 0] else None, (0, .85))
                ax.legend(loc="lower right", fontsize=8.5)
        else:
            fig, ((ax,),) = new_figure("Training on CIFAR-10N human labels",
                                       "Exploratory · Kaggle · CIFAR-10N Worst labels · one seed per method")
            handles = []
            for run, (method, label) in zip(runs, METHODS.items()):
                color = METHOD_COLORS[method]
                series_plot(ax, [run], "test_acc", color)
                handles.append(swatch(f"{label} ({1 - run.base_clean:.2%} realized noise)", color))
            epoch_axis(ax, 60, "Clean test accuracy", (0, .85))
            legend(fig, handles, 3)
        save(fig, out, name)
    return missing


def write_index(out, missing, has_pilot):
    figures = {
        "F1": "Clean fit, recovery and wrong-label fit over training",
        "F2": "Clean test trajectories and noisy-validation early stopping",
        "F3": "Peak-to-final test accuracy drop by noise rate",
        "F4": "Training and test accuracy with true versus random labels",
        "F5": "Loss-based detection AUROC over training",
        "F6": "Clean versus corrupted per-image loss histograms",
        "F7": "Clean test accuracy for the three training methods",
        "F8": "Selection precision for small-loss and Co-teaching",
        "X1": "Test accuracy dip around crossover",
        "X2": "Matched 50% thresholds and fit gap over training",
        "X3": "Noisy versus clean validation and test trajectories",
        "S1": "Realized training-label transition matrices",
    }
    tables = {"T1": "Noise sweep and per-run diagnostics", "T2": "Random-label fit thresholds",
              "T3": "Per-seed and summary method scores", "T4": "Pre-specified method decision rule",
              "T5": "Selection precision against realized clean share", "S2": "Run inventory and log events",
              "S3": "Clean test drops strictly greater than 10 percentage points after epoch 20"}
    lines = ["# CIFAR-10 results", "", "## Figures", "", "| ID | Figure | Files |", "| --- | --- | --- |"]
    lines.extend(f"| {key} | {label} | [PNG]({key}.png) · [PDF]({key}.pdf) |"
                 for key, label in figures.items())
    if has_pilot:
        lines.append("| S4 | Existing pilot figure | [PNG](S4.png) |")
    lines.extend(["", "## Tables", ""])
    lines.extend(f"- [{key}]({key}.csv) · {label}" for key, label in tables.items())
    lines.extend(["", "## Optional Kaggle figures", ""])
    for key in OPTIONAL_INPUTS:
        if key in missing:
            lines.append(f"- {key} requires these missing run folders: " +
                         ", ".join(f"`{name}`" for name in missing[key]) + ".")
        else:
            lines.append(f"- {key} · [PNG]({key}.png) · [PDF]({key}.pdf)")
    lines.extend([
        "", "## Reading the results", "",
        "- Figures X1–X3 and O1–O4 are exploratory. S4 is the pre-existing pilot image.",
        "- Epoch 0 appears in trajectories when a measurement exists. Calculated milestones use trained epochs.",
        "- Early stopping (ES) uses the highest noisy validation accuracy in epochs 1–60, with ties going to the earliest epoch. Test peaks are descriptive. Test labels are clean.",
        "- A blank milestone means the threshold was not reached or the measurement did not apply. F1 shows any distinct seed-specific milestone as a vertical line. F2 and F5 show individual seed markers even where the line is a seed mean.",
        "- Shaded bands are minimum to maximum across seeds. SD in the tables and F3 is sample standard deviation. The method comparison rule in T4 is a rule from the analysis plan, with no significance claim.",
        "- X1 uses the highest score up to and including crossover and the lowest score strictly after it. F6 reports a share of each subset per bin, and the final column rescales the loss axis by 1,000.",
        "- F8 shows the mean realized clean share as a visual reference. T5 tests each seed against its own realized clean share over every post-warm-up epoch.",
        "- T4 covers the final-accuracy part of H3.3. T5 contains the second-half selection precision for its other condition.",
        "", "## Regenerate", "",
        "From the repository root, using the NumPy, pandas and Matplotlib dependencies in `pyproject.toml`:", "",
        "```sh",
        "python scripts/plot_results.py --raw ../raw --output results/figures",
        "```", "",
        "Use `--kaggle PATH` if the optional runs are stored outside `results/kaggle`. No GPU or training is needed.",
    ])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "results/raw")
    parser.add_argument("--kaggle", type=Path, default=ROOT / "results/kaggle")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures")
    args = parser.parse_args()
    runs, optional = load_runs(args.raw), load_runs(args.kaggle)
    if not runs:
        parser.error(f"No run logs in {args.raw}. Pass the folder containing the runs with --raw.")
    try:
        sweep = {0: [runs["clean_ce_s0"]], .4: seed_group(runs, "sym0.4_ce")}
        sweep.update({rate: [runs[f"sym{rate}_ce_s0"]] for rate in (.2, .6, .8)})
        subset = [runs[f"sub10k_{mode}_ce_s0"] for mode in ("clean", "random")]
        methods = {(noise, method): seed_group(runs, f"{noise}0.4_{method}")
                   for noise in NOISES for method in METHODS}
    except KeyError as err:
        parser.error(f"Missing required run {err.args[0]} in {args.raw}")
    pair = methods["pair", "ce"]
    args.output.mkdir(parents=True, exist_ok=True)
    main_figures(args.output, sweep, pair, subset, methods)
    main_tables(args.output, sweep, pair, subset, methods)
    descriptive_diagnostics(args.output, sweep, pair)
    transition_heatmaps(args.output, (sweep[.4][0], pair[0]))
    supporting_tables(args.output, runs, optional)
    pilot = ROOT / "results/pilot.png"
    if pilot.is_file():
        shutil.copyfile(pilot, args.output / "S4.png")
    missing = optional_figures(args.output, optional, pair[0])
    write_index(args.output, missing, pilot.is_file())
    print(f"Generated figures and tables in {args.output.resolve()}")
    if missing:
        print("Optional figures missing inputs: " + ", ".join(missing))


if __name__ == "__main__":
    main()

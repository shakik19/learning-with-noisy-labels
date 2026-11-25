"""Controlled label corruption (numpy only, fully determined by the generator passed in)."""

from __future__ import annotations

import numpy as np


def corrupt_labels(y: np.ndarray, kind: str, eta: float, num_classes: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy_labels, is_corrupted).

    sym    : each label flipped with probability eta to a class drawn uniformly from the
             *other* K-1 classes (a flip never lands on the true class).
    pair   : each label flipped with probability eta to (y + 1) mod K.
    random : every label resampled uniformly from all K classes (Zhang et al. 2017); about
             1/K of them land on the true class by chance and are therefore not corrupted.
    none   : labels unchanged.
    """
    y = np.asarray(y, dtype=np.int64)
    n, k = len(y), num_classes
    noisy = y.copy()
    if kind == "none":
        pass
    elif kind == "sym":
        flip = rng.random(n) < eta
        noisy[flip] = (y[flip] + rng.integers(1, k, size=int(flip.sum()))) % k
    elif kind == "pair":
        flip = rng.random(n) < eta
        noisy[flip] = (y[flip] + 1) % k
    elif kind == "random":
        noisy = rng.integers(0, k, size=n).astype(np.int64)
    else:
        raise ValueError(f"unknown noise kind: {kind}")
    return noisy, noisy != y


def transition_matrix(y: np.ndarray, noisy: np.ndarray, num_classes: int) -> np.ndarray:
    """Empirical T[i, j] = P(noisy = j | true = i)."""
    counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(counts, (np.asarray(y), np.asarray(noisy)), 1)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

"""Write a small fake CIFAR-10 in the original python-pickle layout (for tests and smoke runs).

Images are a class prototype plus noise, so a network can learn them and accuracy visibly rises.

    python tests/fake_cifar.py OUT_DIR [--per-batch 1000] [--test 1000]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def write_fake_cifar(out_dir: str | Path, per_batch: int = 1000, n_test: int = 1000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(out_dir) / "cifar-10-batches-py"
    root.mkdir(parents=True, exist_ok=True)
    protos = rng.integers(0, 256, size=(10, 3072))

    def make(n: int, name: str):
        y = np.arange(n) % 10
        rng.shuffle(y)
        x = (0.6 * protos[y] + 0.4 * rng.integers(0, 256, size=(n, 3072))).astype(np.uint8)
        with open(root / name, "wb") as f:
            pickle.dump({b"data": x, b"labels": y.tolist()}, f)

    for i in range(1, 6):
        make(per_batch, f"data_batch_{i}")
    make(n_test, "test_batch")
    return Path(out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--per-batch", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    a = ap.parse_args()
    print(write_fake_cifar(a.out_dir, a.per_batch, a.test))

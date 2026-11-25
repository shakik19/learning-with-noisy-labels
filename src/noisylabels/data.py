"""CIFAR-10 loading, the fixed train/validation split, and GPU-resident batches.

The whole dataset is kept on the device as uint8 (about 180 MB), so batches are formed by
indexing and no DataLoader / CPU workers are involved.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10N_FILE = "CIFAR-10_human.pt"
CIFAR10N_MD5 = "96644db5574813cae939af5f7f2a0aec"


def load_cifar10(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x_train (N,3,32,32) uint8, y_train, x_test, y_test in the original file order."""
    root = Path(data_dir) / "cifar-10-batches-py"
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found. See data/README.md for how to download CIFAR-10.")

    def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
        with open(root / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32), np.asarray(d[b"labels"], dtype=np.int64)

    parts = [_load(f"data_batch_{i}") for i in range(1, 6)]
    x_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    x_test, y_test = _load("test_batch")
    return x_train, y_train, x_test, y_test


def stratified_split(y: np.ndarray, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed stratified split of all training indices into (train_idx, val_idx), both sorted."""
    rng = np.random.default_rng(seed)
    per_class = val_size // NUM_CLASSES
    val = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(y == c)
        if len(idx) <= per_class:
            raise ValueError(f"class {c} has {len(idx)} samples; cannot hold out {per_class}")
        val.append(rng.permutation(idx)[:per_class])
    val_idx = np.sort(np.concatenate(val))
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    return train_idx, val_idx


def stratified_subset(idx: np.ndarray, y: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Stratified subset of `idx` with `size` elements (sorted)."""
    rng = np.random.default_rng([seed, 1])
    per_class = size // NUM_CLASSES
    out = []
    for c in range(NUM_CLASSES):
        cls = idx[y[idx] == c]
        if len(cls) < per_class:
            raise ValueError(f"class {c} has {len(cls)} samples; subset needs {per_class}")
        out.append(rng.permutation(cls)[:per_class])
    return np.sort(np.concatenate(out))


def load_cifar10n(data_dir: str | Path, key: str,
                  expected_md5: str | None = CIFAR10N_MD5) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy_labels, clean_labels) from the official CIFAR-10N file.

    The file is a pickle, so its checksum is verified before it is loaded.
    """
    import torch

    path = Path(data_dir) / CIFAR10N_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. See data/README.md.")
    if expected_md5 is not None:
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if md5 != expected_md5:
            raise ValueError(f"{path} has md5 {md5}, expected {expected_md5}; refusing to unpickle it")
    d = torch.load(path, weights_only=False)  # dict of numpy label arrays (verified file)
    return np.asarray(d[key], dtype=np.int64), np.asarray(d["clean_label"], dtype=np.int64)


class DeviceImages:
    """Images kept on the device as uint8; `batch()` returns normalised float tensors."""

    def __init__(self, x_uint8: np.ndarray, device, channels_last: bool = False):
        import torch

        self.torch = torch
        self.x = torch.from_numpy(np.ascontiguousarray(x_uint8)).to(device)
        self.mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
        self.channels_last = channels_last

    def __len__(self) -> int:
        return self.x.shape[0]

    def batch(self, idx, augment: bool = False):
        x = self.x[idx].float().div_(255.0)
        if augment:
            x = random_crop_flip(x)
        x = (x - self.mean) / self.std
        if self.channels_last:
            x = x.contiguous(memory_format=self.torch.channels_last)
        return x


def random_crop_flip(x, pad: int = 4):
    """Per-sample random crop (zero padding) and horizontal flip, vectorised on the device."""
    import torch
    import torch.nn.functional as F

    b, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad)).permute(0, 2, 3, 1)  # B, H+2p, W+2p, C
    dy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    dx = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = dy[:, None] + torch.arange(h, device=x.device)
    cols = dx[:, None] + torch.arange(w, device=x.device)
    bidx = torch.arange(b, device=x.device)[:, None, None]
    out = xp[bidx, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out).contiguous()

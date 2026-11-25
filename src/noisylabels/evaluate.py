"""End-of-epoch evaluation: per-sample records on the training split plus summary metrics.

Everything is measured in eval mode without augmentation, so every sample is scored by the same
model state. Per-sample quantities are always computed with plain cross-entropy against the
*noisy* label, whatever loss the method trains with, so runs stay comparable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def predict_logits(net, images, idx: torch.Tensor, batch_size: int, autocast) -> torch.Tensor:
    out = []
    for s in range(0, idx.numel(), batch_size):
        with autocast():
            out.append(net(images.batch(idx[s:s + batch_size])).float())
    return torch.cat(out)


def _margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Logit of the given label minus the largest other logit (negative = predicted elsewhere)."""
    z_y = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.scatter(1, y[:, None], float("-inf")).amax(1)
    return z_y - other


def _mean(mask_values: np.ndarray) -> float:
    return float(mask_values.mean()) if mask_values.size else float("nan")


@torch.no_grad()
def evaluate(learner, train_images, test_images, tensors: dict, batch_size: int) -> tuple[dict, dict]:
    """Return (per_sample arrays for the training split, metrics dict)."""
    learner.eval()
    net = learner.nets[0]
    ac = learner.autocast

    # --- training split: per-sample records -------------------------------------------
    tr = tensors["train_idx"]
    y_noisy = tensors["y_noisy"][tr]
    logits = predict_logits(net, train_images, tr, batch_size, ac)
    loss = F.cross_entropy(logits, y_noisy, reduction="none")
    pred = logits.argmax(1)
    margin = _margin(logits, y_noisy)

    loss_np = loss.cpu().numpy()
    pred_np = pred.cpu().numpy()
    noisy_np = y_noisy.cpu().numpy()
    true_np = tensors["y_true"][tr].cpu().numpy()
    corr_np = tensors["corrupted"][tr].cpu().numpy()
    clean_np = ~corr_np

    m = {
        "train_acc_noisy": _mean(pred_np == noisy_np),
        "fit_clean": _mean(pred_np[clean_np] == noisy_np[clean_np]),
        "recovered": _mean(pred_np[corr_np] == true_np[corr_np]),
        "memorized": _mean(pred_np[corr_np] == noisy_np[corr_np]),
        "loss_clean": _mean(loss_np[clean_np]),
        "loss_corrupted": _mean(loss_np[corr_np]),
        "auroc_loss": (float(roc_auc_score(corr_np, loss_np))
                       if 0 < corr_np.sum() < corr_np.size else float("nan")),
    }

    # --- validation split (noisy, as in practice; clean accuracy kept for analysis) -----
    va = tensors["val_idx"]
    pv = predict_logits(net, train_images, va, batch_size, ac).argmax(1)
    m["val_acc_noisy"] = (pv == tensors["y_noisy"][va]).float().mean().item()
    m["val_acc_clean"] = (pv == tensors["y_true"][va]).float().mean().item()

    # --- test set (clean labels) ----------------------------------------------------------
    te = tensors["test_idx"]
    for i, n in enumerate(learner.nets):
        pt = predict_logits(n, test_images, te, batch_size, ac).argmax(1)
        key = "test_acc" if i == 0 else f"test_acc_net{i + 1}"
        m[key] = (pt == tensors["y_test"]).float().mean().item()

    learner.train()
    per_sample = {
        "loss": loss_np.astype(np.float16),
        "pred": pred_np.astype(np.uint8),
        "margin": margin.cpu().numpy().astype(np.float16),
    }
    return per_sample, m

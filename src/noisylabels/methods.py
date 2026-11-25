"""Training methods: cross-entropy, GCE, small-loss selection and Co-teaching.

All methods share one `Learner`, so they differ only in how a minibatch update is formed.

Network 1 is always created first right after `torch.manual_seed(seed)`, so for a given seed it
has the same initial weights under every method. Co-teaching's network 2 is created next from
the continuing random stream, so it has a different (but reproducible) initialisation.

`smallloss` uses a single network that trains on its own small-loss selection. This is exactly
network 1 of "Co-teaching without the cross-update": without the exchange the two networks never
interact, so the second one would only add compute.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from .model import build_model

SELECTION_METHODS = ("smallloss", "coteach")


def per_sample_loss(logits: torch.Tensor, y: torch.Tensor, loss: str, q: float | None = None):
    logits = logits.float()
    if loss == "ce":
        return F.cross_entropy(logits, y, reduction="none")
    if loss == "gce":  # Zhang & Sabuncu (2018): (1 - p_y^q) / q
        p_y = F.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
        return (1.0 - p_y.pow(q)) / q
    raise ValueError(f"unknown loss: {loss}")


def forget_rate(epoch: int, tau: float, tk: int) -> float:
    """Fraction of each minibatch to drop in training epoch `epoch` (1-based).

    Han et al. (2018): R(t) = 1 - tau * min(t / T_k, 1) is the fraction *kept*, with t counted
    from 0, so the first epoch keeps everything and the full rate tau is reached at t = T_k.
    """
    return tau * min((epoch - 1) / tk, 1.0)


class Learner:
    def __init__(self, args, device: torch.device, amp_dtype: torch.dtype | None, tau: float | None):
        self.method = args.method
        self.loss = "gce" if args.method == "gce" else "ce"
        self.q = args.gce_q
        self.tau = tau
        self.tk = args.tk
        self.device = device
        self.amp_dtype = amp_dtype
        channels_last = device.type == "cuda"

        torch.manual_seed(args.seed)
        n_nets = 2 if args.method == "coteach" else 1
        self.nets = []
        for _ in range(n_nets):
            net = build_model(args.arch).to(device)
            if channels_last:
                net = net.to(memory_format=torch.channels_last)
            self.nets.append(net)
        self.opts = [
            torch.optim.SGD(n.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
            for n in self.nets
        ]
        use_scaler = amp_dtype == torch.float16
        self.scalers = [torch.amp.GradScaler("cuda", enabled=use_scaler) for _ in self.nets]
        self._reset_stats()

    # ---- precision context -------------------------------------------------------------
    def autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    # ---- per-epoch statistics (kept on the device; synchronised once per epoch) --------
    def _reset_stats(self):
        z = lambda: torch.zeros((), device=self.device, dtype=torch.float64)  # noqa: E731
        self._stats = {"loss_sum": z(), "n_updates": z(), "n_seen": z(),
                       "sel1_clean": z(), "sel2_clean": z(), "n_sel": z()}

    def pop_epoch_stats(self) -> dict:
        s = {k: v.item() for k, v in self._stats.items()}
        self._reset_stats()
        out = {"train_loss": s["loss_sum"] / max(s["n_updates"], 1)}
        if self.method in SELECTION_METHODS:
            out["sel_precision"] = s["sel1_clean"] / max(s["n_sel"], 1)
            out["sel_fraction"] = s["n_sel"] / max(s["n_seen"], 1)
        if self.method == "coteach":
            out["sel_precision_net2"] = s["sel2_clean"] / max(s["n_sel"], 1)
        return out

    # ---- one minibatch -------------------------------------------------------------------
    def _update(self, i: int, loss: torch.Tensor):
        opt, scaler = self.opts[i], self.scalers[i]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    def step(self, x: torch.Tensor, y: torch.Tensor, corrupted: torch.Tensor, fr: float):
        losses = []
        for net in self.nets:
            with self.autocast():
                logits = net(x)
            losses.append(per_sample_loss(logits, y, self.loss, self.q))

        b = y.shape[0]
        st = self._stats
        st["n_seen"] += b
        if self.method in ("ce", "gce"):
            loss = losses[0].mean()
            self._update(0, loss)
            st["loss_sum"] += loss.detach()
            st["n_updates"] += 1
            return

        keep = max(1, int(round((1.0 - fr) * b)))
        pick1 = torch.argsort(losses[0].detach())[:keep]  # samples network 1 believes are clean
        st["sel1_clean"] += (~corrupted[pick1]).sum()
        st["n_sel"] += keep
        if self.method == "smallloss":
            loss = losses[0][pick1].mean()
            self._update(0, loss)
        else:  # coteach: each network learns from the samples its peer selected
            pick2 = torch.argsort(losses[1].detach())[:keep]
            st["sel2_clean"] += (~corrupted[pick2]).sum()
            loss = losses[0][pick2].mean()
            self._update(0, loss)
            self._update(1, losses[1][pick1].mean())
        st["loss_sum"] += loss.detach()
        st["n_updates"] += 1

    def train(self):
        for n in self.nets:
            n.train()

    def eval(self):
        for n in self.nets:
            n.eval()

    def state_dict(self) -> dict:
        return {
            "nets": [n.state_dict() for n in self.nets],
            "opts": [o.state_dict() for o in self.opts],
            "scalers": [s.state_dict() for s in self.scalers],
        }

    def load_state_dict(self, state: dict):
        for n, s in zip(self.nets, state["nets"]):
            n.load_state_dict(s)
        for o, s in zip(self.opts, state["opts"]):
            o.load_state_dict(s)
        for sc, s in zip(self.scalers, state["scalers"]):
            sc.load_state_dict(s)

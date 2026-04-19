"""
RAMT Loss Functions

Combined loss: MSE + Directional penalty
MSE alone optimizes magnitude accuracy.
Directional loss penalizes wrong-direction predictions.
In trading, direction matters more than magnitude.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalLoss(nn.Module):
    """
    Penalizes predictions with wrong sign vs actual return.

    For each sample:
      product = y_true × y_pred
      If same sign (correct direction): product > 0
      If opposite sign (wrong direction): product < 0

      loss = ReLU(-product)
      Correct direction → ReLU(negative) = 0 (no penalty)
      Wrong direction   → ReLU(positive) > 0 (penalty)

    Input:  y_pred (batch, 1), y_true (batch, 1)
    Output: scalar loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        product = y_true * y_pred
        loss = F.relu(-product)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined MSE + Directional Loss.

    total_loss = mse_loss + lambda_dir × directional_loss

    lambda_dir = 0.3:
      70% weight on magnitude (MSE)
      30% weight on direction

    Why this balance:
      Pure MSE: ignores direction completely
      Pure directional: ignores magnitude (bad for Sharpe)
      Combined: optimizes both simultaneously

    Input:  y_pred (batch, 1), y_true (batch, 1)
    Output: scalar loss, mse component, dir component
    """

    def __init__(self, lambda_dir=0.3):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.mse = nn.MSELoss()
        self.directional = DirectionalLoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        dir_loss = self.directional(y_pred, y_true)
        total = mse_loss + self.lambda_dir * dir_loss
        return total, mse_loss, dir_loss


class TournamentRankingLoss(nn.Module):
    """
    Cross-sectional pairwise margin ranking with magnitude weighting.

    For every pair (i, j) with y_true[i] > y_true[j]:
        pair_loss = ReLU(margin - (pred[i] - pred[j])) * (y_true[i] - y_true[j])

    The (y_true[i] - y_true[j]) weight makes the tournament focus on the
    high-stakes matchups (large alpha spreads) — the trades that actually
    drive portfolio P&L. This replaces the prior top-k vs bottom-k scheme
    which dropped ~99% of the pairs and left the middle of the cross-section
    with zero gradient signal (a root cause of the pessimism bias).

    **pair_mode** (default ``"random"``) avoids O(N²) work on large cross-sections:
    - ``"full"``: dense all-pairs loss (exact, slow for ~200 names).
    - ``"random"``: up to ``max_pairs`` uniform samples among valid (i, j) with y[i] > y[j].
    - ``"top_bottom"``: all pairs between top-``k`` and bottom-``k`` by y (fast, ranking-focused).

    Args:
        margin: minimum required gap between pred[i] and pred[j] in UNSCALED
                alpha units (e.g. 0.02 = 2% monthly-alpha gap). Set based on
                the natural spread of the cross-section, not the RobustScaler
                transformed target.
        max_pairs: cap for ``"random"`` mode; ignored for ``"full"`` / ``"top_bottom"``.
                   If ``None`` and mode is ``"random"``, falls back to ``"full"``.
        pair_mode: ``"full"`` | ``"random"`` | ``"top_bottom"``.
        top_bottom_k: k for ``"top_bottom"`` (same spirit as ``_margin_rank_loss``).

    Input:
        pred:   (N,) or (N, 1)  — model scores for N cross-sectional items
        y_true: (N,) or (N, 1)  — true alpha, ideally in native % units

    Output: scalar loss
    """

    def __init__(
        self,
        margin: float = 0.02,
        *,
        max_pairs: int | None = 500,
        pair_mode: str = "random",
        top_bottom_k: int = 10,
    ):
        super().__init__()
        self.margin = float(margin)
        self.max_pairs = max_pairs
        self.pair_mode = str(pair_mode)
        self.top_bottom_k = int(top_bottom_k)

    def _hinge_weighted_mean(
        self, p: torch.Tensor, y: torch.Tensor, i: torch.Tensor, j: torch.Tensor
    ) -> torch.Tensor:
        """Weighted mean of hinge * |y_i - y_j| over pairs (i, j), y[i] > y[j] assumed."""
        yd = y[i] - y[j]
        pd = p[i] - p[j]
        weight = yd.abs()
        hinge = F.relu(self.margin - pd)
        den = weight.sum() + 1e-8
        return (hinge * weight).sum() / den

    def _forward_full(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        N = y.shape[0]
        y_diff = y.unsqueeze(1) - y.unsqueeze(0)  # (N, N)
        p_diff = p.unsqueeze(1) - p.unsqueeze(0)  # (N, N)
        mask = (y_diff > 0).float()
        weight = y_diff.abs() * mask
        hinge = F.relu(self.margin - p_diff)
        den = weight.sum() + 1e-8
        return (hinge * weight).sum() / den

    def _forward_random(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        N = int(y.shape[0])
        cap = self.max_pairs
        if cap is None or cap <= 0:
            return self._forward_full(p, y)

        device = y.device
        dtype = y.dtype
        # Oversample — roughly half of random ordered pairs satisfy y[i] > y[j] for continuous y.
        M = min(cap * 6, max(cap * 2, N * N))
        i = torch.randint(0, N, (M,), device=device)
        j = torch.randint(0, N, (M,), device=device)
        valid = (i != j) & (y[i] > y[j])
        i, j = i[valid], j[valid]
        if i.numel() == 0:
            return torch.zeros((), device=device, dtype=dtype)
        if i.numel() > cap:
            sel = torch.randperm(i.numel(), device=device)[:cap]
            i, j = i[sel], j[sel]
        return self._hinge_weighted_mean(p, y, i, j)

    def _forward_top_bottom(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        N = int(y.shape[0])
        k = min(self.top_bottom_k, N // 2)
        if k < 1:
            return torch.zeros((), device=y.device, dtype=y.dtype)
        top_idx = torch.topk(y, k=k, largest=True).indices
        bot_idx = torch.topk(y, k=k, largest=False).indices
        i = top_idx.unsqueeze(1).expand(k, k).reshape(-1)
        j = bot_idx.unsqueeze(0).expand(k, k).reshape(-1)
        yd = y[i] - y[j]
        keep = yd > 0
        if not keep.any():
            return torch.zeros((), device=y.device, dtype=y.dtype)
        i, j = i[keep], j[keep]
        return self._hinge_weighted_mean(p, y, i, j)

    def forward(self, pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        p = pred.reshape(-1)
        y = y_true.reshape(-1)
        N = int(y.shape[0])
        if N < 2:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        mode = self.pair_mode
        if mode == "full":
            return self._forward_full(p, y)
        if mode == "random":
            if self.max_pairs is None:
                return self._forward_full(p, y)
            return self._forward_random(p, y)
        if mode == "top_bottom":
            return self._forward_top_bottom(p, y)
        raise ValueError(f"Unknown pair_mode: {mode!r} (use full, random, top_bottom)")

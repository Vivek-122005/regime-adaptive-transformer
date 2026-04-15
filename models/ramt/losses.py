"""
RAMT Loss Functions

Combined loss: MSE + Directional penalty
MSE alone optimizes magnitude accuracy.
Directional loss penalizes wrong-direction predictions.
In trading, direction matters more than magnitude.
"""

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

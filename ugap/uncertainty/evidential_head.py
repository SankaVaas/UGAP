"""
Evidential uncertainty head, following Deep Evidential Regression
(Amini et al., NeurIPS 2020) and adapted the way Trust3R (ICML 2026) applies
it to feed-forward 3D reconstruction.

Instead of predicting a single depth/error value, the head predicts the
parameters (gamma, nu, alpha, beta) of a Normal-Inverse-Gamma distribution
over the per-point reconstruction error. From these we get:

    predicted error (mean)      = gamma
    aleatoric uncertainty       = beta / (alpha - 1)
    epistemic uncertainty       = beta / (nu * (alpha - 1))

The point of using evidential regression rather than a plain confidence
score (like DUSt3R's native output) is that it separates "this point is
inherently ambiguous" (aleatoric -- e.g. a genuinely reflective surface)
from "the model hasn't seen enough views of this region" (epistemic --
exactly what the NBV planner should act on).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialUncertaintyHead(nn.Module):
    """
    Small MLP that maps per-point features to NIG distribution parameters.

    Input features (configurable, default = 6-D):
        - DUSt3R native confidence (1)
        - local point density in a small radius (1)
        - distance to nearest camera center (1)
        - surface normal variance in a local neighborhood (1)
        - RGB gradient magnitude at that pixel (1)
        - number of views that observed this point (1)
    """

    def __init__(self, in_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),  # gamma, nu, alpha, beta (raw)
        )

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (N, in_dim) per-point feature tensor.
        Returns:
            gamma, nu, alpha, beta: each (N,) tensors defining the NIG
            distribution over the predicted per-point reconstruction error.
        """
        raw = self.net(features)
        gamma, log_nu, log_alpha, log_beta = raw.unbind(dim=-1)

        nu = F.softplus(log_nu)
        alpha = F.softplus(log_alpha) + 1.0  # alpha > 1 required
        beta = F.softplus(log_beta)

        return gamma, nu, alpha, beta

    @staticmethod
    def uncertainty_from_params(nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        """Returns (aleatoric, epistemic) uncertainty tensors, each (N,)."""
        aleatoric = beta / (alpha - 1.0).clamp(min=1e-6)
        epistemic = beta / (nu * (alpha - 1.0).clamp(min=1e-6)).clamp(min=1e-6)
        return aleatoric, epistemic


def evidential_loss(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target_error: torch.Tensor,
    lambda_reg: float = 0.01,
):
    """
    Negative log-likelihood of the NIG model plus an evidence regularizer
    that penalizes high confidence (nu) on points with large residual error
    -- this is what pushes epistemic uncertainty to actually track error.

    target_error: (N,) ground-truth per-point reconstruction error, e.g.
        ||predicted_point - ground_truth_point|| from a dataset with known
        geometry (ClearGrasp / TOD / synthetic scenes), or a proxy signal
        such as multi-view photometric/geometric reprojection error when
        ground truth 3D isn't available.
    """
    twoBlambda = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log((target_error - gamma) ** 2 * nu + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    error = torch.abs(target_error - gamma)
    evidence = 2.0 * nu + alpha
    reg = error * evidence

    return (nll + lambda_reg * reg).mean()

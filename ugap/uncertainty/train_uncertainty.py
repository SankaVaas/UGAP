"""
Training script for the evidential uncertainty head. Meant to be called
from notebooks/02_uncertainty_head_training.ipynb, not run as a standalone
CLI (Colab has no persistent local disk to checkpoint against between
sessions -- the notebook handles saving to Drive).

Ground-truth signal for the target error:
    - On ClearGrasp/TOD: use the provided ground-truth depth to compute
      per-point reconstruction error directly.
    - On self-collected sim sequences: exact error is available from the
      simulator.
    - On real, unlabeled scenes: fall back to multi-view geometric
      reprojection error as a self-supervised proxy target.
"""

from typing import Iterable

import torch
from torch.utils.data import DataLoader

from ugap.uncertainty.evidential_head import EvidentialUncertaintyHead, evidential_loss


def train_one_epoch(
    model: EvidentialUncertaintyHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    lambda_reg: float = 0.01,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)        # (B, N, in_dim)
        target_error = batch["target_error"].to(device)  # (B, N)

        features = features.view(-1, features.shape[-1])
        target_error = target_error.view(-1)

        gamma, nu, alpha, beta = model(features)
        loss = evidential_loss(gamma, nu, alpha, beta, target_error, lambda_reg=lambda_reg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: EvidentialUncertaintyHead, loader: DataLoader, device: str = "cuda"):
    """
    Reports sparsification error / AUSE-style ranking metric: does the
    predicted uncertainty actually correlate with true reconstruction
    error? (This is the metric Trust3R uses -- reuse it so results are
    directly comparable.)
    """
    model.eval()
    all_epistemic, all_error = [], []

    for batch in loader:
        features = batch["features"].to(device).view(-1, batch["features"].shape[-1])
        target_error = batch["target_error"].to(device).view(-1)

        gamma, nu, alpha, beta = model(features)
        _, epistemic = model.uncertainty_from_params(nu, alpha, beta)

        all_epistemic.append(epistemic.cpu())
        all_error.append(target_error.cpu())

    epistemic = torch.cat(all_epistemic)
    error = torch.cat(all_error)

    # Spearman-style rank correlation between predicted uncertainty and
    # actual error -- higher is better (well-calibrated uncertainty).
    rank_u = epistemic.argsort().argsort().float()
    rank_e = error.argsort().argsort().float()
    corr = torch.corrcoef(torch.stack([rank_u, rank_e]))[0, 1].item()

    return {"uncertainty_error_rank_correlation": corr}


def run_training(
    model: EvidentialUncertaintyHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    lambda_reg: float = 0.01,
) -> Iterable[dict]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, lambda_reg)
        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)
        print(record)

    return history

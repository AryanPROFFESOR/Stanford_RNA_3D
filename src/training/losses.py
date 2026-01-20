"""
Phase 2.2 — Scientifically correct loss functions
Includes:
- Kabsch-aligned coordinate loss
- Pairwise distance loss
- Mask handling
"""

import torch


def kabsch_align(P, Q, mask):
    """
    Align P to Q using Kabsch algorithm.
    P: predicted coords (L,3)
    Q: true coords (L,3)
    mask: valid residues (L,)
    """
    P = P[mask]
    Q = Q[mask]

    P_centroid = P.mean(dim=0)
    Q_centroid = Q.mean(dim=0)

    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    C = torch.matmul(P_centered.T, Q_centered)
    V, S, W = torch.svd(C)

    d = torch.det(torch.matmul(W, V.T))
    D = torch.diag(torch.tensor([1, 1, d], device=P.device))

    U = torch.matmul(W, torch.matmul(D, V.T))
    P_aligned = torch.matmul(P_centered, U)

    return P_aligned, Q_centered


def coordinate_loss(pred, true, mask):
    P, Q = kabsch_align(pred, true, mask)
    return torch.mean((P - Q) ** 2)


def distance_loss(pred, true, mask):
    pred = pred[mask]
    true = true[mask]

    Dp = torch.cdist(pred, pred)
    Dt = torch.cdist(true, true)

    return torch.mean((Dp - Dt) ** 2)


def total_loss(pred, true, mask, w_coord=1.0, w_dist=0.3):
    return w_coord * coordinate_loss(pred, true, mask) + w_dist * distance_loss(pred, true, mask)

"""Flux and velocity laws for the Reflexive Coherence model."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from rc.field import CoherenceField
from rc.geometry import CoherenceTensor


def velocity(
    field: CoherenceField,
    g_inv: np.ndarray,
    K: Optional[CoherenceTensor] = None,  # noqa: ARG001
    *,
    grad_phi: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    grad_c: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity with dissipative and RC (gradient-following) parts.

    v = -alpha g^{-1} ∇Φ + beta g^{-1} ∇C

    Note: This returns the velocity; the caller is expected to form flux as
    J = C * v for continuity dynamics.
    """
    if grad_c is None:
        grad_c = field.grad()
    if grad_phi is None:
        grad_phi = np.gradient(field.C, field.dx, edge_order=2)

    Cx, Cy = grad_c
    px, py = grad_phi

    vx = (-alpha * (g_inv[..., 0, 0] * px + g_inv[..., 0, 1] * py) +
          beta * (g_inv[..., 0, 0] * Cx + g_inv[..., 0, 1] * Cy))
    vy = (-alpha * (g_inv[..., 1, 0] * px + g_inv[..., 1, 1] * py) +
          beta * (g_inv[..., 1, 0] * Cx + g_inv[..., 1, 1] * Cy))
    return vx, vy


def flux(field: CoherenceField, vel: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return coherence flux J = C v."""
    vx, vy = vel
    return field.C * vx, field.C * vy


def mobility_policy(
    field: CoherenceField,
    g_inv: np.ndarray,
    potential_grad: Tuple[np.ndarray, np.ndarray],
    mobility: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity from a mobility D(C) and potential gradient ∇Φ.

    v = - D(C) * g^{-1} ∇Φ
    """
    if mobility is None:
        mobility = lambda c: np.ones_like(c)  # default κ_C = 1

    D = mobility(field.C)
    px, py = potential_grad
    vx = -D * (g_inv[..., 0, 0] * px + g_inv[..., 0, 1] * py)
    vy = -D * (g_inv[..., 1, 0] * px + g_inv[..., 1, 1] * py)
    return vx, vy


def divergence(
    Jx: np.ndarray,
    Jy: np.ndarray,
    dx: float,
    metric: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """
    Compute divergence of a flux field. If metric is provided, use covariant form.

    metric: tuple (g, g_inv, sqrt_det_g)
    """
    if metric is None:
        _, dJx = np.gradient(Jx, dx, edge_order=2)
        dJy, _ = np.gradient(Jy, dx, edge_order=2)
        return dJx + dJy

    g, g_inv, sqrt_det_g = metric
    flux_x = sqrt_det_g * (g_inv[..., 0, 0] * Jx + g_inv[..., 0, 1] * Jy)
    flux_y = sqrt_det_g * (g_inv[..., 1, 0] * Jx + g_inv[..., 1, 1] * Jy)

    _, d_flux_x = np.gradient(flux_x, dx, edge_order=2)
    d_flux_y, _ = np.gradient(flux_y, dx, edge_order=2)

    divergence = d_flux_x + d_flux_y
    return divergence / sqrt_det_g

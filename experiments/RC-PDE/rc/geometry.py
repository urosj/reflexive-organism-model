"""Geometry and metric construction for the Reflexive Coherence model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from rc.field import CoherenceField


@dataclass
class CoherenceTensor:
    """2-D coherence tensor components."""

    Kxx: np.ndarray
    Kxy: np.ndarray
    Kyy: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return full tensor array with shape (ny, nx, 2, 2)."""
        return np.stack(
            [
                np.stack([self.Kxx, self.Kxy], axis=-1),
                np.stack([self.Kxy, self.Kyy], axis=-1),
            ],
            axis=-2,
        )


def compute_K(
    field: CoherenceField,
    lam: float = 1.0,
    xi: float = 0.5,
    zeta: float = 0.0,
) -> CoherenceTensor:
    """
    Assemble coherence tensor K = λ*C*g₀ + ξ ∇C⊗∇C (+ optional flux term).

    lam weights the density contribution; xi weights gradient-driven anisotropy.
    """
    C = field.C
    Cx, Cy = field.grad()

    ## Base Euclidean metric g0 = I
    Kxx = lam * C + xi * Cx * Cx
    Kxy = xi * Cx * Cy
    Kyy = lam * C + xi * Cy * Cy

    # ---
    #eps = 1e-8
    #grad2 = Cx**2 + Cy**2
    #norm2 = grad2 + eps

    ## Unit vector in direction of ∇C
    #ux = Cx / np.sqrt(norm2)
    #uy = Cy / np.sqrt(norm2)

    ## Orientation-only rank-1 tensor
    #K_orient_xx = ux * ux
    #K_orient_xy = ux * uy
    #K_orient_yy = uy * uy

    #Kxx = lam * C + xi * K_orient_xx
    #Kxy = xi * K_orient_xy
    #Kyy = lam * C + xi * K_orient_yy

    #---
    # After computing Kxx, Kxy, Kyy
    # Approximate eigenvalues of the 2x2 symmetric matrix
    trace = Kxx + Kyy
    det = Kxx * Kyy - Kxy * Kxy
    disc = np.clip(trace*trace / 4 - det, 0.0, None)
    lam1 = trace/2 + np.sqrt(disc)
    lam2 = trace/2 - np.sqrt(disc)

    # Clamp anisotropy ratio
    ratio_max = 10.0   # or whatever you like
    ratio = np.maximum(lam1, 1e-8) / np.maximum(lam2, 1e-8)
    mask = ratio > ratio_max

    # Where too anisotropic, rescale toward isotropy
    lam1_clamped = np.where(mask, lam2 * ratio_max, lam1)

    # Reconstruct K from clamped eigenvalues & eigenvectors (more code),
    # or simply blend toward lam*C*I when ratio is too large:
    alpha = np.clip((ratio - ratio_max) / ratio_max, 0.0, 1.0)
    Kxx = (1 - alpha) * Kxx + alpha * lam * C
    Kxy = (1 - alpha) * Kxy
    Kyy = (1 - alpha) * Kyy + alpha * lam * C


    if zeta != 0.0:  # pragma: no cover - reserved for future extensions
        Kxx = Kxx + zeta * 0.0
        Kxy = Kxy + zeta * 0.0
        Kyy = Kyy + zeta * 0.0

    return CoherenceTensor(Kxx=Kxx, Kxy=Kxy, Kyy=Kyy)


def metric_from_K(
    field: CoherenceField,
    K: CoherenceTensor,
    lam: float = 1.0,
    xi: float = 0.5,  # noqa: ARG001 - retained for interface compatibility
    c_ref: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build metric g directly from K with mild normalization and isotropic floor.

    g_raw = K / (λ * C_ref); g = g_raw + eps * I.
    """
    C = field.C
    C_ref = max(float(np.mean(C)), c_ref)
    denom = lam * C_ref

    g_xx = K.Kxx / denom + eps
    g_xy = K.Kxy / denom
    g_yy = K.Kyy / denom + eps

    g = np.stack(
        [
            np.stack([g_xx, g_xy], axis=-1),
            np.stack([g_xy, g_yy], axis=-1),
        ],
        axis=-2,
    )

    det = g_xx * g_yy - g_xy**2
    det_safe = np.maximum(det, 1e-12)
    sqrt_det = np.sqrt(det_safe)

    g_inv = np.empty_like(g)
    g_inv[..., 0, 0] = g_yy / det_safe
    g_inv[..., 1, 1] = g_xx / det_safe
    g_inv[..., 0, 1] = -g_xy / det_safe
    g_inv[..., 1, 0] = -g_xy / det_safe


    # set stabilizers for debuging
    #g_xx = 1.0
    #g_xy = 0.0
    #g_yy = 1.0
    #sqrt_det_g = 1.0
    #g_inv[..., 0, 0] = 1.0
    #g_inv[..., 0, 1] = 0.0
    #g_inv[..., 1, 0] = 0.0
    #g_inv[..., 1, 1] = 1.0

    return g, g_inv, sqrt_det


def laplace_beltrami(
    field: CoherenceField, g: np.ndarray, g_inv: np.ndarray, sqrt_det_g: np.ndarray
) -> np.ndarray:
    """
    Compute ∇·(√g g^{ij} ∂_j C) / √g using finite differences.

    Parameters
    ----------
    field:
        CoherenceField providing scalar C.
    g, g_inv, sqrt_det_g:
        Metric, its inverse, and sqrt(det(g)) arrays of shapes
        (ny, nx, 2, 2) and (ny, nx).
    """
    Cx, Cy = field.grad()

    flux_x = sqrt_det_g * (g_inv[..., 0, 0] * Cx + g_inv[..., 0, 1] * Cy)
    flux_y = sqrt_det_g * (g_inv[..., 1, 0] * Cx + g_inv[..., 1, 1] * Cy)

    _, d_flux_x = np.gradient(flux_x, field.dx, edge_order=2)
    d_flux_y, _ = np.gradient(flux_y, field.dx, edge_order=2)

    divergence = d_flux_x + d_flux_y
    return divergence / sqrt_det_g

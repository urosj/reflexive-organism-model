"""PDE utilities: variational derivative, RHS assembly, and time stepping.

Model summary:
    ∂_t C = -∇·(C v)
    v = -D(C) g^{-1} ∇Φ + beta * g^{-1} ∇C   (RC correction)
    Φ = -kappa Δ_g C + dV/dC + gamma_curv * trace(K)
Geometry comes from K(C, ∇C) via metric_from_K.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

from rc import flux
from rc.field import CoherenceField
from rc.flux import divergence, mobility_policy
from rc.geometry import compute_K, laplace_beltrami, metric_from_K


PotentialParams = Dict[str, float]


def double_well_potential(C: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """Double-well potential V = -a/2 C^2 + b/4 C^4."""
    return -0.5 * a * C**2 + 0.25 * b * C**4


def double_well_dV_dC(C: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """dV/dC for double-well potential."""
    return -a * C + b * C**3


def quadratic_potential(C: np.ndarray, k: float = 1.0, C0: float = 0.0) -> np.ndarray:
    """Single-well quadratic potential V = k/2 (C - C0)^2."""
    return 0.5 * k * (C - C0) ** 2


def quadratic_dV_dC(C: np.ndarray, k: float = 1.0, C0: float = 0.0) -> np.ndarray:
    """dV/dC for quadratic potential."""
    return k * (C - C0)


def plateau_potential(
    C: np.ndarray,
    C_low: float = 0.1,
    C_high: float = 0.5,
    k_low: float = 1.0,
    k_high: float = 1.0,
) -> np.ndarray:
    """Piecewise-flat potential with quadratic walls outside [C_low, C_high]."""
    V = np.zeros_like(C, dtype=float)
    below = C < C_low
    above = C > C_high
    V[below] = k_low * (C[below] - C_low) ** 2
    V[above] = k_high * (C[above] - C_high) ** 2
    return V


def plateau_dV_dC(
    C: np.ndarray,
    C_low: float = 0.1,
    C_high: float = 0.5,
    k_low: float = 1.0,
    k_high: float = 1.0,
) -> np.ndarray:
    """Derivative of plateau potential."""
    dV = np.zeros_like(C, dtype=float)
    below = C < C_low
    above = C > C_high
    dV[below] = 2 * k_low * (C[below] - C_low)
    dV[above] = 2 * k_high * (C[above] - C_high)
    return dV


def dV_dC(
    C: np.ndarray,
    potential: str = "double_well",
    params: Optional[PotentialParams] = None,
) -> np.ndarray:
    """Dispatch derivative of potential with parameters."""
    params = params or {}
    if potential == "double_well":
        return double_well_dV_dC(C, **{k: params[k] for k in ("a", "b") if k in params})
    if potential == "quadratic":
        return quadratic_dV_dC(C, **{k: params[k] for k in ("k", "C0") if k in params})
    if potential == "plateau":
        return plateau_dV_dC(
            C,
            **{k: params[k] for k in ("C_low", "C_high", "k_low", "k_high") if k in params},
        )
    raise ValueError(f"Unknown potential '{potential}'")


def phi(
    field: CoherenceField,
    g: np.ndarray,
    g_inv: np.ndarray,
    sqrt_det_g: np.ndarray,
    kappa: float = 1.0,
    potential: str = "double_well",
    potential_params: Optional[PotentialParams] = None,
    curv_scalar: Optional[np.ndarray] = None,
    gamma_curv: float = 0.0,
) -> np.ndarray:
#    """Compute variational derivative Φ_C = -κ Δ_g C + dV/dC (+ geometry term)."""
    lap = laplace_beltrami(field, g, g_inv, sqrt_det_g)
    phi_val = -kappa * lap + dV_dC(field.C, potential=potential, params=potential_params)
    if curv_scalar is not None and gamma_curv != 0.0:
        phi_val = phi_val + gamma_curv * curv_scalar
    return phi_val
#    lap = laplace_beltrami(field, g, g_inv, sqrt_det_g)
#    # phi_val = -kappa * lap
#    phi_val = -kappa * lap + dV_dC(field.C, potential=potential, params=potential_params)
#    return phi_val


def rhs(
    field: CoherenceField,
    lam: float = 1.0,
    xi: float = 0.5,
    zeta: float = 0.0,
    kappa: float = 1.0,
    potential: str = "double_well",
    potential_params: Optional[PotentialParams] = None,
    mobility: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 1.0,  # retained for API, scales dissipative part via mobility
    beta: float = 0.0,
    gamma_curv: float = 0.0,
) -> np.ndarray:
#    """
#    Assemble ∂_t C = - ∇·(C v), with
#        v = -D(C) g^{-1} ∇Φ + beta * g^{-1} ∇C.
#
#    The curvature term enters Φ via gamma_curv * trace(K).
#    """
    grad_c = field.grad()
    K = compute_K(field, lam=lam, xi=xi, zeta=zeta)
    g, g_inv, sqrt_det = metric_from_K(field, K, lam=lam, xi=xi)
    curv_scalar = K.Kxx + K.Kyy
    phi_field = phi(
        field,
        g=g,
        g_inv=g_inv,
        sqrt_det_g=sqrt_det,
        kappa=kappa,
        potential=potential,
        potential_params=potential_params,
        curv_scalar=curv_scalar,
        gamma_curv=gamma_curv,
    )

#    px, py = np.gradient(phi_field, field.dx, edge_order=2)
#
#    if mobility is None:
#        D_func = lambda c: np.full_like(c, kappa)  # noqa: E731
#    else:
#        D_func = mobility
#
#    # Dissipative velocity via mobility law
#    vx_diss, vy_diss = mobility_policy(field, g_inv, (px, py), mobility=D_func)
#    # vx_diss *= alpha
#    # vy_diss *= alpha
#    vx, vy = vx_diss, vy_diss
#
#    # Optional RC component following grad C
#    if beta != 0.0:
#        Cx, Cy = grad_c
#        vx += beta * (g_inv[..., 0, 0] * Cx + g_inv[..., 0, 1] * Cy)
#        vy += beta * (g_inv[..., 1, 0] * Cx + g_inv[..., 1, 1] * Cy)
#
#    Jx, Jy = field.C * vx, field.C * vy
#    return -divergence(Jx, Jy, dx=field.dx, metric=(g, g_inv, sqrt_det))

#    lap_euc = field.laplacian()   # Euclidean ∆C on the grid
#    return kappa * lap_euc

    px, py = np.gradient(phi_field, field.dx, edge_order=2)
    D_func = lambda c: np.full_like(c, kappa)
    #vx_diss, vy_diss = mobility_policy(field, g_inv, (px, py), mobility=D_func)
    #vx, vy = vx_diss, vy_diss
    #Jx, Jy = field.C * vx, field.C * vy
    #return -divergence(Jx, Jy, dx=field.dx, metric=(g, g_inv, sqrt_det))

    Jx, Jy = mobility_policy(field, g_inv, (px, py), mobility=D_func)
    dC = -divergence(Jx, Jy, dx=field.dx, metric=(g, g_inv, sqrt_det))
    return dC




def _laplacian_matrix_periodic(grid_shape: Tuple[int, int], dx: float):
    """Build sparse 2-D Laplacian with periodic boundaries."""
    ny, nx = grid_shape
    ex = np.ones(nx)
    ey = np.ones(ny)

    Tx = diags([ex, -2 * ex, ex], offsets=[-1, 0, 1], shape=(nx, nx))
    Ty = diags([ey, -2 * ey, ey], offsets=[-1, 0, 1], shape=(ny, ny))
    # Periodic wrap
    Tx = Tx + diags([ex, ex], offsets=[-(nx - 1), nx - 1], shape=(nx, nx))
    Ty = Ty + diags([ey, ey], offsets=[-(ny - 1), ny - 1], shape=(ny, ny))

    L = kron(identity(ny), Tx) + kron(Ty, identity(nx))
    return L / (dx**2)


def _coherence_renormalise(field: CoherenceField, target: float, tol: float = 1e-6):
    """Rescale field to preserve total coherence if drifted."""
    current = field.total_coherence()
    drift = abs(current - target)
    if drift > tol and current > 0:
        factor = target / current
        field.C *= factor


def _safe_renorm(field, target, tol=1e-6, max_factor=2.0):
    current = field.total_coherence()
    if current <= 0:
        return
    drift = abs(current - target)
    if drift <= tol:
        return

    factor = target / current

    # If renorm would multiply by too much, skip it (or log a warning)
    if factor < 1/max_factor or factor > max_factor:
        # print("Skipping renorm: factor too large", factor)
        return

    field.C *= factor

def step_euler(
    field: CoherenceField,
    dt: float,
    *,
    lam: float = 1.0,
    xi: float = 0.5,
    zeta: float = 0.0,
    kappa: float = 1.0,
    potential: str = "double_well",
    potential_params: Optional[PotentialParams] = None,
    mobility: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma_curv: float = 0.0,
    target_coherence: Optional[float] = None,
    renormalize_every: int = 1,
    step_index: int = 0,
) -> None:
    """Advance field by one explicit Euler step."""
    target = target_coherence if target_coherence is not None else field.total_coherence()
    dC_dt = rhs(
        field,
        lam=lam,
        xi=xi,
        zeta=zeta,
        kappa=kappa,
        potential=potential,
        potential_params=potential_params,
        mobility=mobility,
        alpha=alpha,
        beta=beta,
        gamma_curv=gamma_curv,
    )

    field.C = field.C + dt * dC_dt
    _C_min = 1e-6
    field.C = np.maximum(field.C, _C_min)
    field.clip_nonnegative()
    if renormalize_every <= 1 or (step_index + 1) % renormalize_every == 0:
        _coherence_renormalise(field, target)
        #_safe_renorm(field, target)        # tests fail, this was just an experiment

    #field.C = field.C + dt * dC_dt

    # DO NOT hard-clip to 0 here.
    # Maybe allow small negatives or soft-clamp:

    #C_min = -1e-3
    #C_cap = 1000
    #field.C = np.maximum(field.C, C_min)   # with C_min a small negative, like -1e-3

    #_coherence_renormalise(field, target)
    # Optionally: cap max(C) to something tied to the plateau, e.g. 2*C_high
    #field.C = np.minimum(field.C, C_cap)



def step(
    field: CoherenceField,
    dt: float,
    renormalize_every: int = 1,
    step_index: int = 0,
    **rhs_kwargs: Union[float, str, dict, Callable[[np.ndarray], np.ndarray]],
) -> None:
    """
    Convenience stepper: advance using rhs and renormalise coherence.

    Parameters are forwarded to rhs (lam, xi, kappa, etc.).
    """
    target = field.total_coherence()
    dC_dt = rhs(field, **rhs_kwargs)
    field.C = field.C + dt * dC_dt
    _C_min = 1e-6
    field.C = np.maximum(field.C, _C_min)
    field.clip_nonnegative()
    if renormalize_every <= 1 or (step_index + 1) % renormalize_every == 0:
        _coherence_renormalise(field, target)


def step_crank_nicolson(
    field: CoherenceField,
    dt: float,
    *,
    lam: float = 1.0,
    xi: float = 0.5,
    zeta: float = 0.0,
    kappa: float = 1.0,
    potential: str = "double_well",
    potential_params: Optional[PotentialParams] = None,
    mobility: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma_curv: float = 0.0,
    target_coherence: Optional[float] = None,
    renormalize_every: int = 1,
    step_index: int = 0,
) -> None:
    """
    Semi-implicit Crank–Nicolson treating Laplacian part implicitly (periodic).

    Uses a Euclidean Laplacian approximation for the implicit piece and keeps
    nonlinear/potential terms explicit.
    """
    ny, nx = field.grid_shape
    target = target_coherence if target_coherence is not None else field.total_coherence()

    # Decompose RHS into diffusive part (approximate) and remaining residual.
    lap_euc = field.laplacian()
    diff_part = kappa * lap_euc
    full_rhs = rhs(
        field,
        lam=lam,
        xi=xi,
        zeta=zeta,
        kappa=kappa,
        potential=potential,
        potential_params=potential_params,
        mobility=mobility,
        alpha=alpha,
        beta=beta,
        gamma_curv=gamma_curv,
    )
    residual = full_rhs - diff_part

    L = _laplacian_matrix_periodic(field.grid_shape, field.dx)
    I = identity(nx * ny, format="csc")

    c_flat = field.C.ravel()
    rhs_vec = c_flat + dt * (residual.ravel() + 0.5 * diff_part.ravel())
    A = I - (dt * kappa / 2.0) * L
    new_flat = spsolve(A, rhs_vec)
    field.C = new_flat.reshape(field.grid_shape)
    field.clip_nonnegative()
    if renormalize_every <= 1 or (step_index + 1) % renormalize_every == 0:
        _coherence_renormalise(field, target)

"""
RC-PDE v12 CUDA: GPU-accelerated version of simulation-v12.py

Essential changes from original:
1. PyTorch backend with CUDA support - all tensors on GPU device
2. Larger default grid (512x512, scalable to 1024+)
3. Increased identity budget (64 max identities, mass cap 50.0)
4. Pure torch.randperm for identity birth selection (CuPy removed)
5. Identity fields vectorized as single tensor [max_identities, Nx, Ny] - major scalability win
6. Laplacian fully vectorized via laplacian_3d using torch.roll across all identities at once
7. Delta-zeroing: zero only recently deactivated slots I_tensor[n_survived:old_n_active].zero_()
8. All PDE operations vectorized using torch.roll, torch.clamp_, tensor ops
9. CPU-GPU transfers minimized: only for matplotlib visualization & numpy.random
10. Same RK2/Heun integration, mass-conserving projection, metric updates
11. Precomputed index grids (Xg_idx, Yg_idx) for identity seeding - avoids rebuild per birth
12. Vectorized batch identity seeding - eliminates per-birth .item() loop
13. Visualization throttling (RENDER_INTERVAL, HEADLESS_MODE) to reduce host transfers
14. Reduced sync in update_identities: GPU-only predicates with single CPU transfer
15. Headless snapshot system: configurable storage, downsampling, and memory cap warnings
16. Offline animation from snapshots (memory) with full diagnostics
17. Streaming animation from disk with optional two-pass global sqrt_g scaling
"""

import os
import json
import torch
# CuPy removed - using pure PyTorch for identity birth selection (torch.randperm)
import numpy as np      # Still needed for some CPU operations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# GLOBAL PARAMETERS - Easily scalable for large grids
# ============================================================
Nx, Ny = 512, 512   # Default larger grid (can go to 1024+)
dx = dy = 0.1

# Coherence & potential
lam_pot     = 0.5
xi_grad     = 0.2
zeta_flux   = 0.001
kappa_grad  = 0.2
mobility    = 0.2
C_min, C_max = 0.0, 5.0

# Identity parameters (increased for larger grids)
max_identities        = 64
g_id                  = 0.35   # growth from coherence
d_id                  = 0.28   # decay
D_id                  = 0.12   # diffusion
id_birth_amp          = 0.6
id_birth_sigma        = 1.5
id_birth_interval     = 25
id_birth_per_step     = 2
id_min_mass           = 3e-3
eta_id                = 0.1   # identity -> geometry coupling
alpha_id              = 0.08   # identity -> coherence source

# New: birth conditions (v11)
spark_birth_sparks_min = 20     # need this many spark pixels to allow birth
I_global_mass_cap      = 50.0   # max total identity mass budget (increased)

# CFL / dt
cfl_safety  = 0.3
vel_max_cap = 3.0

# Spark detection
spark_rel_det_thresh  = 0.15
spark_rel_grad_thresh = 0.30

# Metric regularisation
G_ABS_MAX   = 10.0
DETG_MIN    = 1e-3
DETG_MAX    = 1e3
DETK_MIN    = 1e-6

eps = 1e-12

# Optional extra potential terms
USE_SPARK_DEEPENING = True
USE_IDENTITY_TILT   = True

# Visualization throttling (for performance on large grids)
RENDER_INTERVAL = 10  # Update visualization every N frames (1 = every frame)
HEADLESS_MODE = False  # Skip all matplotlib for pure compute runs
HEADLESS_STEPS = 2000  # Number of simulation steps when running in headless mode

# HEADLESS MODE SNAPSHOT & ANIMATION PARAMETERS
SNAPSHOT_INTERVAL = 10      # Store state every N steps (coarse: 10-50 for sync point reduction)
DOWNSAMPLE_FACTOR = None    # None=full fidelity, int>1=downsample for storage
STORAGE_MODE = 'memory'     # 'memory': keep in RAM | 'disk': stream to .npz files
SNAPSHOT_DIR = './snapshots'
MAX_SNAPSHOTS_MEMORY = 500   # Cap warning threshold for memory mode

# ============================================================
# BASIC HELPERS (CUDA-compatible)
# ============================================================
def compute_gradients(A):
    dAx = torch.empty_like(A)
    dAy = torch.empty_like(A)

    dAx[1:-1] = (A[2:] - A[:-2]) / (2*dx)
    dAy[:,1:-1] = (A[:,2:] - A[:, :-2]) / (2*dy)

    dAx[0]  = (A[1] - A[-1]) / (2*dx)
    dAx[-1] = (A[0] - A[-2]) / (2*dx)
    dAy[:,0]  = (A[:,1] - A[:,-1]) / (2*dy)
    dAy[:, -1] = (A[:,0] - A[:,-2]) / (2*dy)
    return dAx, dAy

def laplacian(A):
    """Compute Laplacian for 2D or 3D tensor. Rolls over spatial dimensions."""
    # For 3D [*, Nx, Ny], roll over dims 1 and 2 (spatial)
    ndim = A.dim()
    if ndim == 2:
        dim0, dim1 = 0, 1
    else:  # 3D
        dim0, dim1 = 1, 2
    
    return ((torch.roll(A, -1, dim0) - 2*A + torch.roll(A, 1, dim0))/dx**2 +
            (torch.roll(A, -1, dim1) - 2*A + torch.roll(A, 1, dim1))/dy**2)

def laplacian_3d(I):
    """Fully vectorized Laplacian for 3D tensor [n_id, Nx, Ny]."""
    return ((torch.roll(I, -1, 1) - 2*I + torch.roll(I, 1, 1))/dx**2 +
            (torch.roll(I, -1, 2) - 2*I + torch.roll(I, 1, 2))/dy**2)

def smooth_gaussian(A):
    w00 = 1/16; w01 = 2/16; w02 = 1/16
    A0 = A
    Axp = torch.roll(A,-1,0)
    Axm = torch.roll(A,1,0)
    Ayp = torch.roll(A,-1,1)
    Aym = torch.roll(A,1,1)
    Axpyp = torch.roll(Axp,-1,1)
    Axpym = torch.roll(Axp, 1,1)
    Axmyp = torch.roll(Axm,-1,1)
    Axmym = torch.roll(Axm, 1,1)
    return (w00*(Axmym+Axmyp+Axpym+Axpyp) +
            w01*(Axm+Axp+Aym+Ayp) +
            w02*A0)

# ============================================================
# METRIC & GEOMETRY
# ============================================================
def regularise_metric(g_xx, g_xy, g_yy):
    torch.clamp(g_xx, -G_ABS_MAX, G_ABS_MAX, out=g_xx)
    torch.clamp(g_xy, -G_ABS_MAX, G_ABS_MAX, out=g_xy)
    torch.clamp(g_yy, -G_ABS_MAX, G_ABS_MAX, out=g_yy)

    det_g = g_xx*g_yy - g_xy*g_xy
    bad = (~torch.isfinite(det_g)) | (det_g <= DETG_MIN)
    if torch.any(bad):
        g_xx[bad] = 1.0
        g_xy[bad] = 0.0
        g_yy[bad] = 1.0
        det_g = g_xx*g_yy - g_xy*g_xy

    big = det_g > DETG_MAX
    if torch.any(big):
        alpha = torch.sqrt(1.0 / (det_g[big] + eps))
        g_xx[big] *= alpha
        g_xy[big] *= alpha
        g_yy[big] *= alpha

    det_g = g_xx*g_yy - g_xy*g_xy
    det_g = torch.where(det_g <= DETG_MIN, DETG_MIN, det_g)
    return g_xx, g_xy, g_yy, det_g

def metric_det_and_inv(g_xx, g_xy, g_yy):
    g_xx, g_xy, g_yy, det_g = regularise_metric(g_xx, g_xy, g_yy)
    sqrt_g = torch.sqrt(det_g)
    inv_det = 1.0 / (det_g + eps)
    gxx_inv =  g_yy * inv_det
    gxy_inv = -g_xy * inv_det
    gyy_inv =  g_xx * inv_det
    return det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv

# ============================================================
# POTENTIAL & SPARKS
# ============================================================
def Vprime_base(C):
    return lam_pot * (C - C*C)

def Vprime_with_sparks_and_identity(C, spark_mask, I_sum):
    dV = Vprime_base(C)
    if USE_SPARK_DEEPENING:
        dV -= 0.05 * spark_mask
    if USE_IDENTITY_TILT and I_sum is not None:
        dV -= 0.02 * I_sum
    return dV

def compute_spark_mask(C, g_xx, g_xy, g_yy, step=None):
    C_s = smooth_gaussian(C)

    Cxx = (torch.roll(C_s,-1,0) - 2*C_s + torch.roll(C_s,1,0))/dx**2
    Cyy = (torch.roll(C_s,-1,1) - 2*C_s + torch.roll(C_s,1,1))/dy**2
    Cxy = (torch.roll(torch.roll(C_s,-1,0),-1,1)
         - torch.roll(torch.roll(C_s,-1,0), 1,1)
         - torch.roll(torch.roll(C_s, 1,0),-1,1)
         + torch.roll(torch.roll(C_s, 1,0), 1,1))/(4*dx*dy)

    detH = Cxx*Cyy - Cxy*Cxy
    abs_detH = torch.abs(detH)
    max_abs = torch.max(abs_detH)
    if max_abs < 1e-12:
        return torch.zeros_like(C)

    rel_det = abs_detH / (max_abs + eps)

    dC_dx, dC_dy = compute_gradients(C_s)
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)
    grad_up = gxx_inv*dC_dx + gxy_inv*dC_dy
    grad_vp = gxy_inv*dC_dx + gyy_inv*dC_dy
    grad_norm = grad_up**2 + grad_vp**2
    max_grad = torch.max(grad_norm)
    if max_grad < 1e-12:
        return torch.zeros_like(C)

    rel_grad = grad_norm / (max_grad + eps)
    spark_mask = ((rel_det < spark_rel_det_thresh) &
                  (rel_grad > spark_rel_grad_thresh)).to(torch.float32)

    if step is not None and step % 50 == 0:
        print(f"[SPARK] step={step}: {torch.count_nonzero(spark_mask)}")

    return spark_mask

# ============================================================
# IDENTITY FIELDS - VECTORIZED (3D tensor [n_id, Nx, Ny])
# ============================================================
def seed_identity_at(ix, iy):
    """Seed a new identity field at position (ix, iy). Returns 2D tensor."""
    # Use precomputed index grids Xg_idx/Yg_idx to match integer ix/iy from torch.where
    r2 = (Xg_idx - ix)**2 + (Yg_idx - iy)**2
    I_new = id_birth_amp * torch.exp(-r2 / (2 * (id_birth_sigma/dx)**2))
    return I_new

def seed_identities_at_batch(ix_coords, iy_coords):
    """Seed multiple identity fields at once using vectorized operations.
    
    Args:
        ix_coords: 1D tensor of x-indices (row indices in ij indexing)
        iy_coords: 1D tensor of y-indices (col indices in ij indexing)
    Returns:
        Tensor of shape [n_new, Nx, Ny] with identity fields for each seed
    """
    n_new = ix_coords.shape[0]
    # Broadcast: diff_x[i, j, k] = Xg_idx[j,k] - ix_coords[i]
    diff_x = Xg_idx.unsqueeze(0) - ix_coords.view(n_new, 1, 1)
    diff_y = Yg_idx.unsqueeze(0) - iy_coords.view(n_new, 1, 1)
    r2 = diff_x**2 + diff_y**2
    I_batch = id_birth_amp * torch.exp(-r2 / (2 * (id_birth_sigma/dx)**2))
    return I_batch

def update_identities(C, I_tensor, n_active, spark_mask, g_xx, g_xy, g_yy, dt, step):
    """Evolve identities with Heun RK2, apply collapse, and conditionally create new ones.
    
    Args:
        C: coherence field [Nx, Ny]
        I_tensor: identity fields tensor [max_identities, Nx, Ny] (padded for unused slots)
        n_active: number of active identities
        spark_mask: spark locations mask [Nx, Ny]
        g_xx, g_xy, g_yy: metric components [Nx, Ny]
        dt: timestep
        step: current step count
    
    Returns:
        I_tensor (updated in-place): identity fields tensor [max_identities, Nx, Ny]
        n_active: updated number of active identities
        I_sum: combined identity richness [Nx, Ny] or None if no active identities
    """
    # 1. Evolve & prune existing identities (Heun / RK2) - fully vectorized over all active ids
    if n_active > 0:
        # Get active identities slice [n_active, Nx, Ny]
        I_active = I_tensor[:n_active]
        
        # First RHS - fully vectorized Laplacian using laplacian_3d
        lapI = laplacian_3d(I_active)
        dIdt1 = g_id * C.unsqueeze(0) * I_active - d_id * I_active + D_id * lapI
        I_mid = I_active + 0.5 * dt * dIdt1

        # Midpoint RHS - fully vectorized
        lapI_mid = laplacian_3d(I_mid)
        dIdt2 = g_id * C.unsqueeze(0) * I_mid - d_id * I_mid + D_id * lapI_mid

        # Corrector - vectorized
        I_new = I_active + dt * dIdt2
        torch.clamp_(I_new, min=0.0)

        # Prune dead identities based on mass - GPU only predicates, single sync at end
        masses = torch.sum(I_new, dim=(1,2)) * dx * dy              # [n_active]
        alive_mask = masses >= id_min_mass                         # [n_active] bool
        n_survived_t = alive_mask.sum(dtype=torch.int32)           # 0-dim tensor
        
    else:
        n_survived_t = torch.tensor(0, dtype=torch.int32, device=device)

    # Birth gating - GPU only predicates, single sync at end
    num_sparks_t = (spark_mask > 0.5).sum(dtype=torch.int32)       # 0-dim tensor
    enough_sparks_t = num_sparks_t >= spark_birth_sparks_min
    
    total_I_mass_t = torch.tensor(0.0, device=device)
    if n_active > 0:
        # Use post-update/post-prune mass from I_new[alive_mask] to match original behavior
        surviving_mass = (I_new[alive_mask]).sum() * dx * dy
        total_I_mass_t = surviving_mass.detach()
    
    has_mass_budget_t = total_I_mass_t < I_global_mass_cap
    has_slots_t = n_survived_t < max_identities
    has_interval_t = torch.scalar_tensor(step % id_birth_interval == 0, device=device)
    
    # no Python 'and' chain; tensor reduction instead
    birth_allowed_t = torch.stack([
        has_interval_t, enough_sparks_t, has_mass_budget_t, has_slots_t
    ]).all()
    
    # Single host sync for both decisions (reduced from 4+ syncs to 1)
    decision = torch.stack([
        n_survived_t,
        birth_allowed_t.to(torch.int32),
    ])
    n_survived, birth_allowed = decision.cpu().tolist()           # ONE sync
    birth_allowed = bool(birth_allowed)

    # Apply prune results (after syncing n_survived)
    if n_active > 0:
        I_tensor[:n_survived] = I_new[alive_mask]
        # Delta-zeroing: zero only recently deactivated slots (vectorized for the range)
        old_n_active = n_active
        if n_survived < old_n_active:
            I_tensor[n_survived:old_n_active].zero_()
    
    n_active = n_survived  # Update active count after pruning

    # Note: num_sparks_t already synced via decision.cpu().tolist()
    if birth_allowed:
        # Get spark locations on GPU
        spark_locs = torch.where(spark_mask > 0.5)
        ix = spark_locs[0]  # row index -> X coordinate in indexing='ij'
        iy = spark_locs[1]  # col index -> Y coordinate in indexing='ij'
        num_candidates = len(ix)

        if num_candidates > 0:
            n_new = min(id_birth_per_step, max_identities - n_active)
            # Use torch.randperm for random selection without replacement (pure PyTorch)
            perm = torch.randperm(num_candidates, device=device)[:n_new]
            selected_ix = ix[perm]
            selected_iy = iy[perm]
            
            # Create new identities and insert into tensor (vectorized - no .item() calls)
            I_tensor[n_active:n_active+n_new] = seed_identities_at_batch(selected_ix, selected_iy)
            n_active += n_new

    # 3. Combined identity richness - sum over active identities
    if n_active == 0:
        I_sum = None
    else:
        I_sum = torch.sum(I_tensor[:n_active], dim=0)

    return I_tensor, n_active, I_sum

# ============================================================
# COHERENCE FUNCTIONAL + FLUX
# ============================================================
def delta_P_over_delta_C(C, g_xx, g_xy, g_yy, spark_mask, I_sum):
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)
    dC_dx, dC_dy = compute_gradients(C)

    dC_up = gxx_inv*dC_dx + gxy_inv*dC_dy
    dC_vp = gxy_inv*dC_dx + gyy_inv*dC_dy

    flux_x = sqrt_g * dC_up
    flux_y = sqrt_g * dC_vp

    div_x = (torch.roll(flux_x,-1,0) - torch.roll(flux_x,1,0))/(2*dx)
    div_y = (torch.roll(flux_y,-1,1) - torch.roll(flux_y,1,1))/(2*dy)

    laplace_B = (div_x + div_y) / (sqrt_g + eps)

    dV = Vprime_with_sparks_and_identity(C, spark_mask, I_sum)
    return -kappa_grad * laplace_B + dV

def compute_flux(C, g_xx, g_xy, g_yy, spark_mask, I_sum):
    phi = delta_P_over_delta_C(C, g_xx, g_xy, g_yy, spark_mask, I_sum)
    dphi_dx, dphi_dy = compute_gradients(phi)

    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)
    grad_phi_up = gxx_inv*dphi_dx + gxy_inv*dphi_dy
    grad_phi_vp = gxy_inv*dphi_dx + gyy_inv*dphi_dy

    v_up = -mobility * grad_phi_up
    v_vp = -mobility * grad_phi_vp
    torch.clamp_(v_up, min=-vel_max_cap, max=vel_max_cap)
    torch.clamp_(v_vp, min=-vel_max_cap, max=vel_max_cap)

    Jx = C * (g_xx*v_up + g_xy*v_vp)
    Jy = C * (g_xy*v_up + g_yy*v_vp)
    torch.clamp_(Jx, min=-10, max=10)
    torch.clamp_(Jy, min=-10, max=10)

    return Jx, Jy, v_up, v_vp, sqrt_g

def covariant_divergence(Jx, Jy, g_xx, g_xy, g_yy):
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)

    J_up = gxx_inv*Jx + gxy_inv*Jy
    J_vp = gxy_inv*Jx + gyy_inv*Jy

    flux_x = sqrt_g * J_up
    flux_y = sqrt_g * J_vp

    div_x = (torch.roll(flux_x,-1,0) - torch.roll(flux_x,1,0))/(2*dx)
    div_y = (torch.roll(flux_y,-1,1) - torch.roll(flux_y,1,1))/(2*dy)

    return -(div_x + div_y) / (sqrt_g + eps)

def update_metric_from_K(C, Jx, Jy, g_xx, g_xy, g_yy, I_sum):
    """Update metric with explicit ξ-gradient and η-identity terms."""
    dC_dx, dC_dy = compute_gradients(C)

    # ordinary gradient contribution (ξ term)
    grad_term_xx = xi_grad * dC_dx * dC_dx
    grad_term_xy = xi_grad * dC_dx * dC_dy
    grad_term_yy = xi_grad * dC_dy * dC_dy

    # identity-only curvature contribution (η term)
    if I_sum is not None:
        id_term_xx = eta_id * I_sum * dC_dx * dC_dx
        id_term_xy = eta_id * I_sum * dC_dx * dC_dy
        id_term_yy = eta_id * I_sum * dC_dy * dC_dy
    else:
        id_term_xx = 0.0
        id_term_xy = 0.0
        id_term_yy = 0.0

    K_xx = lam_pot*C*g_xx + grad_term_xx + id_term_xx + zeta_flux*Jx*Jx
    K_xy = lam_pot*C*g_xy + grad_term_xy + id_term_xy + zeta_flux*Jx*Jy
    K_yy = lam_pot*C*g_yy + grad_term_yy + id_term_yy + zeta_flux*Jy*Jy

    detK = K_xx*K_yy - K_xy*K_xy
    detK = torch.where(torch.abs(detK) < DETK_MIN, DETK_MIN, detK)
    inv_detK = 1.0 / detK

    g_new_xx =  K_yy * inv_detK
    g_new_xy = -K_xy * inv_detK
    g_new_yy =  K_xx * inv_detK

    torch.nan_to_num(g_new_xx, nan=1.0, out=g_new_xx)
    torch.nan_to_num(g_new_xy, nan=0.0, out=g_new_xy)
    torch.nan_to_num(g_new_yy, nan=1.0, out=g_new_yy)

    blend = 0.05
    g_xx = (1-blend)*g_xx + blend*g_new_xx
    g_xy = (1-blend)*g_xy + blend*g_new_xy
    g_yy = (1-blend)*g_yy + blend*g_new_yy

    g_xx, g_xy, g_yy, _ = regularise_metric(g_xx, g_xy, g_yy)
    return g_xx, g_xy, g_yy

# ============================================================
# COHERENCE RHS & STEP
# ============================================================
def rhs_C(C, g_xx, g_xy, g_yy, spark_mask, I_sum):
    Jx, Jy, v_up, v_vp, sqrt_g = compute_flux(C, g_xx, g_xy, g_yy, spark_mask, I_sum)
    dCdt_flux = covariant_divergence(Jx, Jy, g_xx, g_xy, g_yy)
    if I_sum is not None:
        dCdt = dCdt_flux + alpha_id * I_sum
    else:
        dCdt = dCdt_flux
    return dCdt, Jx, Jy, v_up, v_vp

def estimate_dt(v_up, v_vp):
    """Combined CFL: advection (coherence) + diffusion (identities)."""
    vmax = torch.max(torch.sqrt(v_up*v_up + v_vp*v_vp))
    if vmax < 1e-6:
        dt_adv = 1e-3
    else:
        dt_adv = cfl_safety * min(dx, dy) / (vmax.item() + eps)

    dt_diff = cfl_safety * min(dx**2, dy**2) / (4.0 * D_id + eps)
    return min(dt_adv, dt_diff)

def project_to_invariant(C, g_xx, g_xy, g_yy, target_mass):
    """
    Enforce ∫ C√|g| = target_mass while preserving spatial pattern
    as much as possible: correct by adding a small term ∝ C, rather
    than uniform rescaling.
    """
    det_g, sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
    current_mass = torch.sum(C * sqrt_g) * dx * dy
    delta_mass = target_mass - current_mass.item()

    if abs(delta_mass) > eps:
        # Weight proportional to existing C, with a tiny floor
        weight = torch.where(C > 0.0, C, torch.tensor(1e-12, device=device))
        weight_sum = torch.sum(weight * sqrt_g) * dx * dy
        correction = (delta_mass / (weight_sum.item() + eps)) * weight
        C = C + correction
        # Optionally keep C inside bounds
        torch.clamp_(C, min=C_min, max=C_max)

    return C

def rk2_step(C, g_xx, g_xy, g_yy, target_mass, spark_mask, I_sum):
    dCdt1, Jx1, Jy1, v1, w1 = rhs_C(C, g_xx, g_xy, g_yy, spark_mask, I_sum)
    dt = estimate_dt(v1, w1)

    C_tilde = C + dt*dCdt1
    torch.clamp_(C_tilde, min=C_min, max=C_max)

    dCdt2, Jx2, Jy2, v2, w2 = rhs_C(C_tilde, g_xx, g_xy, g_yy, spark_mask, I_sum)

    C_new = C + 0.5*dt*(dCdt1 + dCdt2)
    torch.clamp_(C_new, min=C_min, max=C_max)

    C_new = project_to_invariant(C_new, g_xx, g_xy, g_yy, target_mass)
    g_xx, g_xy, g_yy = update_metric_from_K(C_new, Jx2, Jy2, g_xx, g_xy, g_yy, I_sum)

    return C_new, g_xx, g_xy, g_yy, dt, v2, w2

# ============================================================
# INITIAL CONDITIONS
# ============================================================
def gaussian_blob(Xg, Yg, cx, cy, s):
    """Create a Gaussian blob centered at (cx, cy) with width s."""
    return torch.exp(-((Xg-cx)**2 + (Yg-cy)**2)/(2*s*s))

# Create meshgrid on device (both physical units and integer indices)
X = torch.arange(Nx, device=device) * dx
Y = torch.arange(Ny, device=device) * dy
Xg, Yg = torch.meshgrid(X, Y, indexing='ij')
# Integer index grids for identity seeding (use these in seed_identity_at to match ix/iy indices)
X_idx = torch.arange(Nx, device=device)
Y_idx = torch.arange(Ny, device=device)
Xg_idx, Yg_idx = torch.meshgrid(X_idx, Y_idx, indexing='ij')

# Initialize coherence field with Gaussian blobs + noise
C = torch.zeros(Nx, Ny, device=device)
C += gaussian_blob(Xg, Yg, X[Nx//3],   Y[Ny//2],   1.5)
C += gaussian_blob(Xg, Yg, X[2*Nx//3], Y[2*Ny//3], 2.0)
C += 0.05 * torch.randn(Nx, Ny, device=device)
torch.clamp_(C, min=C_min, max=C_max)

# Initialize metric (flat space initially)
g_xx = torch.ones_like(C)
g_xy = torch.zeros_like(C)
g_yy = torch.ones_like(C)

# Compute initial mass target for invariant projection
det_g0, sqrt_g0, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
target_mass0 = torch.sum(C * sqrt_g0) * dx * dy
target_mass0_value = float(target_mass0.item())
print(f"Initial coherence mass: {target_mass0_value:.4f}")

# Identity fields tensor [max_identities, Nx, Ny] - vectorized storage (padded with zeros)
I_tensor = torch.zeros(max_identities, Nx, Ny, device=device)
n_active = 0  # number of active identities
I_sum = None

# ============================================================
# HEADLESS MODE SNAPSHOT MANAGER
# ============================================================
class SnapshotManager:
    """Manage headless mode snapshots with memory or disk storage."""
    
    def __init__(self, storage_mode='memory', snapshot_dir='./snapshots'):
        self.storage_mode = storage_mode
        self.snapshot_dir = snapshot_dir
        self.warned_about_memory = False
        self.meta = {
            'Nx': Nx,
            'Ny': Ny,
            'dx': dx,
            'dy': dy,
            'C_min': C_min,
            'C_max': C_max,
            'lam_pot': lam_pot,
            'xi_grad': xi_grad,
            'zeta_flux': zeta_flux,
            'kappa_grad': kappa_grad,
            'mobility': mobility,
            'max_identities': max_identities,
            'g_id': g_id,
            'd_id': d_id,
            'D_id': D_id,
            'id_birth_amp': id_birth_amp,
            'id_birth_sigma': id_birth_sigma,
            'id_birth_interval': id_birth_interval,
            'id_birth_per_step': id_birth_per_step,
            'id_min_mass': id_min_mass,
            'eta_id': eta_id,
            'alpha_id': alpha_id,
            'spark_birth_sparks_min': spark_birth_sparks_min,
            'I_global_mass_cap': I_global_mass_cap,
            'spark_rel_det_thresh': spark_rel_det_thresh,
            'spark_rel_grad_thresh': spark_rel_grad_thresh,
            'cfl_safety': cfl_safety,
            'vel_max_cap': vel_max_cap,
        }
        
        if storage_mode == 'disk':
            os.makedirs(snapshot_dir, exist_ok=True)
        
        # Initialize storage containers
        self.steps = []
        if storage_mode == 'memory':
            self.C_snapshots = []
            self.sqrt_g_snapshots = []
            self.spark_snapshots = []  # bool type for clean semantics
            self.diagnostics = {}      # dict of lists: dt, mass, I_mass, ids
    
    def add_snapshot(self, step, C, sqrt_g, spark_mask,
                    dt_val=None, mass_val=None, I_mass_val=None, n_active_val=None):
        """Store a snapshot with optional diagnostics."""
        
        self.steps.append(step)
        ds = DOWNSAMPLE_FACTOR
        
        def maybe_downsample(tensor):
            if ds is not None and ds > 1:
                return tensor[::ds, ::ds].cpu().numpy()
            else:
                return tensor.cpu().numpy()
        
        C_np = maybe_downsample(C)
        sqrt_g_np = maybe_downsample(sqrt_g)
        
        # Store spark as bool for clean semantics (0 or 1, not 255)
        spark_bool = (spark_mask > 0.5).cpu().bool().numpy()
        if ds is not None and ds > 1:
            spark_bool = spark_bool[::ds, ::ds]
        
        if self.storage_mode == 'memory':
            # Memory mode: check cap warning
            n_snapshots = len(self.C_snapshots) + 1
            if n_snapshots >= MAX_SNAPSHOTS_MEMORY and not self.warned_about_memory:
                print(f"[WARNING] Snapshot count ({n_snapshots}) approaching memory limit. "
                      f"Consider switching to STORAGE_MODE='disk' for long runs.")
                self.warned_about_memory = True
            
            self.C_snapshots.append(C_np)
            self.sqrt_g_snapshots.append(sqrt_g_np)
            self.spark_snapshots.append(spark_bool)
            
            # Store diagnostics
            if dt_val is not None:
                self.diagnostics.setdefault('dt', []).append(dt_val)
            if mass_val is not None:
                self.diagnostics.setdefault('mass', []).append(mass_val)
            if I_mass_val is not None:
                self.diagnostics.setdefault('I_mass', []).append(I_mass_val)
            if n_active_val is not None:
                self.diagnostics.setdefault('ids', []).append(n_active_val)
        
        elif self.storage_mode == 'disk':
            if len(self.steps) == 1:
                meta_path = os.path.join(self.snapshot_dir, 'meta.json')
                if not os.path.exists(meta_path):
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(self.meta, f, indent=2)
            # Disk mode: stream to file immediately (no RAM accumulation)
            filename = f"{self.snapshot_dir}/snap_{step:06d}.npz"
            np.savez(filename,
                     C=C_np, sqrt_g=sqrt_g_np, spark=spark_bool,
                     step=np.int64(step) if hasattr(np, 'int64') else int(step),
                     dt=float(dt_val) if dt_val is not None else 0.0,
                     mass=float(mass_val) if mass_val is not None else 0.0,
                     I_mass=float(I_mass_val) if I_mass_val is not None else 0.0,
                     n_active=int(n_active_val) if n_active_val is not None else 0)
    
    def get_all(self):
        """Load all snapshots (for disk mode: loads into memory at end)."""
        
        if self.storage_mode == 'memory':
            return {
                'steps': self.steps,
                'C': self.C_snapshots,
                'sqrt_g': self.sqrt_g_snapshots,
                'spark': self.spark_snapshots,
                'diagnostics': self.diagnostics,
                'meta': self.meta
            }
        
        elif self.storage_mode == 'disk':
            # Load all .npz files and consolidate (WARNING: loads into RAM)
            files = sorted(f for f in os.listdir(self.snapshot_dir) if f.endswith('.npz'))
            
            steps, C_list, sqrt_g_list, spark_list = [], [], [], []
            diagnostics = {'dt': [], 'mass': [], 'I_mass': [], 'ids': []}
            
            for f in files:
                data = np.load(f"{self.snapshot_dir}/{f}")
                steps.append(int(data['step']))
                C_list.append(data['C'])
                sqrt_g_list.append(data['sqrt_g'])
                spark_list.append(data['spark'].astype(bool))
                diagnostics['dt'].append(float(data['dt']))
                diagnostics['mass'].append(float(data['mass']))
                diagnostics['I_mass'].append(float(data['I_mass']))
                diagnostics['ids'].append(int(data['n_active']))
            
            meta_path = os.path.join(self.snapshot_dir, 'meta.json')
            meta = None
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

            return {
                'steps': steps,
                'C': C_list,
                'sqrt_g': sqrt_g_list,
                'spark': spark_list,
                'diagnostics': diagnostics,
                'meta': meta
            }
    
    def cleanup(self):
        """Clean up disk files after animation creation."""
        import shutil
        if self.storage_mode == 'disk' and os.path.exists(self.snapshot_dir):
            shutil.rmtree(self.snapshot_dir)
            print(f"[CLEANUP] Removed snapshot directory: {self.snapshot_dir}")
    
    def has_data(self):
        """Check if any snapshots have been stored."""
        return len(self.steps) > 0

# ============================================================
# OFFLINE ANIMATION FROM SNAPSHOTS
# ============================================================
def create_animation_from_snapshots(snapshot_data):
    """Create full animation including diagnostic plots from saved data."""
    steps = snapshot_data.get('steps', [])
    C_snaps = snapshot_data.get('C', [])
    sqrt_g_snaps = snapshot_data.get('sqrt_g', [])
    spark_snaps = snapshot_data.get('spark', [])
    diag = snapshot_data.get('diagnostics', {})

    n_frames = len(C_snaps)
    if n_frames == 0:
        print("[WARNING] No snapshots available for animation.")
        return None

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1])

    # Main visualizations
    axC = fig.add_subplot(gs[0, 0])
    imC = axC.imshow(C_snaps[0].T, origin='lower', cmap='viridis',
                     vmin=C_min, vmax=C_max)
    axC.set_title("Coherence C")

    axG = fig.add_subplot(gs[0, 1])
    sg_min = min(s.min() for s in sqrt_g_snaps)
    sg_max = max(s.max() for s in sqrt_g_snaps)
    imG = axG.imshow(sqrt_g_snaps[0].T, origin='lower', cmap='magma',
                     vmin=sg_min, vmax=sg_max)
    axG.set_title("Geometry √|g|")

    axS = fig.add_subplot(gs[0, 2])
    imS = axS.imshow(spark_snaps[0].astype(np.float64).T, origin='lower', cmap='inferno',
                     vmin=0.0, vmax=1.0)
    axS.set_title("Sparks")

    # Diagnostic plots
    ax_dt = fig.add_subplot(gs[1, 0])
    line_dt, = ax_dt.plot([], [])
    ax_dt.set_ylabel("dt")

    ax_mass = fig.add_subplot(gs[1, 1])
    line_mass, = ax_mass.plot([], [], label='C mass')
    if diag.get('I_mass'):
        line_Imass, = ax_mass.plot([], [], label='I mass', color='orange')
    else:
        line_Imass = None
    ax_mass.legend()
    ax_mass.set_ylabel("mass")

    ax_ids = fig.add_subplot(gs[1, 2])
    line_ids, = ax_ids.plot([], [])
    ax_ids.set_ylabel("# identities")

    title = fig.suptitle("")

    def animate(i):
        imC.set_data(C_snaps[i].T)
        imG.set_data(sqrt_g_snaps[i].T)
        imS.set_data(spark_snaps[i].astype(np.float64).T)

        title.set_text(f"step={steps[i] if i < len(steps) else i}")

        n = i + 1
        line_dt.set_data(range(n), diag.get('dt', [])[:n])
        ax_dt.set_xlim(0, max(10, n))

        line_mass.set_data(range(n), diag.get('mass', [])[:n])
        if line_Imass is not None:
            line_Imass.set_data(range(n), diag.get('I_mass', [])[:n])
        ax_mass.set_xlim(0, max(10, n))

        line_ids.set_data(range(n), diag.get('ids', [])[:n])
        ax_ids.set_xlim(0, max(10, n))

        artists = [imC, imG, imS, title, line_dt, line_mass, line_ids]
        if line_Imass is not None:
            artists.append(line_Imass)
        return artists

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=False)
    plt.tight_layout()

    output_file = 'simulation_output.mp4'
    try:
        anim.save(output_file, writer='ffmpeg', fps=10)
        print(f"✓ Animation saved: {output_file}")
    except Exception as e:
        print(f"FFmpeg not available ({e}), saving as GIF...")
        anim.save('simulation_output.gif', writer='pillow', fps=10)
        print("✓ Animation saved: simulation_output.gif")

    plt.close(fig)
    return anim

def create_animation_streaming(snapshot_dir, fps=10, interval=100, two_pass=True):
    """Stream snapshots from disk without loading all into RAM."""
    files = sorted(f for f in os.listdir(snapshot_dir) if f.endswith('.npz'))
    if not files:
        print("[WARNING] No snapshots found for streaming animation.")
        return None

    # Optional two-pass for global sqrt_g range
    if two_pass:
        sg_min, sg_max = None, None
        for f in files:
            data = np.load(os.path.join(snapshot_dir, f), mmap_mode='r')
            sg = data['sqrt_g']
            vmin = sg.min()
            vmax = sg.max()
            sg_min = vmin if sg_min is None else min(sg_min, vmin)
            sg_max = vmax if sg_max is None else max(sg_max, vmax)
    else:
        sg_min = sg_max = None

    # Load first frame for initialization
    first = np.load(os.path.join(snapshot_dir, files[0]), mmap_mode='r')
    C0 = first['C']
    sqrt_g0 = first['sqrt_g']
    spark0 = first['spark'].astype(bool)

    if sg_min is None or sg_max is None:
        sg_min, sg_max = sqrt_g0.min(), sqrt_g0.max()

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1])

    axC = fig.add_subplot(gs[0, 0])
    imC = axC.imshow(C0.T, origin='lower', cmap='viridis', vmin=C_min, vmax=C_max)
    axC.set_title("Coherence C")

    axG = fig.add_subplot(gs[0, 1])
    imG = axG.imshow(sqrt_g0.T, origin='lower', cmap='magma', vmin=sg_min, vmax=sg_max)
    axG.set_title("Geometry √|g|")

    axS = fig.add_subplot(gs[0, 2])
    imS = axS.imshow(spark0.astype(np.float64).T, origin='lower', cmap='inferno', vmin=0.0, vmax=1.0)
    axS.set_title("Sparks")

    ax_dt = fig.add_subplot(gs[1, 0])
    line_dt, = ax_dt.plot([], [])
    ax_dt.set_ylabel("dt")

    ax_mass = fig.add_subplot(gs[1, 1])
    line_mass, = ax_mass.plot([], [], label='C mass')
    line_Imass, = ax_mass.plot([], [], label='I mass', color='orange')
    ax_mass.legend()
    ax_mass.set_ylabel("mass")

    ax_ids = fig.add_subplot(gs[1, 2])
    line_ids, = ax_ids.plot([], [])
    ax_ids.set_ylabel("# identities")

    title = fig.suptitle("")

    dt_vals, mass_vals, I_mass_vals, ids_vals, steps = [], [], [], [], []

    def animate(i):
        path = os.path.join(snapshot_dir, files[i])
        data = np.load(path, mmap_mode='r')

        C = data['C']
        sqrt_g = data['sqrt_g']
        spark = data['spark'].astype(bool)

        imC.set_data(C.T)
        imG.set_data(sqrt_g.T)
        imS.set_data(spark.astype(np.float64).T)

        step = int(data['step'])
        title.set_text(f"step={step}")

        steps.append(step)
        dt_vals.append(float(data['dt']))
        mass_vals.append(float(data['mass']))
        I_mass_vals.append(float(data['I_mass']))
        ids_vals.append(int(data['n_active']))

        n = len(steps)
        line_dt.set_data(range(n), dt_vals)
        ax_dt.set_xlim(0, max(10, n))

        line_mass.set_data(range(n), mass_vals)
        line_Imass.set_data(range(n), I_mass_vals)
        ax_mass.set_xlim(0, max(10, n))

        line_ids.set_data(range(n), ids_vals)
        ax_ids.set_xlim(0, max(10, n))

        return [imC, imG, imS, title, line_dt, line_mass, line_Imass, line_ids]

    anim = FuncAnimation(fig, animate, frames=len(files), interval=interval, blit=False)

    output_file = 'simulation_output.mp4'
    try:
        anim.save(output_file, writer='ffmpeg', fps=fps)
        print(f"✓ Animation saved: {output_file}")
    except Exception as e:
        print(f"FFmpeg not available ({e}), saving as GIF...")
        anim.save('simulation_output.gif', writer='pillow', fps=fps)
        print("✓ Animation saved: simulation_output.gif")

    plt.close(fig)
    return anim

# ============================================================
# VISUALISATION - only build if not headless
# ============================================================
if not HEADLESS_MODE:
    fig = plt.figure(figsize=(13,7))
    gs = fig.add_gridspec(2, 3, height_ratios=[6,1])

    axC = fig.add_subplot(gs[0,0])
    imC = axC.imshow(C.cpu().detach().T.numpy(), origin='lower', cmap='viridis',
                     vmin=C_min, vmax=C_max)
    axC.set_title("Coherence C")

    axG = fig.add_subplot(gs[0,1])
    imG = axG.imshow(sqrt_g0.cpu().detach().T.numpy(), origin='lower', cmap='magma')
    axG.set_title("Geometry √|g|")

    axS = fig.add_subplot(gs[0,2])
    imS = axS.imshow(torch.zeros_like(C).cpu().T.numpy(), origin='lower', cmap='inferno', vmin=0, vmax=1)
    axS.set_title("Sparks")

    # Quiver plot for velocity field
    step_skip = 8
    Xq, Yq = np.meshgrid(np.arange(0, Nx, step_skip), np.arange(0, Ny, step_skip), indexing='ij')
    quiv = axC.quiver(Xq, Yq, np.zeros_like(Xq), np.zeros_like(Xq))

    ax_dt = fig.add_subplot(gs[1,0])
    line_dt, = ax_dt.plot([],[])
    ax_dt.set_ylabel("dt")

    ax_mass = fig.add_subplot(gs[1,1])
    line_mass, = ax_mass.plot([],[])
    line_Imass, = ax_mass.plot([],[])
    ax_mass.set_ylabel("mass (C, I_tot)")

    ax_ids = fig.add_subplot(gs[1,2])
    line_ids, = ax_ids.plot([],[])
    ax_ids.set_ylabel("# identities")

    # Title for animation display
    title = fig.suptitle("")

else:
    # Headless mode placeholders (not used but needed for update function closure)
    fig = None
    imC = imG = imS = quiv = None
    line_dt = line_mass = line_Imass = line_ids = title = None
    ax_dt = ax_mass = ax_ids = None
    
    # Initialize snapshot manager for headless mode
    snap_mgr = SnapshotManager(storage_mode=STORAGE_MODE, snapshot_dir=SNAPSHOT_DIR)

# History for diagnostics (collected on sampled frames)
dt_hist = []
mass_hist = []
ids_hist = []
I_mass_hist = []

step_counter = 0

# ============================================================
# SIMULATION STEP
# ============================================================
def update(frame):
    global C, g_xx, g_xy, g_yy, I_tensor, n_active, I_sum, step_counter

    sampled_frame = (frame % RENDER_INTERVAL == 0)
    render_now = (not HEADLESS_MODE) and sampled_frame
    
    # Increment counter first, then compute snapshot trigger for headless mode
    step_counter += 1
    snapshot_frame = (step_counter % SNAPSHOT_INTERVAL == 0) if HEADLESS_MODE else False

    spark_mask = compute_spark_mask(C, g_xx, g_xy, g_yy, step=step_counter)

    C, g_xx, g_xy, g_yy, dt, vu, vv = rk2_step(
        C, g_xx, g_xy, g_yy, target_mass0_value, spark_mask, I_sum
    )

    I_tensor, n_active, I_sum = update_identities(C, I_tensor, n_active, spark_mask,
                                                   g_xx, g_xy, g_yy, dt, step_counter)

    det_g, sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
    mass = torch.sum(C * sqrt_g) * dx * dy

    # Identity diagnostics
    if I_sum is not None:
        I_mass = torch.sum(I_sum) * dx * dy
    else:
        I_mass = 0.0

    # Snapshot storage - immediate CPU transfer + optional disk stream
    if HEADLESS_MODE and snapshot_frame:
        snap_mgr.add_snapshot(
            step=step_counter,
            C=C, sqrt_g=sqrt_g, spark_mask=spark_mask,
            dt_val=dt, mass_val=mass.item(),
            I_mass_val=I_mass.item() if isinstance(I_mass, torch.Tensor) else I_mass,
            n_active_val=n_active
        )

    # Update diagnostics on sampled frames (rendering) or snapshot frames (headless)
    if (not HEADLESS_MODE and sampled_frame) or (HEADLESS_MODE and snapshot_frame):
        dt_hist.append(dt)
        mass_hist.append(mass.item())
        ids_hist.append(n_active)
        I_mass_hist.append(I_mass.item() if isinstance(I_mass, torch.Tensor) else I_mass)

    # Update visualizations only on render frames (move tensors to CPU for matplotlib)
    # Update visualizations only on render frames
    if render_now:
        imC.set_data(C.cpu().detach().T.numpy())
        
        imG.set_data(sqrt_g.cpu().detach().T.numpy())
        sqrt_min = sqrt_g.min().item()
        sqrt_max = sqrt_g.max().item()
        if sqrt_min < sqrt_max:
            imG.set_clim(sqrt_min, sqrt_max)
        
        imS.set_data(spark_mask.cpu().detach().T.numpy())

        # Update quiver with velocity field (only on render frames)
        vx = vu[::step_skip, ::step_skip].cpu().detach().numpy()
        vy = vv[::step_skip, ::step_skip].cpu().detach().numpy()
        quiv.set_UVC(vx, vy)

    n = len(dt_hist)

    # Update plot lines and title only on render frames
    if render_now:
        # dt plot updates
        line_dt.set_data(range(n), dt_hist)
        ax_dt.set_xlim(0, max(10, n))
        ax_dt.set_ylim(0, max(dt_hist)*1.1 if dt_hist else 1)

        # mass plots (C mass + identity total mass)
        line_mass.set_data(range(n), mass_hist)
        line_Imass.set_data(range(n), I_mass_hist)
        ax_mass.set_xlim(0, max(10, n))

        if mass_hist and I_mass_hist:
            y_min = min(mass_hist + I_mass_hist)
            y_max = max(mass_hist + I_mass_hist)
            ax_mass.set_ylim(y_min*0.99, y_max*1.01)

        # identities count plot
        line_ids.set_data(range(n), ids_hist)
        ax_ids.set_xlim(0, max(10, n))
        ax_ids.set_ylim(0, max(1, max(ids_hist)+1) if ids_hist else 1)

        title.set_text(
            f"step={step_counter}  dt={dt:.3e}  mass={mass.item():.4f}  "
            f"I_mass={I_mass.item() if isinstance(I_mass, torch.Tensor) else I_mass:.4f}  ids={n_active}"
        )

    return [imC, imG, imS, quiv, line_dt, line_mass, line_Imass, line_ids, title]

if not HEADLESS_MODE:
    anim = FuncAnimation(fig, update, interval=60, blit=False)
    plt.tight_layout()
    plt.show()
else:
    for frame in range(HEADLESS_STEPS):
        update(frame)

    if dt_hist:
        print(
            f"[HEADLESS DONE] steps={step_counter}  dt={dt_hist[-1]:.3e}  "
            f"mass={mass_hist[-1]:.4f}  I_mass={I_mass_hist[-1]:.4f}  ids={ids_hist[-1]}"
        )

    # Create animation from saved snapshots
    if snap_mgr.has_data():
        print(f"[ANIMATION] Processing {len(snap_mgr.steps)} snapshots...")
        if STORAGE_MODE == 'disk':
            create_animation_streaming(SNAPSHOT_DIR, two_pass=True)
        else:
            snapshot_data = snap_mgr.get_all()
            create_animation_from_snapshots(snapshot_data)
        snap_mgr.cleanup()

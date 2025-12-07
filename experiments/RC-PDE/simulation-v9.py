"""
===============================================================
RC–PDE v9: Reflexive Coherence with Spark-Driven Identity Injection
===============================================================

This version extends the v8 geometric RC–PDE by introducing a 
minimal but functional **identity injection mechanism**:

1. v8 PROBLEM:
   - Coherence C relaxes (gradient flow).
   - Geometry & sparks remain active.
   - Identity basins persist but weaken ("identity anaemia").
   - Sparks were diagnostic only: they did not modify C.

2. v9 SOLUTION:
   Sparks now feed back into coherence dynamics through:
   (A) Potential deepening:
         V'(C) → V'(C) – alpha_spark * spark_mask
   (B) Mass-neutral coherence redistribution:
         C ← C + gamma_spark * spark_mask * (C_mean – C)

   This:
   - strengthens basins at spark sites,
   - prevents identity anaemia,
   - allows new basin-like structures to form,
   - maintains full RC geometric reflexivity,
   - conserves total coherence mass exactly.

3. WHAT THIS ACHIEVES:
   - Identity basins remain alive.
   - Sparks now *do something* meaningful.
   - The PDE layer becomes truly reflexive.
   - Still numerically stable.

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# PARAMETERS
# ============================================================
Nx, Ny = 128, 128
dx = dy = 0.1

x = np.arange(Nx) * dx
y = np.arange(Ny) * dy

# Coherence & potential
lam_pot     = 0.5 # 0.8
xi_grad     = 0.2
zeta_flux   = 0.001 # 0.005
kappa_grad  = 0.2
mobility    = 0.2
C_min, C_max = 0.0, 5.0

# Identity injection params
# alpha_spark = 0.15     # deepens potential at spark sites
# gamma_spark = 0.02     # conservative redistribution toward sparks
alpha_spark = 0.05   # instead of 0.15
gamma_spark = 0.005  # instead of 0.02



# dt computation
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

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_gradients(A):
    dAx = np.empty_like(A)
    dAy = np.empty_like(A)

    dAx[1:-1] = (A[2:] - A[:-2]) / (2*dx)
    dAy[:,1:-1] = (A[:,2:] - A[:, :-2]) / (2*dy)

    # periodic BCs
    dAx[0]  = (A[1] - A[-1]) / (2*dx)
    dAx[-1] = (A[0] - A[-2]) / (2*dx)
    dAy[:,0]  = (A[:,1] - A[:,-1]) / (2*dy)
    dAy[:, -1] = (A[:,0] - A[:,-2]) / (2*dy)
    return dAx, dAy


def smooth_gaussian(A):
    """3×3 gaussian smoothing."""
    w00 = 1/16; w01 = 2/16; w02 = 1/16
    A0 = A
    Axp = np.roll(A, -1, 0); Axm = np.roll(A, 1, 0)
    Ayp = np.roll(A, -1, 1); Aym = np.roll(A, 1, 1)
    Axpyp = np.roll(Axp, -1, 1)
    Axpym = np.roll(Axp,  1, 1)
    Axmyp = np.roll(Axm, -1, 1)
    Axmym = np.roll(Axm,  1, 1)

    return (w00*(Axmym + Axmyp + Axpym + Axpyp)
            + w01*(Axm + Axp + Aym + Ayp)
            + w02*A0)


# ---------------------------
# METRIC REGULARISATION
# ---------------------------
def regularise_metric(g_xx, g_xy, g_yy):
    """Ensure metric remains finite, positive-definite."""
    np.clip(g_xx, -G_ABS_MAX, G_ABS_MAX, out=g_xx)
    np.clip(g_xy, -G_ABS_MAX, G_ABS_MAX, out=g_xy)
    np.clip(g_yy, -G_ABS_MAX, G_ABS_MAX, out=g_yy)

    det_g = g_xx*g_yy - g_xy*g_xy

    # Reset pathological regions to identity
    bad = (~np.isfinite(det_g)) | (det_g <= DETG_MIN)
    if np.any(bad):
        g_xx[bad] = 1.0
        g_xy[bad] = 0.0
        g_yy[bad] = 1.0
        det_g = g_xx*g_yy - g_xy*g_xy

    # Rescale if too large
    big = det_g > DETG_MAX
    if np.any(big):
        alpha = np.sqrt(1.0 / (det_g[big] + eps))
        g_xx[big] *= alpha
        g_xy[big] *= alpha
        g_yy[big] *= alpha

    det_g = g_xx*g_yy - g_xy*g_xy
    det_g = np.where(det_g <= DETG_MIN, DETG_MIN, det_g)
    return g_xx, g_xy, g_yy, det_g


def metric_det_and_inv(g_xx, g_xy, g_yy):
    g_xx, g_xy, g_yy, det_g = regularise_metric(g_xx, g_xy, g_yy)
    sqrt_g = np.sqrt(det_g)
    inv_det = 1.0 / (det_g + eps)
    gxx_inv =  g_yy * inv_det
    gxy_inv = -g_xy * inv_det
    gyy_inv =  g_xx * inv_det
    return det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv


# ============================================================
# POTENTIAL AND IDENTITY INJECTION
# ============================================================

def Vprime_base(C):
    """Baseline potential derivative."""
    return lam_pot * (C - C*C)

def Vprime_with_sparks(C, spark_mask):
    """Spark-deepened potential."""
    return Vprime_base(C) - alpha_spark * spark_mask

def conservative_injection(C, spark_mask, g_xx, g_xy, g_yy):
    """
    Mass-neutral coherence redistribution in the *geometric* measure:

        C_new = C + gamma_spark * spark_mask * (C_avg_spark - C)

    where C_avg_spark is the average of C over spark pixels,
    weighted by sqrt|g|. This guarantees

        ∑ C_new sqrt|g| = ∑ C sqrt|g|
    """
    # compute sqrt|g|
    det_g = g_xx*g_yy - g_xy*g_xy
    det_g = np.where(det_g <= DETG_MIN, DETG_MIN, det_g)
    sqrt_g = np.sqrt(det_g)

    w = spark_mask * sqrt_g
    total_w = np.sum(w)
    if total_w < 1e-12:
        return C  # no sparks → no injection

    C_avg = np.sum(C * w) / total_w

    C_new = C + gamma_spark * spark_mask * (C_avg - C)
    return C_new


# ============================================================
# SPARK DETECTION
# ============================================================

def compute_spark_mask(C, g_xx, g_xy, g_yy, step=None):
    C_s = smooth_gaussian(C)
    Cxx = (np.roll(C_s, -1, 0) - 2*C_s + np.roll(C_s, 1, 0)) / (dx*dx)
    Cyy = (np.roll(C_s, -1, 1) - 2*C_s + np.roll(C_s, 1, 1)) / (dy*dy)

    Cxy = (np.roll(np.roll(C_s, -1, 0), -1, 1)
         - np.roll(np.roll(C_s, -1, 0),  1, 1)
         - np.roll(np.roll(C_s,  1, 0), -1, 1)
         + np.roll(np.roll(C_s,  1, 0),  1, 1)) / (4*dx*dy)

    detH = Cxx*Cyy - Cxy*Cxy
    abs_det = np.abs(detH)
    max_abs = np.max(abs_det)
    if max_abs < 1e-12:
        return np.zeros_like(C)

    rel_det = abs_det / (max_abs + eps)

    # metric-weighted gradient magnitude
    dC_dx, dC_dy = compute_gradients(C_s)
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)
    grad_up = gxx_inv*dC_dx + gxy_inv*dC_dy
    grad_vp = gxy_inv*dC_dx + gyy_inv*dC_dy
    grad_norm = grad_up**2 + grad_vp**2

    max_grad = np.max(grad_norm)
    if max_grad < 1e-12:
        return np.zeros_like(C)

    rel_grad = grad_norm / (max_grad + eps)

    spark_mask = (rel_det < spark_rel_det_thresh) & (rel_grad > spark_rel_grad_thresh)
    spark_mask = spark_mask.astype(float)

    if step is not None and step % 50 == 0:
        print(f"[SPARK] step={step}: {np.count_nonzero(spark_mask)} points")

    return spark_mask


# ============================================================
# COHERENCE FUNCTIONAL
# ============================================================

def delta_P_over_delta_C(C, g_xx, g_xy, g_yy, spark_mask):
    """δP/δC with spark-modified potential."""
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)
    dC_dx, dC_dy = compute_gradients(C)

    dC_up = gxx_inv*dC_dx + gxy_inv*dC_dy
    dC_vp = gxy_inv*dC_dx + gyy_inv*dC_dy

    flux_x = sqrt_g * dC_up
    flux_y = sqrt_g * dC_vp

    div_x = (np.roll(flux_x, -1, 0) - np.roll(flux_x, 1, 0)) / (2*dx)
    div_y = (np.roll(flux_y, -1, 1) - np.roll(flux_y, 1, 1)) / (2*dy)

    laplace_B = (div_x + div_y) / (sqrt_g + eps)

    dV = Vprime_with_sparks(C, spark_mask)
    return -kappa_grad * laplace_B + dV


# ============================================================
# FLUX, DIV, METRIC UPDATE
# ============================================================

def compute_flux(C, g_xx, g_xy, g_yy, spark_mask):
    phi = delta_P_over_delta_C(C, g_xx, g_xy, g_yy, spark_mask)
    dphi_dx, dphi_dy = compute_gradients(phi)

    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)

    grad_phi_up = gxx_inv*dphi_dx + gxy_inv*dphi_dy
    grad_phi_vp = gxy_inv*dphi_dx + gyy_inv*dphi_dy

    v_up = -mobility * grad_phi_up
    v_vp = -mobility * grad_phi_vp
    v_up = np.clip(v_up, -vel_max_cap, vel_max_cap)
    v_vp = np.clip(v_vp, -vel_max_cap, vel_max_cap)

    Jx = C * (g_xx*v_up + g_xy*v_vp)
    Jy = C * (g_xy*v_up + g_yy*v_vp)
    np.clip(Jx, -10, 10, out=Jx)
    np.clip(Jy, -10, 10, out=Jy)

    return Jx, Jy, v_up, v_vp, sqrt_g


def covariant_divergence(Jx, Jy, g_xx, g_xy, g_yy):
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)

    J_up = gxx_inv*Jx + gxy_inv*Jy
    J_vp = gxy_inv*Jx + gyy_inv*Jy

    flux_x = sqrt_g * J_up
    flux_y = sqrt_g * J_vp

    div_x = (np.roll(flux_x, -1, 0) - np.roll(flux_x, 1, 0)) / (2*dx)
    div_y = (np.roll(flux_y, -1, 1) - np.roll(flux_y, 1, 1)) / (2*dy)

    return -(div_x + div_y) / (sqrt_g + eps)


def update_metric_from_K(C, Jx, Jy, g_xx, g_xy, g_yy):
    dC_dx, dC_dy = compute_gradients(C)

    K_xx = lam_pot*C*g_xx + xi_grad*dC_dx*dC_dx + zeta_flux*Jx*Jx
    K_xy = lam_pot*C*g_xy + xi_grad*dC_dx*dC_dy + zeta_flux*Jx*Jy
    K_yy = lam_pot*C*g_yy + xi_grad*dC_dy*dC_dy + zeta_flux*Jy*Jy

    detK = K_xx*K_yy - K_xy*K_xy
    detK = np.where(np.abs(detK) < DETK_MIN, DETK_MIN, detK)
    inv_detK = 1.0 / detK

    g_new_xx =  K_yy * inv_detK
    g_new_xy = -K_xy * inv_detK
    g_new_yy =  K_xx * inv_detK

    np.nan_to_num(g_new_xx, copy=False, nan=1.0)
    np.nan_to_num(g_new_xy, copy=False, nan=0.0)
    np.nan_to_num(g_new_yy, copy=False, nan=1.0)

    blend = 0.05 # 0.1
    g_xx = (1-blend)*g_xx + blend*g_new_xx
    g_xy = (1-blend)*g_xy + blend*g_new_xy
    g_yy = (1-blend)*g_yy + blend*g_new_yy

    g_xx, g_xy, g_yy, _ = regularise_metric(g_xx, g_xy, g_yy)
    return g_xx, g_xy, g_yy


# ============================================================
# FULL PDE STEP (RK2 + IDENTITY INJECTION)
# ============================================================

def rhs_C(C, g_xx, g_xy, g_yy, spark_mask):
    Jx, Jy, v_up, v_vp, _ = compute_flux(C, g_xx, g_xy, g_yy, spark_mask)
    dCdt = covariant_divergence(Jx, Jy, g_xx, g_xy, g_yy)
    return dCdt, Jx, Jy, v_up, v_vp


def estimate_dt(v_up, v_vp):
    vmax = np.max(np.sqrt(v_up*v_up + v_vp*v_vp))
    if vmax < 1e-6:
        return 1e-3
    return cfl_safety * min(dx, dy) / vmax


def project_to_invariant(C, g_xx, g_xy, g_yy, target_mass):
    det_g, sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
    mass = np.sum(C * sqrt_g) * dx * dy
    C *= target_mass / (mass + eps)
    return C


def rk2_step(C, g_xx, g_xy, g_yy, target_mass, spark_mask):
    dCdt1, Jx1, Jy1, v1, w1 = rhs_C(C, g_xx, g_xy, g_yy, spark_mask)
    dt = estimate_dt(v1, w1)

    C_tilde = C + dt*dCdt1
    np.clip(C_tilde, C_min, C_max, out=C_tilde)

    dCdt2, Jx2, Jy2, v2, w2 = rhs_C(C_tilde, g_xx, g_xy, g_yy, spark_mask)

    C_new = C + 0.5*dt*(dCdt1 + dCdt2)
    np.clip(C_new, C_min, C_max, out=C_new)

    # Identity injection
    C_new = conservative_injection(C_new, spark_mask, g_xx, g_xy, g_yy)

    # Projection (global coherence invariance)
    C_new = project_to_invariant(C_new, g_xx, g_xy, g_yy, target_mass)

    # Metric update
    g_xx, g_xy, g_yy = update_metric_from_K(C_new, Jx2, Jy2, g_xx, g_xy, g_yy)

    return C_new, g_xx, g_xy, g_yy, dt, v2, w2

# ============================================================
# INITIAL CONDITIONS
# ============================================================

def gaussian_blob(X, Y, cx, cy, s):
    return np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*s*s))

X, Y = np.meshgrid(x, y, indexing='ij')

C = np.zeros((Nx, Ny))
C += gaussian_blob(X, Y, x[Nx//3], y[Ny//2], 2.0)
#C += gaussian_blob(X, Y, x[Nx//2], y[Ny//3], 3.0)
C += gaussian_blob(X, Y, x[2*Nx//3], y[2*Ny//3], 4.0)
#C += 0.05*np.random.randn(Nx, Ny)
np.clip(C, C_min, C_max, out=C)

g_xx = np.ones_like(C)
g_xy = np.zeros_like(C)
g_yy = np.ones_like(C)

det_g0, sqrt_g0, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
target_mass0 = np.sum(C * sqrt_g0) * dx * dy
print("Initial total coherence:", target_mass0)


# ============================================================
# VISUALIZATION SETUP (unchanged from v8)
# ============================================================

fig = plt.figure(figsize=(13,7))
gs = fig.add_gridspec(2, 3, height_ratios=[6,1])

axC = fig.add_subplot(gs[0,0])
imC = axC.imshow(C.T, origin='lower', cmap='viridis')
axC.set_title("Coherence C")

axG = fig.add_subplot(gs[0,1])
imG = axG.imshow(sqrt_g0.T, origin='lower', cmap='magma')
axG.set_title("Geometry √|g|")

axS = fig.add_subplot(gs[0,2])
imS = axS.imshow(np.zeros_like(C).T, origin='lower', cmap='inferno', vmin=0, vmax=1)
axS.set_title("Sparks")

ax_dt = fig.add_subplot(gs[1,0])
line_dt, = ax_dt.plot([],[])
ax_dt.set_ylabel("dt")

ax_mass = fig.add_subplot(gs[1,1])
line_mass, = ax_mass.plot([],[])
ax_mass.set_ylabel("mass")

title = fig.suptitle("")

dt_hist = []
mass_hist = []

# For velocity quiver (optional)
step_skip = 8
Xq = x[::step_skip]
Yq = y[::step_skip]
qx, qy = np.meshgrid(Xq, Yq, indexing='ij')
quiv = axC.quiver(qx, qy, qx*0, qy*0, color='white')

# ============================================================
# ANIMATION LOOP
# ============================================================

step_counter = 0

def update(frame):
    global C, g_xx, g_xy, g_yy, step_counter

    step_counter += 1

    # compute spark mask (before step)
    spark_mask = compute_spark_mask(C, g_xx, g_xy, g_yy, step=step_counter)

    C, g_xx, g_xy, g_yy, dt, vu, vv = rk2_step(
        C, g_xx, g_xy, g_yy, target_mass0, spark_mask
    )

    det_g, sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
    imG.set_data(sqrt_g.T)
    imG.set_clim(sqrt_g.min(), sqrt_g.max())

    mass = np.sum(C * sqrt_g) * dx * dy

    # store diagnostics
    dt_hist.append(dt)
    mass_hist.append(mass)

    # update figures
    imC.set_data(C.T)
    imG.set_data(sqrt_g.T)
    imS.set_data(spark_mask.T)

    # quiver velocities
    vx = vu[::step_skip, ::step_skip]
    vy = vv[::step_skip, ::step_skip]
    quiv.set_UVC(vx, vy)

    # update dt plot
    n = len(dt_hist)
    line_dt.set_data(np.arange(n), dt_hist)
    ax_dt.set_xlim(0, max(10, n))
    ax_dt.set_ylim(0, max(dt_hist)*1.1)

    line_mass.set_data(np.arange(n), mass_hist)
    ax_mass.set_xlim(0, max(10, n))
    ax_mass.set_ylim(min(mass_hist)*0.99, max(mass_hist)*1.01)

    title.set_text(f"step={step_counter}   dt={dt:.3e}   mass={mass:.4f}")

    return [imC, imG, imS, quiv, line_dt, line_mass, title]


anim = FuncAnimation(fig, update, interval=50, blit=False)
plt.tight_layout()
plt.show()

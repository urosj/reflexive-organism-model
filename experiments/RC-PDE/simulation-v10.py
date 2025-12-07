"""
===============================================================
RC–PDE v10: Reflexive Coherence + Explicit Identity Dynamics
===============================================================

This script extends the v9 geometric RC–PDE in three key ways:

1. Identity Fields I_k(x,t)
   ------------------------
   We introduce an explicit *identity richness* field for each 
   active basin:

       I_k(x,t) >= 0,   k = 1..N(t)

   Each I_k obeys a simple PDE on the same grid:

       ∂t I_k = g_id * C * I_k  -  d_id * I_k  +  D_id ∆ I_k

   where
       g_id  ~ growth/reinforcement from coherence C,
       d_id  ~ decay, 
       D_id  ~ diffusion of identity.

   This means identity is *amplified* where coherence C is high,
   but decays and diffuses otherwise.

2. Spark → Identity Creation  (A_N → A_{N+1})
   -----------------------------------------
   The existing spark detector (Hessian degeneracy) now also 
   *creates new identity fields*. On each step:

   - find spark pixels (spark_mask == 1),
   - if we still have room (N < max_identities),
   - choose up to `id_birth_per_step` random spark locations,
   - seed a new identity field:

         I_new(x) = A_birth * exp(-|x - x_s|^2 / (2 σ_birth^2))

   This implements the RC statement:

       "The attractor set changes A_N → A_{N+1} at a spark."

3. Collapse → Identity Pruning  (A_N → A_{N-1})
   -------------------------------------------
   Each identity has a total "mass"

       M_k = ∫ I_k(x) dx dy.

   When M_k drops below a threshold `id_min_mass`, the identity is 
   removed from the list. This is a *collapse event* in the 
   identity layer: the basin no longer contributes meaningfully 
   to the dynamics.

4. Identity → Coherence + Geometry Feedback
   ----------------------------------------
   Identity fields feed back in two ways:

   (a) Additional source term in C:

       ∂t C = (RC–PDE flux term) + α_id * I_sum

   where I_sum = Σ_k I_k. This reinforces coherence near active 
   identities. Global invariance of C is still enforced by the 
   projection step, so identity feedback redistributes C rather 
   than creating it from nothing.

   (b) Additional curvature coupling in K:

       K_μν ← K_μν + η_id * I_sum * ∂_μ C ∂_ν C

   i.e. places with strong identity + coherence gradients get 
   extra curvature. This sharpens geometry around identities and 
   creates geodesic-like channels.

5. Conserved Quantities
   ---------------------
   - Total coherence mass ∫ C sqrt|g| dx dy is kept close to its 
     initial value by projection after each RK step.
   - Spark-based identity birth is local and small; decay and 
     collapse prevent unbounded proliferation.

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# GLOBAL PARAMETERS
# ============================================================
Nx, Ny = 128, 128
dx = dy = 0.1

x = np.arange(Nx) * dx
y = np.arange(Ny) * dy

# Coherence & potential
lam_pot     = 0.5
xi_grad     = 0.2
zeta_flux   = 0.001
kappa_grad  = 0.2
mobility    = 0.2
C_min, C_max = 0.0, 5.0

# Identity parameters
max_identities      = 50
g_id                = 0.4   # identity growth from coherence
d_id                = 0.2   # identity decay
D_id                = 0.1   # identity diffusion
id_birth_amp        = 0.6   # amplitude of new identity blob
id_birth_sigma      = 1.5   # radius of new identity blob
id_birth_interval   = 25    # steps between birth attempts
id_birth_per_step   = 1     # max new identities per step
id_min_mass         = 1e-3  # collapse threshold
eta_id              = 0.5   # identity → geometry coupling
alpha_id            = 0.08  # identity → coherence source

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

# ============================================================
# BASIC HELPERS
# ============================================================

def compute_gradients(A):
    dAx = np.empty_like(A)
    dAy = np.empty_like(A)

    dAx[1:-1] = (A[2:] - A[:-2]) / (2*dx)
    dAy[:,1:-1] = (A[:,2:] - A[:, :-2]) / (2*dy)

    dAx[0]  = (A[1] - A[-1]) / (2*dx)
    dAx[-1] = (A[0] - A[-2]) / (2*dx)
    dAy[:,0]  = (A[:,1] - A[:,-1]) / (2*dy)
    dAy[:, -1] = (A[:,0] - A[:,-2]) / (2*dy)
    return dAx, dAy

def laplacian(A):
    return ((np.roll(A,-1,0) - 2*A + np.roll(A,1,0))/dx**2 +
            (np.roll(A,-1,1) - 2*A + np.roll(A,1,1))/dy**2)

def smooth_gaussian(A):
    w00 = 1/16; w01 = 2/16; w02 = 1/16
    A0 = A
    Axp = np.roll(A,-1,0); Axm = np.roll(A,1,0)
    Ayp = np.roll(A,-1,1); Aym = np.roll(A,1,1)
    Axpyp = np.roll(Axp,-1,1)
    Axpym = np.roll(Axp, 1,1)
    Axmyp = np.roll(Axm,-1,1)
    Axmym = np.roll(Axm, 1,1)
    return (w00*(Axmym+Axmyp+Axpym+Axpyp) +
            w01*(Axm+Axp+Aym+Ayp) +
            w02*A0)

# ============================================================
# METRIC & GEOMETRY
# ============================================================

def regularise_metric(g_xx, g_xy, g_yy):
    np.clip(g_xx, -G_ABS_MAX, G_ABS_MAX, out=g_xx)
    np.clip(g_xy, -G_ABS_MAX, G_ABS_MAX, out=g_xy)
    np.clip(g_yy, -G_ABS_MAX, G_ABS_MAX, out=g_yy)

    det_g = g_xx*g_yy - g_xy*g_xy
    bad = (~np.isfinite(det_g)) | (det_g <= DETG_MIN)
    if np.any(bad):
        g_xx[bad] = 1.0
        g_xy[bad] = 0.0
        g_yy[bad] = 1.0
        det_g = g_xx*g_yy - g_xy*g_xy

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
# POTENTIAL & SPARKS
# ============================================================

def Vprime_base(C):
    return lam_pot * (C - C*C)

def Vprime_with_sparks_and_identity(C, spark_mask, I_sum):
    """Baseline + spark deepening + very small identity-induced tilt."""
    dV = Vprime_base(C) - 0.05 * spark_mask
    if I_sum is not None:
        dV -= 0.02 * I_sum
    return dV

def compute_spark_mask(C, g_xx, g_xy, g_yy, step=None):
    C_s = smooth_gaussian(C)

    Cxx = (np.roll(C_s,-1,0) - 2*C_s + np.roll(C_s,1,0))/dx**2
    Cyy = (np.roll(C_s,-1,1) - 2*C_s + np.roll(C_s,1,1))/dy**2
    Cxy = (np.roll(np.roll(C_s,-1,0),-1,1)
         - np.roll(np.roll(C_s,-1,0), 1,1)
         - np.roll(np.roll(C_s, 1,0),-1,1)
         + np.roll(np.roll(C_s, 1,0), 1,1))/(4*dx*dy)

    detH = Cxx*Cyy - Cxy*Cxy
    abs_detH = np.abs(detH)
    max_abs = np.max(abs_detH)
    if max_abs < 1e-12:
        return np.zeros_like(C)

    rel_det = abs_detH / (max_abs + eps)

    dC_dx, dC_dy = compute_gradients(C_s)
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)
    grad_up = gxx_inv*dC_dx + gxy_inv*dC_dy
    grad_vp = gxy_inv*dC_dx + gyy_inv*dC_dy
    grad_norm = grad_up**2 + grad_vp**2
    max_grad = np.max(grad_norm)
    if max_grad < 1e-12:
        return np.zeros_like(C)

    rel_grad = grad_norm / (max_grad + eps)
    spark_mask = ((rel_det < spark_rel_det_thresh) &
                  (rel_grad > spark_rel_grad_thresh)).astype(float)

    if step is not None and step % 50 == 0:
        print(f"[SPARK] step={step}: {np.count_nonzero(spark_mask)}")

    return spark_mask

# ============================================================
# IDENTITY FIELDS
# ============================================================

def seed_identity_at(I_fields, ix, iy, Nx, Ny):
    """Create a new identity centred at (ix,iy)."""
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    r2 = (X - ix)**2 + (Y - iy)**2
    I_new = id_birth_amp * np.exp(-r2 / (2 * (id_birth_sigma/dx)**2))
    I_fields.append(I_new)

def update_identities(C, I_fields, spark_mask, g_xx, g_xy, g_yy, dt, step):
    """Evolve existing identity fields, create new ones, prune dead ones."""
    # 1. evolve existing identities
    new_I_fields = []
    for I in I_fields:
        lapI = laplacian(I)
        dIdt = g_id * C * I - d_id * I + D_id * lapI
        I_new = I + dt * dIdt
        np.clip(I_new, 0.0, None, out=I_new)

        mass_I = np.sum(I_new) * dx * dy
        if mass_I >= id_min_mass:
            new_I_fields.append(I_new)  # keep
        # else: collapse event (discard)

    I_fields[:] = new_I_fields

    # 2. birth of new identities at spark points
    if step % id_birth_interval == 0 and np.count_nonzero(spark_mask) > 0:
        if len(I_fields) < max_identities:
            ys, xs = np.where(spark_mask > 0.5)
            num_candidates = len(xs)
            if num_candidates > 0:
                n_new = min(id_birth_per_step, max_identities - len(I_fields))
                idx = np.random.choice(num_candidates, size=n_new, replace=False)
                for k in idx:
                    ix, iy = xs[k], ys[k]
                    seed_identity_at(I_fields, ix, iy, Nx, Ny)

    # 3. combined identity richness
    if len(I_fields) == 0:
        I_sum = None
    else:
        I_sum = np.zeros_like(C)
        for I in I_fields:
            I_sum += I
    return I_fields, I_sum

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

    div_x = (np.roll(flux_x,-1,0) - np.roll(flux_x,1,0))/(2*dx)
    div_y = (np.roll(flux_y,-1,1) - np.roll(flux_y,1,1))/(2*dy)

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

    div_x = (np.roll(flux_x,-1,0) - np.roll(flux_x,1,0))/(2*dx)
    div_y = (np.roll(flux_y,-1,1) - np.roll(flux_y,1,1))/(2*dy)

    return -(div_x + div_y) / (sqrt_g + eps)

def update_metric_from_K(C, Jx, Jy, g_xx, g_xy, g_yy, I_sum):
    dC_dx, dC_dy = compute_gradients(C)

    I_factor = 1.0
    if I_sum is not None:
        I_factor += eta_id * I_sum

    K_xx = lam_pot*C*g_xx + xi_grad*I_factor*dC_dx*dC_dx + zeta_flux*Jx*Jx
    K_xy = lam_pot*C*g_xy + xi_grad*I_factor*dC_dx*dC_dy + zeta_flux*Jx*Jy
    K_yy = lam_pot*C*g_yy + xi_grad*I_factor*dC_dy*dC_dy + zeta_flux*Jy*Jy

    detK = K_xx*K_yy - K_xy*K_xy
    detK = np.where(np.abs(detK) < DETK_MIN, DETK_MIN, detK)
    inv_detK = 1.0 / detK

    g_new_xx =  K_yy * inv_detK
    g_new_xy = -K_xy * inv_detK
    g_new_yy =  K_xx * inv_detK

    np.nan_to_num(g_new_xx, copy=False, nan=1.0)
    np.nan_to_num(g_new_xy, copy=False, nan=0.0)
    np.nan_to_num(g_new_yy, copy=False, nan=1.0)

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
    vmax = np.max(np.sqrt(v_up*v_up + v_vp*v_vp))
    if vmax < 1e-6:
        return 1e-3
    return cfl_safety * min(dx, dy) / vmax

def project_to_invariant(C, g_xx, g_xy, g_yy, target_mass):
    det_g, sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
    mass = np.sum(C * sqrt_g) * dx * dy
    C *= target_mass / (mass + eps)
    return C

def rk2_step(C, g_xx, g_xy, g_yy, target_mass, spark_mask, I_sum):
    dCdt1, Jx1, Jy1, v1, w1 = rhs_C(C, g_xx, g_xy, g_yy, spark_mask, I_sum)
    dt = estimate_dt(v1, w1)

    C_tilde = C + dt*dCdt1
    np.clip(C_tilde, C_min, C_max, out=C_tilde)

    dCdt2, Jx2, Jy2, v2, w2 = rhs_C(C_tilde, g_xx, g_xy, g_yy, spark_mask, I_sum)

    C_new = C + 0.5*dt*(dCdt1 + dCdt2)
    np.clip(C_new, C_min, C_max, out=C_new)

    C_new = project_to_invariant(C_new, g_xx, g_xy, g_yy, target_mass)
    g_xx, g_xy, g_yy = update_metric_from_K(C_new, Jx2, Jy2, g_xx, g_xy, g_yy, I_sum)

    return C_new, g_xx, g_xy, g_yy, dt, v2, w2

# ============================================================
# INITIAL CONDITIONS
# ============================================================

def gaussian_blob(X, Y, cx, cy, s):
    return np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*s*s))

Xg, Yg = np.meshgrid(x, y, indexing='ij')

C = np.zeros((Nx, Ny))
C += gaussian_blob(Xg, Yg, x[Nx//3],     y[Ny//2],   1.5)
#C += gaussian_blob(Xg, Yg, x[Nx//2],     y[Ny//3],   3.5)
C += gaussian_blob(Xg, Yg, x[2*Nx//3],   y[2*Ny//3], 1.5)
#C += 0.05*np.random.randn(Nx, Ny)
np.clip(C, C_min, C_max, out=C)

g_xx = np.ones_like(C)
g_xy = np.zeros_like(C)
g_yy = np.ones_like(C)

det_g0, sqrt_g0, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
target_mass0 = np.sum(C * sqrt_g0) * dx * dy
print("Initial coherence mass:", target_mass0)

# Identity fields list
I_fields = []
I_sum = None

# ============================================================
# VISUALISATION
# ============================================================

fig = plt.figure(figsize=(13,7))
gs = fig.add_gridspec(2, 3, height_ratios=[6,1])

axC = fig.add_subplot(gs[0,0])
imC = axC.imshow(C.T, origin='lower', cmap='viridis',
                 vmin=C_min, vmax=C_max)
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

ax_ids = fig.add_subplot(gs[1,2])
line_ids, = ax_ids.plot([],[])
ax_ids.set_ylabel("# identities")

title = fig.suptitle("")

dt_hist = []
mass_hist = []
ids_hist = []

step_counter = 0

# quiver
step_skip = 8
Xq, Yq = np.meshgrid(x[::step_skip], y[::step_skip], indexing='ij')
quiv = axC.quiver(Xq, Yq, Xq*0, Yq*0, color='white', scale=5)

# ============================================================
# ANIMATION LOOP
# ============================================================

def update(frame):
    global C, g_xx, g_xy, g_yy, I_fields, I_sum, step_counter

    step_counter += 1

    # 1. compute sparks
    spark_mask = compute_spark_mask(C, g_xx, g_xy, g_yy, step=step_counter)

    # 2. coherence step
    C, g_xx, g_xy, g_yy, dt, vu, vv = rk2_step(
        C, g_xx, g_xy, g_yy, target_mass0, spark_mask, I_sum
    )

    # 3. update identities (using dt from coherence step)
    I_fields, I_sum = update_identities(C, I_fields, spark_mask,
                                        g_xx, g_xy, g_yy, dt, step_counter)

    det_g, sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
    mass = np.sum(C * sqrt_g) * dx * dy

    # diagnostics
    dt_hist.append(dt)
    mass_hist.append(mass)
    ids_hist.append(len(I_fields))

    # update images
    imC.set_data(C.T)

    imG.set_data(sqrt_g.T)
    imG.set_clim(sqrt_g.min(), sqrt_g.max())

    imS.set_data(spark_mask.T)

    # quiver velocities
    vx = vu[::step_skip, ::step_skip]
    vy = vv[::step_skip, ::step_skip]
    quiv.set_UVC(vx, vy)

    # dt plot
    n = len(dt_hist)
    line_dt.set_data(np.arange(n), dt_hist)
    ax_dt.set_xlim(0, max(10, n))
    ax_dt.set_ylim(0, max(dt_hist)*1.1)

    # mass plot
    line_mass.set_data(np.arange(n), mass_hist)
    ax_mass.set_xlim(0, max(10, n))
    ax_mass.set_ylim(min(mass_hist)*0.99, max(mass_hist)*1.01)

    # identity count plot
    line_ids.set_data(np.arange(n), ids_hist)
    ax_ids.set_xlim(0, max(10, n))
    ax_ids.set_ylim(0, max(1, max(ids_hist)+1))

    title.set_text(f"step={step_counter}  dt={dt:.3e}  mass={mass:.4f}  ids={len(I_fields)}")

    return [imC, imG, imS, quiv, line_dt, line_mass, line_ids, title]

anim = FuncAnimation(fig, update, interval=60, blit=False)
plt.tight_layout()
plt.show()

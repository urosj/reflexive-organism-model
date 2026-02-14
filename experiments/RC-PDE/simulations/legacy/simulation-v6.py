import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# the structural points:
# full metric + inversion from a tensorial K,
# covariant continuity with √|g|,
# flux from a functional derivative,
# reserve (flux) contribution in K,
# geometry-aware sparks via metric gradient norm,
# and exact global coherence conservation via projection.
# ============================================================

# ============================================================
# RC-inspired PDE v8
#
# Fields:
#   C(x,y,t)          - coherence density
#   g_xx,g_xy,g_yy    - metric tensor components
#
# PDE structure:
#   ∂t C + (1/√|g|) ∂_μ(√|g| J^μ) = 0
#
#   J^μ from gradient flow of P[C,g]:
#     P[C] = ∫ [ (kappa_grad/2) g^{μν} ∂_μC ∂_νC - V(C) ] √|g| dx
#
#   δP/δC ≈ -kappa_grad Δ_g C + V'(C)
#   v^μ = -mobility * g^{μν} ∂_ν(δP/δC)
#   J_μ = C g_{μν} v^ν
#
# Geometry:
#   K_μν = λ C g_μν + ξ ∂_μC ∂_νC + ζ J_μ J_ν
#   g_μν (next step) from algebraic inversion of K_μν
#
# Sparks:
#   - computed from smoothed C
#   - use Euclidean Hessian for simplicity but metric gradient norm
#   - currently only used for diagnostics, but could modulate ξ, ζ, etc.
#
# Global invariant:
#   ∫ C √|g| dx dy kept constant via projection after each full step.
# ============================================================

# ---------------------------
# Grid & parameters
# ---------------------------
Nx, Ny = 128, 128
dx = dy = 0.1

x = np.arange(Nx) * dx
y = np.arange(Ny) * dy

# Coherence & potential
lam_pot     = 0.8    # appears in K and V'(C)
xi_grad     = 0.2    # gradient contribution in K
zeta_flux   = 0.005  # flux-quadratic contribution in K (reduced)
kappa_grad  = 0.2    # gradient term in P (reduced a bit)
mobility    = 0.2    # mobility in v^μ (reduced a lot)
C_min, C_max = 0.0, 5.0

# Flux / time stepping
cfl_safety  = 0.3
vel_max_cap = 3.0     # cap on |v|

# Spark thresholds
spark_rel_det_thresh  = 0.15
spark_rel_grad_thresh = 0.30

# Metric regularisation
G_ABS_MAX   = 10.0    # bound on metric components
DETG_MIN    = 1e-3    # min allowed det(g)
DETG_MAX    = 1e+3    # max allowed det(g) before renormalisation
DETK_MIN    = 1e-6    # min allowed det(K) in inversion

eps = 1e-12

# ---------------------------
# Potential V(C) and derivative
# ---------------------------
def Vprime(C):
    # simple double-well-ish derivative: λ (C - C^2)
    return lam_pot * (C - C*C)

# ---------------------------
# Grid helpers
# ---------------------------
def compute_gradients(A):
    dAx = np.empty_like(A)
    dAy = np.empty_like(A)

    dAx[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2.0 * dx)
    dAy[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2.0 * dy)

    # periodic BCs
    dAx[0, :]    = (A[1, :]   - A[-1, :]) / (2.0 * dx)
    dAx[-1, :]   = (A[0, :]   - A[-2, :]) / (2.0 * dx)
    dAy[:, 0]    = (A[:, 1]   - A[:, -1]) / (2.0 * dy)
    dAy[:, -1]   = (A[:, 0]   - A[:, -2]) / (2.0 * dy)

    return dAx, dAy

def smooth_gaussian_like(A):
    # 3x3 kernel (1,2,1; 2,4,2; 1,2,1)/16
    w00 = 1.0/16.0
    w01 = 2.0/16.0
    w02 = 1.0/16.0

    A0 = A
    A_xp = np.roll(A, -1, axis=0)
    A_xm = np.roll(A,  1, axis=0)
    A_yp = np.roll(A, -1, axis=1)
    A_ym = np.roll(A,  1, axis=1)

    A_xp_yp = np.roll(A_xp, -1, axis=1)
    A_xp_ym = np.roll(A_xp,  1, axis=1)
    A_xm_yp = np.roll(A_xm, -1, axis=1)
    A_xm_ym = np.roll(A_xm,  1, axis=1)

    return (w00*(A_xm_ym + A_xm_yp + A_xp_ym + A_xp_yp)
            + w01*(A_xm + A_xp + A_ym + A_yp)
            + w02*A0)

# ---------------------------
# Initialization
# ---------------------------
def gaussian_blob(X, Y, cx, cy, sigma):
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))

X, Y = np.meshgrid(x, y, indexing='ij')

C = np.zeros((Nx, Ny), dtype=np.float64)
C += 1.0 * gaussian_blob(X, Y, x[Nx//4],     y[Ny//3],   sigma=2.5*dx)
C += 0.9 * gaussian_blob(X, Y, x[3*Nx//4],  y[2*Ny//3], sigma=2.8*dx)
C += 0.7 * gaussian_blob(X, Y, x[Nx//2],    y[Ny//2],   sigma=6.6*dx)
C += 0.8 * gaussian_blob(X, Y, x[3*Nx//5],  y[Ny//2],   sigma=4.6*dx)
C += 0.8 * gaussian_blob(X, Y, x[2*Nx//5],  y[Ny//2],   sigma=3.6*dx)
C += 0.8 * gaussian_blob(X, Y, x[1*Nx//5],  y[Ny//2],   sigma=5.6*dx)

rng = np.random.default_rng(42)
C += 0.02 * rng.normal(size=C.shape)
np.clip(C, C_min, C_max, out=C)

# Initial metric: flat
g_xx = np.ones_like(C)
g_xy = np.zeros_like(C)
g_yy = np.ones_like(C)

# ---------------------------
# Metric helpers
# ---------------------------
def regularise_metric(g_xx, g_xy, g_yy):
    """
    Keep metric components and determinant in a safe range.
    """
    # clamp components
    np.clip(g_xx, -G_ABS_MAX, G_ABS_MAX, out=g_xx)
    np.clip(g_xy, -G_ABS_MAX, G_ABS_MAX, out=g_xy)
    np.clip(g_yy, -G_ABS_MAX, G_ABS_MAX, out=g_yy)

    det_g = g_xx*g_yy - g_xy**2

    # where det_g is non-positive or NaN, reset to identity
    bad = (~np.isfinite(det_g)) | (det_g <= DETG_MIN)
    if np.any(bad):
        g_xx[bad] = 1.0
        g_xy[bad] = 0.0
        g_yy[bad] = 1.0
        det_g = g_xx*g_yy - g_xy**2

    # if determinant is too large, rescale metric to bring it closer to 1
    big = det_g > DETG_MAX
    if np.any(big):
        # scaling metric by alpha scales det by alpha^2
        alpha = np.sqrt(1.0 / (det_g[big] + eps))
        g_xx[big] *= alpha
        g_xy[big] *= alpha
        g_yy[big] *= alpha

    det_g = g_xx*g_yy - g_xy**2
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

# ---------------------------
# Coherence functional derivative δP/δC
# ---------------------------
def delta_P_over_delta_C(C, g_xx, g_xy, g_yy):
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)

    dC_dx, dC_dy = compute_gradients(C)

    # ∂^μ C = g^{μν} ∂_ν C
    dC_up = gxx_inv*dC_dx + gxy_inv*dC_dy
    dC_vp = gxy_inv*dC_dx + gyy_inv*dC_dy

    flux_x = sqrt_g * dC_up
    flux_y = sqrt_g * dC_vp

    div_x = (np.roll(flux_x, -1, axis=0) - np.roll(flux_x, 1, axis=0)) / (2.0 * dx)
    div_y = (np.roll(flux_y, -1, axis=1) - np.roll(flux_y, 1, axis=1)) / (2.0 * dy)

    laplace_B = (div_x + div_y) / (sqrt_g + eps)

    dVdC = Vprime(C)
    return -kappa_grad * laplace_B + dVdC

# ---------------------------
# Flux from gradient flow
# ---------------------------
def compute_flux(C, g_xx, g_xy, g_yy):
    phi = delta_P_over_delta_C(C, g_xx, g_xy, g_yy)
    dphi_dx, dphi_dy = compute_gradients(phi)

    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)

    # ∂^μ φ = g^{μν} ∂_ν φ
    grad_phi_up = gxx_inv*dphi_dx + gxy_inv*dphi_dy
    grad_phi_vp = gxy_inv*dphi_dx + gyy_inv*dphi_dy

    # v^μ = -mobility * ∂^μ φ
    v_up = -mobility * grad_phi_up
    v_vp = -mobility * grad_phi_vp

    # cap velocities
    v_up = np.clip(v_up,  -vel_max_cap, vel_max_cap)
    v_vp = np.clip(v_vp,  -vel_max_cap, vel_max_cap)

    # covariant flux J_μ = C g_{μν} v^ν
    Jx = C * (g_xx*v_up + g_xy*v_vp)
    Jy = C * (g_xy*v_up + g_yy*v_vp)

    # cap flux too (important for K)
    J_max = 10.0
    np.clip(Jx, -J_max, J_max, out=Jx)
    np.clip(Jy, -J_max, J_max, out=Jy)

    return Jx, Jy, v_up, v_vp, sqrt_g

# ---------------------------
# Covariant divergence for continuity
# ---------------------------
def covariant_divergence(Jx, Jy, g_xx, g_xy, g_yy):
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)

    # J^μ = g^{μν} J_ν
    J_up = gxx_inv*Jx + gxy_inv*Jy
    J_vp = gxy_inv*Jx + gyy_inv*Jy

    flux_x = sqrt_g * J_up
    flux_y = sqrt_g * J_vp

    div_x = (np.roll(flux_x, -1, axis=0) - np.roll(flux_x, 1, axis=0)) / (2.0 * dx)
    div_y = (np.roll(flux_y, -1, axis=1) - np.roll(flux_y, 1, axis=1)) / (2.0 * dy)

    rhs = -(div_x + div_y) / (sqrt_g + eps)
    return rhs

# ---------------------------
# Coherence tensor K and metric update
# ---------------------------
def update_metric_from_K(C, Jx, Jy, g_xx, g_xy, g_yy):
    # gradient of C
    dC_dx, dC_dy = compute_gradients(C)

    # K_μν = λ C g_μν + ξ ∂_μC ∂_νC + ζ J_μ J_ν
    K_xx = lam_pot*C*g_xx + xi_grad*dC_dx*dC_dx + zeta_flux*Jx*Jx
    K_xy = lam_pot*C*g_xy + xi_grad*dC_dx*dC_dy + zeta_flux*Jx*Jy
    K_yy = lam_pot*C*g_yy + xi_grad*dC_dy*dC_dy + zeta_flux*Jy*Jy

    # determinant with guard
    detK = K_xx*K_yy - K_xy**2
    # clamp extremely small |detK|
    bad_detK = (~np.isfinite(detK)) | (np.abs(detK) < DETK_MIN)
    detK[bad_detK] = DETK_MIN

    inv_detK = 1.0 / detK

    g_new_xx =  K_yy * inv_detK
    g_new_xy = -K_xy * inv_detK
    g_new_yy =  K_xx * inv_detK

    # clean NaNs / infs
    np.nan_to_num(g_new_xx, copy=False, nan=1.0, posinf=G_ABS_MAX, neginf=-G_ABS_MAX)
    np.nan_to_num(g_new_xy, copy=False, nan=0.0, posinf=G_ABS_MAX, neginf=-G_ABS_MAX)
    np.nan_to_num(g_new_yy, copy=False, nan=1.0, posinf=G_ABS_MAX, neginf=-G_ABS_MAX)

    # blend with old metric to avoid violent jumps
    blend = 0.1
    g_xx[:] = (1.0 - blend)*g_xx + blend*g_new_xx
    g_xy[:] = (1.0 - blend)*g_xy + blend*g_new_xy
    g_yy[:] = (1.0 - blend)*g_yy + blend*g_new_yy

    # final regularisation step
    g_xx, g_xy, g_yy, _ = regularise_metric(g_xx, g_xy, g_yy)
    return g_xx, g_xy, g_yy

# ---------------------------
# Spark detection (geometry-aware-ish)
# ---------------------------
def compute_spark_mask(C, g_xx, g_xy, g_yy, step=None):
    C_s = smooth_gaussian_like(C)

    # Euclidean Hessian (approx)
    Cxx = (np.roll(C_s, -1, axis=0) - 2.0*C_s + np.roll(C_s, 1, axis=0)) / (dx*dx)
    Cyy = (np.roll(C_s, -1, axis=1) - 2.0*C_s + np.roll(C_s, 1, axis=1)) / (dy*dy)
    Cxy = (np.roll(np.roll(C_s, -1, axis=0), -1, axis=1)
         - np.roll(np.roll(C_s, -1, axis=0),  1, axis=1)
         - np.roll(np.roll(C_s,  1, axis=0), -1, axis=1)
         + np.roll(np.roll(C_s,  1, axis=0),  1, axis=1)) / (4.0 * dx * dy)

    detH = Cxx*Cyy - Cxy*Cxy
    abs_detH = np.abs(detH)
    max_abs = np.max(abs_detH)

    if max_abs < 1e-12:
        return np.zeros_like(C)

    rel_det = abs_detH / (max_abs + eps)

    # gradient magnitude in the current metric
    dC_dx, dC_dy = compute_gradients(C_s)
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)
    grad_up = gxx_inv*dC_dx + gxy_inv*dC_dy
    grad_vp = gxy_inv*dC_dx + gyy_inv*dC_dy
    grad_norm = grad_up**2 + grad_vp**2
    max_grad = np.max(grad_norm)
    rel_grad = grad_norm / (max_grad + eps)

    spark_mask = ((rel_det < spark_rel_det_thresh) &
                  (rel_grad > spark_rel_grad_thresh)).astype(float)

    if step is not None and step % 50 == 0:
        n_spark = np.count_nonzero(spark_mask)
        print(f"[SPARK] step={step}: {n_spark} spark pixels")

    return spark_mask

# ---------------------------
# Global invariant projection
# ---------------------------
def project_to_invariant(C, g_xx, g_xy, g_yy, target_mass):
    det_g, sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
    mass = np.sum(C * sqrt_g) * dx * dy
    factor = target_mass / (mass + eps)
    C[:] *= factor
    return C

# ---------------------------
# CFL dt estimate (from v^μ)
# ---------------------------
def estimate_dt(v_up, v_vp):
    vmax = np.max(np.sqrt(v_up**2 + v_vp**2))
    if vmax < 1e-8:
        return 1e-3
    return cfl_safety * min(dx, dy) / vmax

# ---------------------------
# RHS(C) with fixed metric this sub-step
# ---------------------------
def rhs_C(C, g_xx, g_xy, g_yy):
    Jx, Jy, v_up, v_vp, sqrt_g = compute_flux(C, g_xx, g_xy, g_yy)
    dCdt = covariant_divergence(Jx, Jy, g_xx, g_xy, g_yy)
    return dCdt, Jx, Jy, v_up, v_vp

# ---------------------------
# RK2 step for C, then metric update via K
# ---------------------------
def rk2_step(C, g_xx, g_xy, g_yy, target_mass, step=None):
    dCdt1, Jx1, Jy1, v_up1, v_vp1 = rhs_C(C, g_xx, g_xy, g_yy)
    dt = estimate_dt(v_up1, v_vp1)

    C_tilde = C + dt*dCdt1
    np.clip(C_tilde, C_min, C_max, out=C_tilde)

    dCdt2, Jx2, Jy2, v_up2, v_vp2 = rhs_C(C_tilde, g_xx, g_xy, g_yy)

    C_new = C + 0.5*dt*(dCdt1 + dCdt2)
    np.clip(C_new, C_min, C_max, out=C_new)

    # project to exact mass invariance
    C_new = project_to_invariant(C_new, g_xx, g_xy, g_yy, target_mass)

    # update metric from K using second-stage flux
    g_xx, g_xy, g_yy = update_metric_from_K(C_new, Jx2, Jy2, g_xx, g_xy, g_yy)

    v_up_mid = 0.5*(v_up1 + v_up2)
    v_vp_mid = 0.5*(v_vp1 + v_vp2)

    return C_new, g_xx, g_xy, g_yy, dt, v_up_mid, v_vp_mid

# ---------------------------
# Set initial invariant mass
# ---------------------------
det_g0, sqrt_g0, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
target_mass0 = np.sum(C * sqrt_g0) * dx * dy
print("Initial mass:", target_mass0)

# ---------------------------
# Visualization
# ---------------------------
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], width_ratios=[5, 5, 5])

axC = fig.add_subplot(gs[0, 0])
imC = axC.imshow(C.T, origin='lower', cmap='viridis',
                 vmin=C_min, vmax=C_max)
cbarC = fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)
axC.set_title('Coherence C(x,y,t)')
axC.set_xlabel('x')
axC.set_ylabel('y')

axGeom = fig.add_subplot(gs[0, 1])
det_g_init, sqrt_g_init, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
imG = axGeom.imshow(sqrt_g_init.T, origin='lower', cmap='magma')
cbarG = fig.colorbar(imG, ax=axGeom, fraction=0.046, pad=0.04)
axGeom.set_title('Geometry √|g|(x,y,t)')

axS = fig.add_subplot(gs[0, 2])
imS = axS.imshow(np.zeros_like(C).T, origin='lower', cmap='inferno',
                 vmin=0.0, vmax=1.0)
cbarS = fig.colorbar(imS, ax=axS, fraction=0.046, pad=0.04)
axS.set_title('Spark mask')

ax_dt = fig.add_subplot(gs[1, 0])
line_dt, = ax_dt.plot([], [], 'b-')
ax_dt.set_xlim(0, 1)
ax_dt.set_ylim(0, 1e-2)
ax_dt.set_xlabel('Frame')
ax_dt.set_ylabel('dt')

ax_mass = fig.add_subplot(gs[1, 1])
line_mass, = ax_mass.plot([], [], 'r-')
ax_mass.set_xlim(0, 1)
ax_mass.set_ylim(0.9*target_mass0, 1.1*target_mass0)
ax_mass.set_xlabel('Frame')
ax_mass.set_ylabel('Total coherence')

dt_history = []
mass_history = []

title_text = fig.suptitle("", fontsize=9)

# Quiver
step_skip = 8
Xq = x[::step_skip]
Yq = y[::step_skip]
qx, qy = np.meshgrid(Xq, Yq, indexing='ij')
quiv = axC.quiver(qx, qy, qx*0.0, qy*0.0,
                  color='white', scale=5.0,
                  alpha=0.6, pivot='mid', linewidths=0.7)

# ---------------------------
# Animation loop
# ---------------------------
step_counter = 0

def update(frame):
    global C, g_xx, g_xy, g_yy, step_counter

    step_counter += 1
    C_new, g_xx_new, g_xy_new, g_yy_new, dt_here, v_up_mid, v_vp_mid = rk2_step(
        C, g_xx, g_xy, g_yy, target_mass0, step=step_counter
    )
    C[:]      = C_new
    g_xx[:]   = g_xx_new
    g_xy[:]   = g_xy_new
    g_yy[:]   = g_yy_new

    det_g, sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
    mass = np.sum(C * sqrt_g) * dx * dy
    if step_counter % 20 == 0:
        print("Mass check:", mass)

    dt_history.append(dt_here)
    mass_history.append(mass)

    imC.set_data(C.T)
    imG.set_data(sqrt_g.T)

    spark_mask = compute_spark_mask(C, g_xx, g_xy, g_yy, step=step_counter)
    imS.set_data(spark_mask.T)

    vx = v_up_mid[::step_skip, ::step_skip]
    vy = v_vp_mid[::step_skip, ::step_skip]
    quiv.set_UVC(vx, vy)

    n_frames = len(dt_history)
    line_dt.set_data(np.arange(n_frames), dt_history)
    ax_dt.set_xlim(0, max(2, 1.5*n_frames))

    line_mass.set_data(np.arange(n_frames), mass_history)

    txt = f"frame={n_frames}, step={step_counter}, dt={dt_here:.3e}, mass={mass:.4f}"
    title_text.set_text(txt)

    return [imC, imG, imS, quiv, line_dt, line_mass, title_text]

anim = FuncAnimation(fig, update, interval=60, blit=False)
plt.tight_layout()
plt.show()

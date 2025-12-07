import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# RC-style PDE toy model (C, G, I) in 2D
#
# Fields:
#   C(x,y,t)  - coherence density    (mass ~ conserved)
#   G(x,y,t)  - geometry scalar      (dynamical, chases G_target)
#   I(x,y,t)  - identity richness    (spark-driven, pruned)
#
# PDEs (schematic):
#
#   ∂t C + ∇·(C v) = 0
#
#   v = v_grad + v_rot
#     = -kappa_pot ∇(G V'(C)) - kappa_curv ∇(G ΔC)
#       + omega_rot * R(∇G)   [R rotates by 90°]
#
#   ∂t G = alpha_geom (G_target - G) + nu_geom ΔG
#
#   G_target = 1 + gamma_geom * (K_norm + k_id * I_norm) * (1 + spark_boost * S)
#   K_base   = lam_pot*C + xi_grad |∇C|^2
#   K_norm   = (K_base - K_min)/(K_max - K_min + eps)
#
#   ∂t I = D_I ΔI + eta_spark * S - eta_prune * P
#
#   S = spark mask from Hessian degeneracy of smoothed C
#   P = prune mask where C < C_prune_thresh
#
# ============================================================

# ---------------------------
# Grid
# ---------------------------
Nx, Ny = 128, 128
dx = dy = 0.1

x = np.arange(Nx) * dx
y = np.arange(Ny) * dy

# ---------------------------
# Parameters
# ---------------------------

# Potential / geometry
lam_pot     = 0.8     # in V'(C)
xi_grad     = 0.2     # |∇C|^2 contribution to K
gamma_geom  = 5.0     # def 3.0, geometry feedback
k_id_geom   = 1.5     # identity contribution to geometry
spark_boost = 1.0     # extra weight in spark regions

# Coherence flux
kappa_pot   = 0.8     # potential-driven velocity scale
kappa_curv  = 0.04    # curvature-driven velocity scale
beta_damp   = 0.1     # mild damping
omega_rot   = 1.0     # def 0.8 strength of rotational (non-gradient) component

# Geometry dynamics
alpha_geom  = 0.5     # relaxation rate towards G_target
nu_geom     = 0.05    # diffusion/smoothing of G

# Identity dynamics
D_I         = 0.02    # diffusion of identity richness
eta_spark   = 0.5     # def 0.3, how strongly sparks increase I
eta_prune   = 0.02    # def 0.05, how strongly low-C regions prune I
C_prune_thresh = 0.05 # prune I where C is tiny

# Spark detection thresholds
spark_rel_det_thresh  = 0.15  # small det(H)
spark_rel_grad_thresh = 0.30  # large |∇C|^2

# Numerics
cfl_safety  = 0.3
geom_min, geom_max = 0.3, 5.0
I_min, I_max       = 0.0, 10.0
vel_max            = 5.0
C_min, C_max       = 0.0, 5.0
eps                = 1e-12

# ---------------------------
# Potential derivative
# ---------------------------
def Vprime(C):
    # double-well-like: minima near C=0 and C=1
    return lam_pot * (C - C*C)

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

# Geometry: slightly perturbed around 1
G = 1.0 + 0.05 * rng.normal(size=C.shape)
G = np.clip(G, geom_min, geom_max)

# Identity richness: start small uniform
I = 0.1 + 0.02 * rng.normal(size=C.shape)
I = np.clip(I, I_min, I_max)

# ---------------------------
# Differential operators (periodic)
# ---------------------------
def compute_gradients(A):
    dAx = np.empty_like(A)
    dAy = np.empty_like(A)

    dAx[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2.0 * dx)
    dAy[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2.0 * dy)

    dAx[0, :]    = (A[1, :]   - A[-1, :]) / (2.0 * dx)
    dAx[-1, :]   = (A[0, :]   - A[-2, :]) / (2.0 * dx)
    dAy[:, 0]    = (A[:, 1]   - A[:, -1]) / (2.0 * dy)
    dAy[:, -1]   = (A[:, 0]   - A[:, -2]) / (2.0 * dy)

    return dAx, dAy

def laplacian(A):
    return (
        np.roll(A,  1, axis=0) + np.roll(A, -1, axis=0) +
        np.roll(A,  1, axis=1) + np.roll(A, -1, axis=1) -
        4.0 * A
    ) / (dx * dx)

def smooth_gaussian_like(A):
    """3x3 (1,2,1; 2,4,2; 1,2,1)/16 kernel, periodic."""
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

def hessian_det(C):
    Cxx = (np.roll(C, -1, axis=0) - 2.0*C + np.roll(C, 1, axis=0)) / (dx*dx)
    Cyy = (np.roll(C, -1, axis=1) - 2.0*C + np.roll(C, 1, axis=1)) / (dy*dy)

    Cxy = (np.roll(np.roll(C, -1, axis=0), -1, axis=1)
         - np.roll(np.roll(C, -1, axis=0),  1, axis=1)
         - np.roll(np.roll(C,  1, axis=0), -1, axis=1)
         + np.roll(np.roll(C,  1, axis=0),  1, axis=1)) / (4.0 * dx * dy)

    return Cxx * Cyy - Cxy * Cxy

# ---------------------------
# Spark & geometry target
# ---------------------------
def compute_spark_mask(C, step=None):
    # smooth C first to avoid noise-driven sparks
    C_s = smooth_gaussian_like(C)
    dCs_dx, dCs_dy = compute_gradients(C_s)
    g2_s = dCs_dx**2 + dCs_dy**2

    detH = hessian_det(C_s)
    abs_detH = np.abs(detH)
    max_abs = np.max(abs_detH)

    if max_abs < 1e-10:
        spark_mask = np.zeros_like(C)
    else:
        rel_det  = abs_detH / (max_abs + eps)
        rel_grad = g2_s / (np.max(g2_s) + eps)
        spark_mask = ((rel_det < spark_rel_det_thresh) &
                      (rel_grad > spark_rel_grad_thresh)).astype(np.float64)

        num_sparks = np.count_nonzero(spark_mask)
        if num_sparks > 0 and step is not None and step % 20 == 0:
            print(f"[SPARK] step={step}: {num_sparks} spark pixels")

    return spark_mask

def compute_K_and_G_target(C, I, spark_mask):
    # K_base from C and |∇C|^2
    dC_dx, dC_dy = compute_gradients(C)
    gmag2 = dC_dx**2 + dC_dy**2

    K_base = lam_pot * C + xi_grad * gmag2

    K_min = np.min(K_base)
    K_max = np.max(K_base)
    K_range = K_max - K_min
    if K_range < eps:
        K_norm = np.zeros_like(C)
    else:
        K_norm = (K_base - K_min) / (K_range + eps)

    # I_norm in [0,1]
    I_pos = np.maximum(I, 0.0)
    I_norm = I_pos / (np.max(I_pos) + eps)

    # G_target
    G_target = 1.0 + gamma_geom * (K_norm + k_id_geom * I_norm) * (1.0 + spark_boost * spark_mask)
    np.clip(G_target, geom_min, geom_max, out=G_target)

    return K_norm, G_target

# ---------------------------
# Velocity field v = v_grad + v_rot
# ---------------------------
def compute_velocity(C, G):
    Vp = Vprime(C)
    Phi_eff = G * Vp

    dPhi_dx, dPhi_dy = compute_gradients(Phi_eff)

    lap_C = laplacian(C)
    G_lapC = G * lap_C
    dCurv_dx, dCurv_dy = compute_gradients(G_lapC)

    # Gradient part
    u_grad = -kappa_pot * dPhi_dx - kappa_curv * dCurv_dx
    v_grad = -kappa_pot * dPhi_dy - kappa_curv * dCurv_dy

    # Rotational part from geometry: v_rot = omega * R(∇G) = omega * (-G_y, G_x)
    dG_dx, dG_dy = compute_gradients(G)
    u_rot = -omega_rot * dG_dy
    v_rot =  omega_rot * dG_dx

    u = u_grad + u_rot
    v = v_grad + v_rot

    if beta_damp > 0.0:
        u /= (1.0 + beta_damp)
        v /= (1.0 + beta_damp)

    u[~np.isfinite(u)] = 0.0
    v[~np.isfinite(v)] = 0.0

    np.clip(u, -vel_max, vel_max, out=u)
    np.clip(v, -vel_max, vel_max, out=v)

    return u, v

# ---------------------------
# Conservative upwind advection
# ---------------------------
def advection_flux_upwind(C, u_face_x, v_face_y):
    C_right = np.roll(C, -1, axis=0)
    F_x = np.where(u_face_x >= 0.0,
                   u_face_x * C,
                   u_face_x * C_right)
    dFx_dx = (F_x - np.roll(F_x, 1, axis=0)) / dx

    C_up = np.roll(C, -1, axis=1)
    F_y = np.where(v_face_y >= 0.0,
                   v_face_y * C,
                   v_face_y * C_up)
    dFy_dy = (F_y - np.roll(F_y, 1, axis=1)) / dy

    return -(dFx_dx + dFy_dy)

# ---------------------------
# RHS for (C, G, I)
# ---------------------------
def rhs(C, G, I, step=None):
    # Sparks
    spark_mask = compute_spark_mask(C, step=step)

    # G_target from K_norm and I
    K_norm, G_target = compute_K_and_G_target(C, I, spark_mask)

    # Velocity
    u, v = compute_velocity(C, G)

    u_face_x = 0.5 * (u + np.roll(u, -1, axis=0))
    v_face_y = 0.5 * (v + np.roll(v, -1, axis=1))

    advC = advection_flux_upwind(C, u_face_x, v_face_y)

    # Geometry dynamics
    lap_G = laplacian(G)
    dGdt = alpha_geom * (G_target - G) + nu_geom * lap_G

    # Identity dynamics: diffusion + sparks - prune
    lap_I = laplacian(I)
    prune_mask = (C < C_prune_thresh).astype(np.float64)
    dIdt = D_I * lap_I + eta_spark * spark_mask - eta_prune * prune_mask * I

    return advC, dGdt, dIdt, u, v, G_target, K_norm, spark_mask

# ---------------------------
# CFL
# ---------------------------
def estimate_dt(u, v):
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    max_vel = max(umax, vmax)
    if max_vel < 1e-8:
        return 1e-3
    return cfl_safety * min(dx, dy) / max_vel

# ---------------------------
# RK2 step for (C, G, I)
# ---------------------------
def rk2_step(C, G, I, step=None):
    adv1, dGdt1, dIdt1, u1, v1, Gt1, K1, S1 = rhs(C, G, I, step=step)
    dt = estimate_dt(u1, v1)

    C_tilde = C + dt * adv1
    np.clip(C_tilde, C_min, C_max, out=C_tilde)

    G_tilde = G + dt * dGdt1
    np.clip(G_tilde, geom_min, geom_max, out=G_tilde)

    I_tilde = I + dt * dIdt1
    np.clip(I_tilde, I_min, I_max, out=I_tilde)

    adv2, dGdt2, dIdt2, u2, v2, Gt2, K2, S2 = rhs(C_tilde, G_tilde, I_tilde, step=step)

    C_new = C + 0.5 * dt * (adv1 + adv2)
    np.clip(C_new, C_min, C_max, out=C_new)

    G_new = G + 0.5 * dt * (dGdt1 + dGdt2)
    np.clip(G_new, geom_min, geom_max, out=G_new)

    I_new = I + 0.5 * dt * (dIdt1 + dIdt2)
    np.clip(I_new, I_min, I_max, out=I_new)

    u_mid = 0.5 * (u1 + u2)
    v_mid = 0.5 * (v1 + v2)
    G_mid = 0.5 * (Gt1 + Gt2)
    S_mid = 0.5 * (S1 + S2)
    K_mid = K2  # just last one

    return C_new, G_new, I_new, dt, u_mid, v_mid, G_mid, K_mid, S_mid

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

axG = fig.add_subplot(gs[0, 1])
imG = axG.imshow(G.T, origin='lower', cmap='magma',
                 vmin=geom_min, vmax=geom_max)
cbarG = fig.colorbar(imG, ax=axG, fraction=0.046, pad=0.04)
axG.set_title('Geometry G(x,y,t)')

axS = fig.add_subplot(gs[0, 2])
imS = axS.imshow(np.zeros_like(C).T, origin='lower', cmap='inferno',
                 vmin=0.0, vmax=1.0)
cbarS = fig.colorbar(imS, ax=axS, fraction=0.046, pad=0.04)
axS.set_title('Spark mask S(x,y,t)')

# Velocity quiver on coarse grid
step_skip = 8
Xq = x[::step_skip]
Yq = y[::step_skip]
qx, qy = np.meshgrid(Xq, Yq, indexing='ij')
quiv = axC.quiver(qx, qy, qx*0.0, qy*0.0,
                  color='white', scale=5.0,
                  alpha=0.6, pivot='mid', linewidths=0.7)

# dt & mass
ax_dt = fig.add_subplot(gs[1, 0])
line_dt, = ax_dt.plot([], [], 'b-')
ax_dt.set_xlim(0, 1)
ax_dt.set_ylim(0, 1e-2)
ax_dt.set_xlabel('Frame')
ax_dt.set_ylabel('dt')

ax_mass = fig.add_subplot(gs[1, 1])
line_mass, = ax_mass.plot([], [], 'r-')
ax_mass.set_xlim(0, 1)
init_mass = float(np.sum(C) * dx * dy)
ax_mass.set_ylim(0.9 * init_mass, 1.1 * init_mass)
ax_mass.set_xlabel('Frame')
ax_mass.set_ylabel('Total coherence')

dt_history = []
mass_history = []

title_text = fig.suptitle("", fontsize=9)

# ---------------------------
# Animation loop
# ---------------------------
step_counter = 0

def update(frame):
    global C, G, I, step_counter

    step_counter += 1
    C_new, G_new, I_new, dt_here, u_mid, v_mid, G_mid, K_mid, S_mid = rk2_step(
        C, G, I, step=step_counter
    )
    C[:] = C_new
    G[:] = G_new
    I[:] = I_new

    total_mass = float(np.sum(C)) * dx * dy
    dt_history.append(dt_here)
    mass_history.append(total_mass)

    imC.set_data(C.T)
    imG.set_data(G_mid.T)
    imS.set_data(S_mid.T)

    qU = u_mid[::step_skip, ::step_skip]
    qV = v_mid[::step_skip, ::step_skip]
    quiv.set_UVC(qU, qV)

    n_frames = len(dt_history)
    line_dt.set_data(np.arange(n_frames), dt_history)
    ax_dt.set_xlim(0, max(1.5 * n_frames, 2))

    line_mass.set_data(np.arange(n_frames), mass_history)

    txt = (f"frame={n_frames}, step={step_counter}, "
           f"dt={dt_here:.3e}, mass={total_mass:.4f}")
    title_text.set_text(txt)

    return [imC, imG, imS, quiv, line_dt, line_mass, title_text]

anim = FuncAnimation(fig, update, interval=60, blit=False)
plt.tight_layout()
plt.show()

# To save instead of show, uncomment:
# import os
# os.makedirs("outputs", exist_ok=True)
# anim.save("outputs/rc_full_pde_toy.gif", writer="pillow", fps=5)

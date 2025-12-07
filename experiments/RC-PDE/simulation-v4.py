import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# RC / Paper IV style PDE in 2D, with DYNAMICAL geometry
#
#   ∂t C + ∇·(C v) = 0
#
#   v = -kappa_pot ∇(G * V'(C))
#       -kappa_curv ∇(G * ΔC)
#
#   ∂t G = alpha_geom * (G_target[C] - G) + nu_geom * ΔG
#
#   G_target = 1 + gamma_geom * (1 + spark_boost*spark_mask) * K_norm
#   K_base   = lam_pot*C + xi_grad|∇C|^2
#   K_norm   = (K_base - K_min) / (K_max - K_min + eps) ∈ [0,1]
#
# Sparks: localized regions where |det(Hess C_smooth)| is small
#         AND |∇C_smooth|^2 is nontrivial.
#
# The RC papers explicitly rely on mechanisms that PDE did *not* include:
#
# ### Missing in PDE implementation:
#
# * **Re-imposition of curvature** after each collapse
# * **Seed updates**
# * **Geometry history** (G doesn’t accumulate curvature events; it just relaxes)
# * **Operator changes** after spark events
# * **Non-dissipative components** of the RC flow
# * **Cycle-driven updates** (RC = *Reflexive Cycle*, not continuous relaxation)
# * **Environmental coupling**
# * **Reserve dynamics**
# * **Identity abundance feedback**
#
# These are the mechanisms that make the RC system:
#
# * avoid fixed points,
# * continuously restructure,
# * and produce open-ended dynamics.
#
# Your PDE captured **only one layer** of the RC loop:
# the “smooth coherence-geometry mutual shaping” part.
#
# That layer alone is dissipative → → fixed point.
#
# ============================================================

# ---------------------------
# Grid & numerical parameters
# ---------------------------
Nx, Ny = 128, 128
dx = dy = 0.1

x = np.arange(Nx) * dx
y = np.arange(Ny) * dy

# RC-like coefficients (dimensionless)
lam_pot     = 0.8     # strength in V'(C)
xi_grad     = 0.2     # |∇C|^2 contribution to K
zeta_flux   = 0.0     # |J|^2 contribution (still disabled)
gamma_geom  = 3.0     # geometry feedback strength
spark_boost = 1.0     # extra geometry weight in spark regions

kappa_pot   = 0.9     # potential-driven velocity scale
kappa_curv  = 0.04    # curvature-driven velocity scale
beta_damp   = 0.1     # mild velocity damping

alpha_geom  = 0.6     # how fast G chases G_target
nu_geom     = 0.05    # smoothing (diffusion) of geometry

cfl_safety  = 0.4     # CFL safety factor

# Geometry & velocity clipping
geom_min, geom_max = 0.5, 10.0
vel_max = 20.0        # max |u|, |v| to avoid blowup

# Coherence bounds
clip_min   = 0.0
clip_max   = 5.0
eps        = 1e-12

# Spark thresholds
spark_rel_det_thresh  = 0.15   # how small det(H) must be (relative)
spark_rel_grad_thresh = 0.30   # how large |∇C|^2 must be (relative)

# ---------------------------
# Potential V(C) and derivative
# ---------------------------
def Vprime(C):
    # dV/dC = lam_pot (C - C^2) => multi-well around C=0 and C=1
    return lam_pot * (C - C*C)

# ---------------------------
# Initialization of coherence field C(x,y)
# ---------------------------
def gaussian_blob(X, Y, cx, cy, sigma):
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))

X, Y = np.meshgrid(x, y, indexing='ij')
C = np.zeros((Nx, Ny), dtype=np.float64)

# Seed multiple basins / identities
C += 1.0 * gaussian_blob(X, Y, x[Nx//4],     y[Ny//3],   sigma=2.5*dx)
C += 0.9 * gaussian_blob(X, Y, x[3*Nx//4],  y[2*Ny//3], sigma=2.8*dx)
C += 0.7 * gaussian_blob(X, Y, x[Nx//2],    y[Ny//2],   sigma=6.6*dx)
C += 0.8 * gaussian_blob(X, Y, x[3*Nx//5],  y[Ny//2],   sigma=4.6*dx)
C += 0.8 * gaussian_blob(X, Y, x[2*Nx//5],  y[Ny//2],   sigma=3.6*dx)
C += 0.8 * gaussian_blob(X, Y, x[1*Nx//5],  y[Ny//2],   sigma=5.6*dx)

# Small noise to break symmetry
rng = np.random.default_rng(42)
C += 0.02 * rng.normal(size=C.shape)

np.clip(C, clip_min, clip_max, out=C)

# Initial geometry: slightly perturbed around 1
G = 1.0 + 0.05 * rng.normal(size=C.shape)
G = np.clip(G, geom_min, geom_max)

# ---------------------------
# Differential operators (periodic)
# ---------------------------
def compute_gradients(A):
    """Central differences with periodic BCs."""
    dAx = np.empty_like(A)
    dAy = np.empty_like(A)

    dAx[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2.0 * dx)
    dAy[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2.0 * dy)

    # periodic in x
    dAx[0, :]    = (A[1, :]   - A[-1, :]) / (2.0 * dx)
    dAx[-1, :]   = (A[0, :]   - A[-2, :]) / (2.0 * dx)
    # periodic in y
    dAy[:, 0]    = (A[:, 1]   - A[:, -1]) / (2.0 * dy)
    dAy[:, -1]   = (A[:, 0]   - A[:, -2]) / (2.0 * dy)

    return dAx, dAy

def laplacian(A):
    """Periodic 5-point Laplacian."""
    return (
        np.roll(A,  1, axis=0) + np.roll(A, -1, axis=0) +
        np.roll(A,  1, axis=1) + np.roll(A, -1, axis=1) -
        4.0 * A
    ) / (dx * dx)

def smooth_gaussian_like(A):
    """
    Simple 3x3 Gaussian-like smoothing kernel (1,2,1; 2,4,2; 1,2,1)/16.
    Periodic boundaries.
    """
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
    """
    Approximate determinant of Hessian of C for spark detection.
    """
    Cxx = (np.roll(C, -1, axis=0) - 2.0*C + np.roll(C, 1, axis=0)) / (dx*dx)
    Cyy = (np.roll(C, -1, axis=1) - 2.0*C + np.roll(C, 1, axis=1)) / (dy*dy)

    Cxy = (np.roll(np.roll(C, -1, axis=0), -1, axis=1)
         - np.roll(np.roll(C, -1, axis=0),  1, axis=1)
         - np.roll(np.roll(C,  1, axis=0), -1, axis=1)
         + np.roll(np.roll(C,  1, axis=0),  1, axis=1)) / (4.0 * dx * dy)

    detH = Cxx * Cyy - Cxy * Cxy
    return detH

# ---------------------------
# Geometry / RC reflexive loop
# ---------------------------
def compute_target_geometry(C, step=None):
    """
    Compute target geometry G_target[C] and spark mask,
    but do NOT update G itself here.
    """
    dC_dx, dC_dy = compute_gradients(C)
    gmag2 = dC_dx**2 + dC_dy**2

    K_base = lam_pot * C + xi_grad * gmag2

    # Normalize K into [0,1]
    K_min = np.min(K_base)
    K_max = np.max(K_base)
    K_range = K_max - K_min
    if K_range < eps:
        K_norm = np.zeros_like(C)
    else:
        K_norm = (K_base - K_min) / (K_range + eps)

    # Spark detection on smoothed C to suppress noise
    C_s = smooth_gaussian_like(C)
    dCs_dx, dCs_dy = compute_gradients(C_s)
    g2_s = dCs_dx**2 + dCs_dy**2

    detH = hessian_det(C_s)
    abs_detH = np.abs(detH)
    max_abs = np.max(abs_detH)

    if max_abs < 1e-10:
        spark_mask = np.zeros_like(C)
    else:
        rel_det  = abs_detH / (max_abs + eps)            # [0,1]
        rel_grad = g2_s / (np.max(g2_s) + eps)           # [0,1]

        spark_mask = ((rel_det < spark_rel_det_thresh) &
                      (rel_grad > spark_rel_grad_thresh)).astype(np.float64)

        num_sparks = np.count_nonzero(spark_mask)
        if num_sparks > 0 and step is not None and step % 20 == 0:
            print(f"[SPARK] step={step}: {num_sparks} spark pixels")

    # Target geometry
    G_target = 1.0 + gamma_geom * (1.0 + spark_boost * spark_mask) * K_norm

    return G_target, K_norm, spark_mask

def compute_velocity(C, G):
    """
    RC-style velocity using CURRENT geometry G:
        v = -kappa_pot ∇(G * V'(C))
            -kappa_curv ∇(G * ΔC)
    """
    Vp = Vprime(C)
    Phi_eff = G * Vp

    dPhi_dx, dPhi_dy = compute_gradients(Phi_eff)

    lap_C = laplacian(C)
    G_lapC = G * lap_C
    dCurv_dx, dCurv_dy = compute_gradients(G_lapC)

    u = -kappa_pot * dPhi_dx - kappa_curv * dCurv_dx
    v = -kappa_pot * dPhi_dy - kappa_curv * dCurv_dy

    if beta_damp > 0.0:
        u /= (1.0 + beta_damp)
        v /= (1.0 + beta_damp)

    u[~np.isfinite(u)] = 0.0
    v[~np.isfinite(v)] = 0.0

    np.clip(u, -vel_max, vel_max, out=u)
    np.clip(v, -vel_max, vel_max, out=v)

    return u, v

# ---------------------------
# Conservative upwind advection: ∂t C + div(C v) = 0
# ---------------------------
def advection_flux_upwind(C, u_face_x, v_face_y):
    """
    First-order upwind flux with periodic BCs, conservative:
      ∂t C = -[∂x F_x + ∂y F_y], where F_x = u C_upwind, etc.
    """
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

def rhs(C, G, step=None):
    """
    Compute RHS:
      ∂t C = -div(C v(C,G))
      ∂t G = alpha_geom (G_target[C] - G) + nu_geom ΔG
    plus auxiliary fields for diagnostics.
    """
    G_target, K_norm, spark_mask = compute_target_geometry(C, step=step)
    u_c, v_c = compute_velocity(C, G)

    # velocities at faces (simple average)
    u_face_x = 0.5 * (u_c + np.roll(u_c, -1, axis=0))
    v_face_y = 0.5 * (v_c + np.roll(v_c, -1, axis=1))

    advC = advection_flux_upwind(C, u_face_x, v_face_y)

    # geometry evolution
    lap_G = laplacian(G)
    dGdt = alpha_geom * (G_target - G) + nu_geom * lap_G

    return advC, dGdt, u_c, v_c, G_target, K_norm, spark_mask

# ---------------------------
# CFL time step
# ---------------------------
def estimate_dt(u, v):
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    max_vel = max(umax, vmax)
    if max_vel < 1e-8:
        return 1e-3
    return cfl_safety * min(dx, dy) / max_vel

# ---------------------------
# RK2 (Heun) time stepping for (C,G)
# ---------------------------
def rk2_step(C, G, step=None):
    adv1, dGdt1, u1, v1, Gt1, K1, spark1 = rhs(C, G, step=step)
    dt = estimate_dt(u1, v1)

    C_tilde = C + dt * adv1
    np.clip(C_tilde, clip_min, clip_max, out=C_tilde)

    G_tilde = G + dt * dGdt1
    np.clip(G_tilde, geom_min, geom_max, out=G_tilde)

    adv2, dGdt2, u2, v2, Gt2, K2, spark2 = rhs(C_tilde, G_tilde, step=step)

    C_new = C + 0.5 * dt * (adv1 + adv2)
    np.clip(C_new, clip_min, clip_max, out=C_new)

    G_new = G + 0.5 * dt * (dGdt1 + dGdt2)
    np.clip(G_new, geom_min, geom_max, out=G_new)

    u_mid = 0.5 * (u1 + u2)
    v_mid = 0.5 * (v1 + v2)
    G_mid = 0.5 * (Gt1 + Gt2)  # diagnostic: near the target
    spark_mid = 0.5 * (spark1 + spark2)

    return C_new, G_new, dt, u_mid, v_mid, G_mid, K2, spark_mid

# ---------------------------
# Visualization setup
# ---------------------------
fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], width_ratios=[5, 5, 5])

axC = fig.add_subplot(gs[0, 0])
imC = axC.imshow(C.T, origin='lower', cmap='viridis',
                 vmin=clip_min, vmax=clip_max)
cbarC = fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)
axC.set_title('Coherence field C(x,y,t)')
axC.set_xlabel('x')
axC.set_ylabel('y')

axGeom = fig.add_subplot(gs[0, 1])
imGeom = axGeom.imshow(G.T, origin='lower', cmap='magma',
                       vmin=geom_min, vmax=geom_max)
cbarGeom = fig.colorbar(imGeom, ax=axGeom, fraction=0.046, pad=0.04)
axGeom.set_title('Geometry factor G(x,y,t)')

axSpark = fig.add_subplot(gs[0, 2])
imSpark = axSpark.imshow(np.zeros_like(C).T, origin='lower', cmap='inferno',
                         vmin=0.0, vmax=1.0)
cbarSpark = fig.colorbar(imSpark, ax=axSpark, fraction=0.046, pad=0.04)
axSpark.set_title('Spark indicator')

# Velocity quiver on coarse grid
step_skip = 8
Xq = x[::step_skip]
Yq = y[::step_skip]
qx, qy = np.meshgrid(Xq, Yq, indexing='ij')
quiv = axC.quiver(qx, qy, qx*0.0, qy*0.0,
                  color='white', scale=5.0,
                  alpha=0.6, pivot='mid', linewidths=0.7)

# dt and mass plots
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
    global C, G, step_counter

    step_counter += 1
    C_new, G_new, dt_here, u_mid, v_mid, G_mid, K_mid, spark_mid = rk2_step(
        C, G, step=step_counter
    )
    C[:] = C_new
    G[:] = G_new

    total_mass = float(np.sum(C)) * dx * dy
    dt_history.append(dt_here)
    mass_history.append(total_mass)

    # Update fields
    imC.set_data(C.T)
    imGeom.set_data(G_mid.T)
    imSpark.set_data(spark_mid.T)

    # Update quiver (downsampled)
    qU = u_mid[::step_skip, ::step_skip]
    qV = v_mid[::step_skip, ::step_skip]
    quiv.set_UVC(qU, qV)

    # Update dt plot
    n_frames = len(dt_history)
    line_dt.set_data(np.arange(n_frames), dt_history)
    ax_dt.set_xlim(0, max(1.5 * n_frames, 2))

    # Update mass plot
    line_mass.set_data(np.arange(n_frames), mass_history)

    txt = f"frame={n_frames}, step={step_counter}, dt={dt_here:.3e}, mass={total_mass:.4f}"
    title_text.set_text(txt)

    return [imC, imGeom, imSpark, quiv, line_dt, line_mass, title_text]

anim = FuncAnimation(fig, update, interval=60, blit=False)
plt.tight_layout()
plt.show()

# To save instead of show, uncomment:
# import os
# os.makedirs("outputs", exist_ok=True)
# anim.save("outputs/rc_paper4_closed_loop_v6.gif", writer="pillow", fps=5)

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# ---------------------------
# Parameters (stable defaults)
# ---------------------------
Nx, Ny = 128, 128
dx = dy = 0.1
x = np.arange(Nx) * dx
y = np.arange(Ny) * dy

# RC-like coefficients (dimensionless)
lam_init   = 0.8      # initial nonlinearity weight (for V'(C))
xi_grad    = 0.1      # gradient term driving velocity via |∇C|^2
kappa_diff = 0.05     # small diffusion for stability
alpha_pot  = 1.0      # potential-driven velocity scale
beta_damp  = 0.1      # mild flux damping (optional)
cfl_safety = 0.4      # safety factor for time step control

# Bounding/clamping for positivity and boundedness
clip_min   = 0.0
clip_max   = 7.0
eps        = 1e-12

# Potential V(C) to produce multi-basin behavior:
# V'(C) = lam_init * (C - C**2), so it has stable minima near 0 and 1.
def Vprime(C):
    return lam_init * (C - C*C)

# Slope limiter for advection term: MC limiter on face values
def slope_limiter(left, right):
    # left/right are arrays defined at faces; returns limited gradient in [-minmod, minmod]
    r = np.where(np.abs(right) > eps, left / (right + 1e-20), 0.0)
    min_mod = np.maximum(0.0, np.minimum(1.0, r))
    return 0.5 * (left + right) * min_mod

# ---------------------------
# Initialization of C field
# ---------------------------
def gaussian_blob(X, Y, cx, cy, sigma):
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))

# Use indexing='ij' so X has shape (Nx, Ny) consistent with C[i,j]
X, Y = np.meshgrid(x, y, indexing='ij')
C = np.zeros((Nx, Ny), dtype=np.float64)

# Add a few blobs to seed multi-basin landscape
C += 1.0 * gaussian_blob(X, Y, x[Nx//4],     y[Ny//3],   sigma=2.5*dx)
C += 0.9 * gaussian_blob(X, Y, x[3*Nx//4],  y[2*Ny//3], sigma=2.8*dx)
C += 0.7 * gaussian_blob(X, Y, x[Nx//2],    y[Ny//2],   sigma=6.6*dx)
C += 0.8 * gaussian_blob(X, Y, x[3*Nx//5],    y[Ny//2],   sigma=4.6*dx)
C += 0.8 * gaussian_blob(X, Y, x[2*Nx//5],    y[Ny//2],   sigma=3.6*dx)
C += 0.8 * gaussian_blob(X, Y, x[1*Nx//5],    y[Ny//2],   sigma=5.6*dx)

# Small noise to break symmetry
rng = np.random.default_rng(42)
C += 0.02 * (rng.normal(size=C.shape))

np.clip(C, clip_min, clip_max, out=C)

# ---------------------------
# Helper: compute gradient and face velocities
# ---------------------------
def compute_gradients(A):
    # central differences; A is cell-centered shape (Nx,Ny)
    dAx = np.zeros_like(A)
    dAy = np.zeros_like(A)
    dAx[1:-1, :]  = (A[2:, :] - A[:-2, :]) / (2.0 * dx)
    dAy[:, 1:-1]  = (A[:, 2:] - A[:, :-2]) / (2.0 * dy)

    # edges: one-sided differences with periodic wrap
    dAx[0, :]     = (A[1, :]   - A[-1, :]) / (2.0 * dx)
    dAx[-1, :]    = (A[0, :]   - A[-2, :]) / (2.0 * dx)

    dAy[:, 0]     = (A[:, 1]   - A[:, -1]) / (2.0 * dy)
    dAy[:, -1]    = (A[:, 0]   - A[:, -2]) / (2.0 * dy)
    return dAx, dAy

def compute_velocity(C):
    # Gradients at centers
    dC_dx_c, dC_dy_c = compute_gradients(C)

    # |∇C|^2 term
    gmag2 = dC_dx_c**2 + dC_dy_c**2

    # Potential gradient (for flux driving)
    Vp = Vprime(C)
    dV_dx, dV_dy = compute_gradients(Vp)

    # Laplacian with periodic BCs (simple and safe)
    lap_C = (
        np.roll(C,  1, axis=0) + np.roll(C, -1, axis=0) +
        np.roll(C,  1, axis=1) + np.roll(C, -1, axis=1) -
        4.0 * C
    ) / (dx**2)

    # Gradients of Laplacian and |∇C|^2
    grad_lap_x, grad_lap_y = compute_gradients(lap_C)
    grad_gmag_x, grad_gmag_y = compute_gradients(gmag2)

    # Velocity field v_C = -(alpha_pot)*∇Vprime(C)
    #                     - kappa_diff*∇(Laplace C)
    #                     - xi_grad * 0.5 * ∇(|∇C|^2)
    u = - alpha_pot * dV_dx - kappa_diff * grad_lap_x - xi_grad * 0.5 * grad_gmag_x
    v = - alpha_pot * dV_dy - kappa_diff * grad_lap_y - xi_grad * 0.5 * grad_gmag_y

    # Optional mild damping to prevent runaway velocities (beta_damp)
    if beta_damp > 0.0:
        u *= (1.0 / (1.0 + beta_damp))
        v *= (1.0 / (1.0 + beta_damp))

    return u, v

def advection_flux_upwind(C, u_face_x, v_face_y):
    """
    Conservative first-order upwind advection with periodic BCs.
    C: (Nx, Ny) cell-centered
    u_face_x: (Nx, Ny) velocity at x-faces (i+1/2, j)
    v_face_y: (Nx, Ny) velocity at y-faces (i, j+1/2)
    """

    # --- X-direction flux F_x at faces i+1/2 ---
    # Positive u: use C[i,j]; negative u: use C[i+1,j]
    C_right = np.roll(C, -1, axis=0)

    F_x = np.where(u_face_x >= 0.0,
                   u_face_x * C,
                   u_face_x * C_right)

    # Divergence in x: (F_{i+1/2} - F_{i-1/2}) / dx
    dFx_dx = (F_x - np.roll(F_x, 1, axis=0)) / dx

    # --- Y-direction flux F_y at faces j+1/2 ---
    C_up = np.roll(C, -1, axis=1)

    F_y = np.where(v_face_y >= 0.0,
                   v_face_y * C,
                   v_face_y * C_up)

    dFy_dy = (F_y - np.roll(F_y, 1, axis=1)) / dy

    # Overall advection term: -div (C v)
    return -(dFx_dx + dFy_dy)


def rhs(C):
    u_c, v_c = compute_velocity(C)

    # velocities at faces
    u_face_x = 0.5 * (u_c + np.roll(u_c, -1, axis=0))   # at i+1/2
    v_face_y = 0.5 * (v_c + np.roll(v_c, -1, axis=1))   # at j+1/2

    adv = advection_flux_upwind(C, u_face_x, v_face_y)
    return adv

# ---------------------------
# CFL-based time step estimation
# ---------------------------
def estimate_dt(C):
    u, v = compute_velocity(C)
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    dt_adv = cfl_safety * min(dx, dy) / max(max(umax, vmax), 1e-8)

    # Diffusive stability (explicit scheme): dt <= dx^2/(4*kappa_diff)
    if kappa_diff > 0.0:
        dt_diff = 0.24 * dx**2 / kappa_diff
    else:
        dt_diff = np.inf

    return min(dt_adv, dt_diff)

# ---------------------------
# RK2 (Heun) stepper with CFL control
# ---------------------------
def rk2_step(C, dt):
    # Stage 1
    k1 = rhs(C)
    C_tilde = C + dt * k1
    np.clip(C_tilde, clip_min, clip_max, out=C_tilde)

    # Stage 2
    k2 = rhs(C_tilde)

    C_new = C + 0.5 * dt * (k1 + k2)
    np.clip(C_new, clip_min, clip_max, out=C_new)
    return C_new

# ---------------------------
# Visualization setup
# ---------------------------
fig = plt.figure(figsize=(9, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], width_ratios=[5, 5, 5])

axC = fig.add_subplot(gs[0, :])
imC = axC.imshow(C.T, origin='lower', cmap='viridis', vmin=clip_min, vmax=clip_max)
cbar = fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)
axC.set_title('Coherence field C(x,y,t)')
axC.set_xlabel('x')
axC.set_ylabel('y')

# Optional velocity quiver on a coarse grid
step_skip = 16
Xq = x[::step_skip]
Yq = y[::step_skip]
qx, qy = np.meshgrid(Xq, Yq, indexing='ij')

# Prepare quiver artists (empty initially)
quiv = axC.quiver(qx, qy, qx*0.0, qy*0.0, color='white',
                  scale=5.0, alpha=0.6, pivot='mid', linewidths=0.8)

ax_dt = fig.add_subplot(gs[1, 0])
line_dt, = ax_dt.plot([], [], 'b-')
ax_dt.set_xlim(0, 1)
ax_dt.set_ylim(0, 1e-2)
ax_dt.set_xlabel('Frame')

ax_mass = fig.add_subplot(gs[1, 1])
line_mass, = ax_mass.plot([], [], 'r-')
ax_mass.set_xlim(0, 1)
# Keep a fixed y-range for coherence total mass; initial is near Nx*Ny (since C ~ O(1))
init_mass = float(np.sum(C) * dx * dy)
ax_mass.set_ylim(0.9*init_mass, 1.1*init_mass)

dt_history = []
mass_history = []

# A suptitle object we can update each frame
title_text = fig.suptitle("", fontsize=9)

# ---------------------------
# Animation loop
# ---------------------------
def update(frame):
    global C

    # debug for direction issue
    # u, v = compute_velocity(C)
    # print("mean u:", np.mean(u), "mean v:", np.mean(v))

    # Number of substeps per animation frame (adjust for speed vs accuracy)
    n_substeps = 2

    # Compute dt using current field and CFL safety
    dt_est = estimate_dt(C)
    dt_here = min(dt_est, 1.5e-3)   # clamp max dt to avoid too large jumps

    for _ in range(n_substeps):
        C = rk2_step(C, dt_here)

    # Update mass/dt histories
    total_mass = float(np.sum(C)) * dx * dy
    dt_history.append(dt_here)
    mass_history.append(total_mass)

    # Update plots
    imC.set_data(C.T)

    u, v = compute_velocity(C)
    qU = u[::step_skip, ::step_skip]
    qV = v[::step_skip, ::step_skip]
    quiv.set_UVC(qU, qV)

    n_frames_plotted = len(dt_history)
    line_dt.set_data(np.arange(n_frames_plotted), dt_history)
    ax_dt.set_xlim(0, max(1.5*len(dt_history), 2))

    line_mass.set_data(np.arange(len(mass_history)), mass_history)

    # Text overlay: time step value
    txt = f"dt={dt_here:.4e}, frames={n_frames_plotted}"
    title_text.set_text(txt)

    return [imC, quiv, line_dt, line_mass]

anim = FuncAnimation(fig, update, interval=60, blit=False)
plt.tight_layout()
plt.show()

# Optional: save animation to GIF
# os.makedirs("outputs", exist_ok=True)
# anim.save('outputs/rc_coherence_sim.gif', writer='pillow', fps=5)

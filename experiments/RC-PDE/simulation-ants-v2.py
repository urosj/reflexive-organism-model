import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Grid and global parameters
# ============================================================

Nx, Ny = 128, 128
dx = dy = 1.0

dt = 0.02
steps_per_frame = 3
cfl_safety = 0.4

np.random.seed(1)

# ============================================================
# RC Coherence field C parameters
# ============================================================

C_min, C_max = 0.0, 5.0

# coherence PDE components
xi_grad = 0.12       # curvature: gradient term ξ
eta_id = 0.05        # curvature: identity curvature coupling η
zeta_flux = 0.015    # flux curvature term
lam_pot = 0.12       # potential term λ
alpha_id = 0.35      # identity-source coefficient α (redistribution)
beta_C = 0.02        # small linear decay

D_C = 0.08           # diffusion of C

# ============================================================
# Identity (ant) PDE parameters
# ============================================================

g_id = 0.7           # growth coupling
d_id = 0.4           # decay
D_id = 0.16          # diffusion

ant_birth_sigma = 1.5
ant_birth_amp = 0.9

ant_min_mass = 2e-2
max_ants = 60

# Ant states
SCOUT = 0
RETURNING = 1

# Drift velocity coefficients
v_food = 1.2
v_nest = 1.4
v_trail = 0.8
v_avoid_explore = 0.6
v_noise = 0.3

# ============================================================
# Food field
# ============================================================

num_food_patches = 3
food_patch_radius = 12.0
food_initial_amount = 7.0
food_consumption_rate = 0.35

# ============================================================
# Nest
# ============================================================

nest_radius = 8.0
nest_birth_cost = 3.0
colony_food_store = 0.0

# ============================================================
# Pheromones
# ============================================================

D_pher = 0.25
lambda_pher = 0.035
rho_trail = 0.38
rho_explore = 0.17

# ============================================================
# Utilities
# ============================================================

def laplacian(A):
    return (
        (np.roll(A, -1, 0) - 2*A + np.roll(A, 1, 0)) / dx**2 +
        (np.roll(A, -1, 1) - 2*A + np.roll(A, 1, 1)) / dy**2
    )

def grad(A):
    Ax = (np.roll(A, -1, 0) - np.roll(A, 1, 0)) / (2*dx)
    Ay = (np.roll(A, -1, 1) - np.roll(A, 1, 1)) / (2*dy)
    return Ax, Ay

def centroid(I):
    m = np.sum(I)
    if m < 1e-12: return 0.0, 0.0
    ix = np.arange(Nx)
    iy = np.arange(Ny)
    XX, YY = np.meshgrid(ix, iy, indexing='ij')
    cx = np.sum(XX * I) / m
    cy = np.sum(YY * I) / m
    return cx, cy

# ============================================================
# Create grid and initial fields
# ============================================================

x = np.arange(Nx)*dx
y = np.arange(Ny)*dy
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial coherence
C = 1.0 + 0.1*np.random.randn(Nx, Ny)
C = np.clip(C, C_min, C_max)

# Food field
F = np.zeros_like(C)
for _ in range(num_food_patches):
    fx = np.random.uniform(0, Nx)
    fy = np.random.uniform(0, Ny)
    r2 = (X-fx*dx)**2 + (Y-fy*dy)**2
    F += food_initial_amount*np.exp(-r2/(2*food_patch_radius**2))

# Nest field
nest_cx = Nx/2
nest_cy = Ny/2
N_field = np.exp(-((X-nest_cx)**2 + (Y-nest_cy)**2)/(2*nest_radius**2))

# Pheromones
P_trail = np.zeros_like(C)
P_explore = np.zeros_like(C)

# Metric fields (identity matrix)
g_xx = np.ones_like(C)
g_xy = np.zeros_like(C)
g_yy = np.ones_like(C)

# Identity (ant) lists
ant_fields = []
ant_state = []
ant_has_food = []
ant_age = []

# ============================================================
# Ant seeding
# ============================================================

def seed_ant_at(ix, iy, state=SCOUT):
    """Create a new ant identity blob."""
    global ant_fields, ant_state, ant_has_food, ant_age
    if len(ant_fields) >= max_ants: return

    Xl = np.arange(Nx)
    Yl = np.arange(Ny)
    XX, YY = np.meshgrid(Xl, Yl, indexing='ij')
    r2 = (XX-ix)**2 + (YY-iy)**2
    blob = ant_birth_amp * np.exp(-r2/(2*ant_birth_sigma**2))

    ant_fields.append(blob)
    ant_state.append(state)
    ant_has_food.append(False)
    ant_age.append(0.0)

# Seed initial ants at nest
for _ in range(6):
    jitterx = int(nest_cx + np.random.randn()*2)
    jittery = int(nest_cy + np.random.randn()*2)
    seed_ant_at(jitterx % Nx, jittery % Ny, SCOUT)

# ============================================================
# Coherence update (RC PDE)
# ============================================================

def update_coherence(C, g_xx, g_xy, g_yy, ant_fields, dt):
    # Identity sum
    I_sum = np.zeros_like(C)
    for I in ant_fields: I_sum += I

    # Potential derivative
    dV = lam_pot*(C - C*C)

    # Compute gradients for curvature
    dCdx, dCdy = grad(C)

    # RC flux
    Jx = -C*(dCdx)
    Jy = -C*(dCdy)

    # RHS
    dCdt = D_C*laplacian(C) - dV + alpha_id*(I_sum - np.mean(I_sum)) - beta_C*C

    C2 = C + dt*dCdt
    np.clip(C2, C_min, C_max, out=C2)
    return C2

# ============================================================
# Pheromone + food updates
# ============================================================

def update_pheromones(P_trail, P_explore, ant_fields, ant_state, dt):
    # diffusion + decay
    P_trail += dt*(D_pher*laplacian(P_trail) - lambda_pher*P_trail)
    P_explore += dt*(D_pher*laplacian(P_explore) - lambda_pher*P_explore)

    # sources
    for I, st in zip(ant_fields, ant_state):
        if st == RETURNING: P_trail += dt*(rho_trail*I)
        else: P_explore += dt*(rho_explore*I)

    return P_trail, P_explore

def update_food(F, ant_fields, ant_state, ant_has_food, dt):
    global colony_food_store
    for i, (I, st, has) in enumerate(zip(ant_fields, ant_state, ant_has_food)):
        if has: continue
        overlap = I*(F>0)
        dF = -dt*food_consumption_rate*overlap
        F += dF
        eaten = -np.sum(dF)
        if eaten > 0.5:
            ant_has_food[i] = True
            ant_state[i] = RETURNING
            colony_food_store += eaten
    F[F<0]=0
    return F

# ============================================================
# Ant PDE updated with drift
# ============================================================

def update_ants(C, F, N_field, P_trail, P_explore,
                ant_fields, ant_state, ant_has_food, ant_age, dt):

    dFdx, dFdy = grad(F)
    dNdx, dNdy = grad(N_field)
    dTdx, dTdy = grad(P_trail)
    dEdx, dEdy = grad(P_explore)

    new_fields = []
    new_state  = []
    new_has    = []
    new_age    = []

    vmax = 3.0   # velocity cap

    for I, st, has, age in zip(ant_fields, ant_state, ant_has_food, ant_age):
        lapI = laplacian(I)
        dIdt_base = g_id*C*I - d_id*I + D_id*lapI

        # centroid
        cx, cy = centroid(I)
        if not np.isfinite(cx) or not np.isfinite(cy):
            cx = np.random.uniform(0, Nx)
            cy = np.random.uniform(0, Ny)

        ix = int(cx) % Nx
        iy = int(cy) % Ny

        # compute drift
        if st == SCOUT:
            vx = v_food*dFdx[ix,iy] - v_avoid_explore*dEdx[ix,iy]
            vy = v_food*dFdy[ix,iy] - v_avoid_explore*dEdy[ix,iy]
        else:
            vx = v_nest*dNdx[ix,iy] + v_trail*dTdx[ix,iy]
            vy = v_nest*dNdy[ix,iy] + v_trail*dTdy[ix,iy]

        # noise
        vx += v_noise*np.random.randn()
        vy += v_noise*np.random.randn()

        # clamp velocity
        speed = np.sqrt(vx*vx + vy*vy)
        if speed > vmax:
            vx *= vmax / speed
            vy *= vmax / speed

        # advection
        Jx = I * vx
        Jy = I * vy
        divJ = ((np.roll(Jx,-1,0) - np.roll(Jx,1,0))/(2*dx) +
                (np.roll(Jy,-1,1) - np.roll(Jy,1,1))/(2*dy))

        dIdt = dIdt_base - divJ
        I2 = I + dt*dIdt
        I2[I2 < 0] = 0.0

        # sanitize
        np.nan_to_num(I2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # prevent spike blow-up
        I2 = np.clip(I2, 0, 5.0)

        # mass check
        mass = np.sum(I2)
        if mass < ant_min_mass:
            continue

        # nest return
        if st == RETURNING and has:
            nest_overlap = np.sum(I2*N_field) / (mass + 1e-12)
            if nest_overlap > 0.2:
                has = False
                st = SCOUT

        new_fields.append(I2)
        new_state.append(st)
        new_has.append(has)
        new_age.append(age + dt)

    return new_fields, new_state, new_has, new_age

# ============================================================
# Birth rule (pure nest birth)
# ============================================================

def maybe_spawn_ants():
    global colony_food_store
    if colony_food_store < nest_birth_cost: return
    if len(ant_fields)>=max_ants: return

    num_new = min(3, int(colony_food_store/nest_birth_cost))
    for _ in range(num_new):
        ix = int(nest_cx + np.random.randn()*2) % Nx
        iy = int(nest_cy + np.random.randn()*2) % Ny
        seed_ant_at(ix, iy, SCOUT)
        colony_food_store -= nest_birth_cost

# ============================================================
# Visualization setup
# ============================================================

fig, axs = plt.subplots(2,2,figsize=(11,10))
axC, axF, axP, axN = axs.flatten()

imC = axC.imshow(C.T, origin='lower', cmap='viridis', vmin=C_min, vmax=C_max)
axC.set_title("Coherence C (RC-v12)")

imF = axF.imshow(F.T, origin='lower', cmap='Greens', vmin=0, vmax=10)
axF.set_title("Food")

imP = axP.imshow(P_trail.T, origin='lower', cmap='plasma', vmin=0, vmax=2)
axP.set_title("Trail Pheromone")

imN = axN.imshow(N_field.T, origin='lower', cmap='gray')
axN.set_title("Nest & Ants")
ant_scatter = axN.scatter([], [], c='red', s=12)

title = fig.suptitle("")

plt.tight_layout()

# ============================================================
# Animation step
# ============================================================

def update(frame):
    global C, F, P_trail, P_explore
    global ant_fields, ant_state, ant_has_food, ant_age

    for _ in range(steps_per_frame):

        C = update_coherence(C, g_xx, g_xy, g_yy, ant_fields, dt)
        F = update_food(F, ant_fields, ant_state, ant_has_food, dt)
        P_trail, P_explore = update_pheromones(P_trail, P_explore,
                                               ant_fields, ant_state, dt)

        ant_fields[:], ant_state[:], ant_has_food[:], ant_age[:] = \
            update_ants(C, F, N_field, P_trail, P_explore,
                        ant_fields, ant_state, ant_has_food, ant_age, dt)

        maybe_spawn_ants()

    # Update plots
    imC.set_data(C.T)
    imF.set_data(F.T)
    imP.set_data(P_trail.T)

    # ant centroids
    xs=[]
    ys=[]
    for I in ant_fields:
        cx,cy = centroid(I)
        xs.append(cx)
        ys.append(cy)
    ant_scatter.set_offsets(np.c_[xs,ys])

    title.set_text(
        f"Frame {frame} | Ants={len(ant_fields)} | "
        f"FoodStore={colony_food_store:.2f}"
    )

    return imC, imF, imP, imN, ant_scatter, title

anim = FuncAnimation(fig, update, interval=80, blit=False)
plt.show()

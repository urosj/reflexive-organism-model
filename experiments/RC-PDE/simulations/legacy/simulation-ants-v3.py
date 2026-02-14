import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Grid & basic parameters
# ============================================================

Nx, Ny = 128, 128
dx = dy = 1.0

dt = 0.02
steps_per_frame = 3

np.random.seed(2)

# ============================================================
# Coherence field C (RC-flavoured utility)
# ============================================================

C_min, C_max = 0.0, 5.0

D_C      = 0.08     # diffusion of C
lam_pot  = 0.12     # potential strength (double-well-ish)
alpha_I  = 0.35     # identity → coherence redistribution
beta_C   = 0.02     # linear decay

# ============================================================
# Identity (ants) PDE parameters
# ============================================================

g_id       = 0.7    # growth from coherence
d_id       = 0.4    # decay
D_id       = 0.16   # diffusion

ant_birth_sigma = 1.5
ant_birth_amp   = 0.9
ant_min_mass    = 2e-2
max_ants        = 80

# Ant states
SCOUT     = 0
RETURNING = 1

# Drift velocity parameters
v_food          = 1.0
v_nest          = 1.2
v_trail         = 3.0     # strong attraction to trail
v_avoid_explore = 0.6     # repel exploration pheromone
v_noise         = 0.25
vmax_drift      = 1.2     # clamp magnitude of drift

# ============================================================
# Food, nest, pheromones
# ============================================================

num_food_patches     = 3
food_patch_radius    = 10.0
food_initial_amount  = 6.0
food_consumption_rate= 0.4

nest_radius          = 7.0
nest_birth_cost      = 3.0
colony_food_store    = 0.0   # global resource

# Pheromones
D_pher       = 0.05
lambda_pher  = 0.01
rho_trail    = 1.8    # strong trail deposition
rho_explore  = 0.05   # weak exploration deposition

# Geometry/weight coupling to trail pheromone
gamma_trail_geom = 0.8  # effective geometry weight from P_trail


# ============================================================
# Utilities
# ============================================================

def laplacian(A):
    return ((np.roll(A,-1,0) - 2*A + np.roll(A,1,0))/dx**2 +
            (np.roll(A,-1,1) - 2*A + np.roll(A,1,1))/dy**2)

def grad(A):
    Ax = (np.roll(A,-1,0) - np.roll(A,1,0))/(2*dx)
    Ay = (np.roll(A,-1,1) - np.roll(A,1,1))/(2*dy)
    return Ax, Ay

def centroid(I):
    m = np.sum(I)
    if m <= 1e-12:
        return np.random.uniform(0, Nx), np.random.uniform(0, Ny)
    ix = np.arange(Nx)
    iy = np.arange(Ny)
    XX, YY = np.meshgrid(ix, iy, indexing='ij')
    cx = np.sum(XX*I) / m
    cy = np.sum(YY*I) / m
    if not np.isfinite(cx) or not np.isfinite(cy):
        cx = np.random.uniform(0, Nx)
        cy = np.random.uniform(0, Ny)
    return cx, cy

def sample_grad_local(A, cx, cy, r=1):
    """Sample gradient of A around (cx,cy) as small patch average."""
    gx, gy = grad(A)
    ix = int(cx) % Nx
    iy = int(cy) % Ny
    xs = slice(max(ix-r,0), min(ix+r+1, Nx))
    ys = slice(max(iy-r,0), min(iy+r+1, Ny))
    patch_gx = gx[xs, ys]
    patch_gy = gy[xs, ys]
    if patch_gx.size == 0:
        return 0.0, 0.0
    return float(np.mean(patch_gx)), float(np.mean(patch_gy))

def nan_sanitize(A, clip_min=None, clip_max=None):
    np.nan_to_num(A, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_min is not None or clip_max is not None:
        np.clip(A, clip_min, clip_max, out=A)


# ============================================================
# Grid & initial fields
# ============================================================

x = np.arange(Nx)*dx
y = np.arange(Ny)*dy
X, Y = np.meshgrid(x, y, indexing='ij')

# Coherence C
C = 1.0 + 0.1*np.random.randn(Nx, Ny)
C = np.clip(C, C_min, C_max)

# Food field
F = np.zeros_like(C)
for _ in range(num_food_patches):
    fx = np.random.uniform(0, Nx)
    fy = np.random.uniform(0, Ny)
    r2 = (X-fx)**2 + (Y-fy)**2
    F += food_initial_amount*np.exp(-r2/(2*food_patch_radius**2))
F[F<0]=0

# Nest
nest_cx = Nx/2
nest_cy = Ny/2
N_field = np.exp(-((X-nest_cx)**2 + (Y-nest_cy)**2)/(2*nest_radius**2))

# Pheromones
P_trail   = np.zeros_like(C)
P_explore = np.zeros_like(C)

# Identity/ant containers
ant_fields    = []
ant_state     = []
ant_has_food  = []
ant_age       = []


# ============================================================
# Ant creation
# ============================================================

def seed_ant_at(ix, iy, state=SCOUT):
    """Create a new ant as a localized Gaussian identity blob."""
    global ant_fields, ant_state, ant_has_food, ant_age
    if len(ant_fields) >= max_ants:
        return

    XX = np.arange(Nx)
    YY = np.arange(Ny)
    XX, YY = np.meshgrid(XX, YY, indexing='ij')
    r2 = (XX-ix)**2 + (YY-iy)**2
    I = ant_birth_amp * np.exp(-r2/(2*ant_birth_sigma**2))

    ant_fields.append(I)
    ant_state.append(state)
    ant_has_food.append(False)
    ant_age.append(0.0)

# Seed a few initial ants at nest
for _ in range(6):
    jx = int(nest_cx + np.random.randn()*2) % Nx
    jy = int(nest_cy + np.random.randn()*2) % Ny
    seed_ant_at(jx, jy, SCOUT)


# ============================================================
# Coherence update
# ============================================================

def update_coherence(C, ant_fields, dt):
    # identity sum
    if ant_fields:
        I_sum = np.zeros_like(C)
        for I in ant_fields:
            I_sum += I
    else:
        I_sum = 0.0

    # double-well-ish potential derivative
    dV = lam_pot*(C - C*C)

    # identity-redistribution source: zero-mean
    if isinstance(I_sum, np.ndarray):
        I_mean = np.mean(I_sum)
        S = alpha_I*(I_sum - I_mean)
    else:
        S = 0.0

    dCdt = D_C*laplacian(C) - dV + S - beta_C*C
    C_new = C + dt*dCdt

    nan_sanitize(C_new, C_min, C_max)
    return C_new


# ============================================================
# Pheromones & food
# ============================================================

def update_pheromones(P_trail, P_explore, ant_fields, ant_state, dt):
    # diffusion + decay
    P_trail   += dt*(D_pher*laplacian(P_trail)   - lambda_pher*P_trail)
    P_explore += dt*(D_pher*laplacian(P_explore) - lambda_pher*P_explore)

    # deposition
    for I, st in zip(ant_fields, ant_state):
        if st == RETURNING:
            P_trail   += dt*(rho_trail * I)
        else:
            P_explore += dt*(rho_explore * I)

    nan_sanitize(P_trail,   0.0, 10.0)
    nan_sanitize(P_explore, 0.0, 10.0)
    return P_trail, P_explore

def update_food(F, ant_fields, ant_state, ant_has_food, dt):
    global colony_food_store
    if not ant_fields:
        return F

    for i, (I, st, has) in enumerate(zip(ant_fields, ant_state, ant_has_food)):
        if has:
            continue
        overlap = I*(F>0)
        dF = -dt*food_consumption_rate*overlap
        F += dF
        eaten = -np.sum(dF)
        if eaten > 0.5:
            ant_has_food[i] = True
            ant_state[i]    = RETURNING
            colony_food_store += eaten

    F[F<0] = 0.0
    nan_sanitize(F, 0.0, None)
    return F


# ============================================================
# Ant PDE with drift & geometry weight
# ============================================================

def update_ants(C, F, N_field, P_trail, P_explore,
                ant_fields, ant_state, ant_has_food, ant_age, dt):

    new_fields = []
    new_state  = []
    new_has    = []
    new_age    = []

    for I, st, has, age in zip(ant_fields, ant_state, ant_has_food, ant_age):
        lapI = laplacian(I)
        dIdt_base = g_id*C*I - d_id*I + D_id*lapI

        # centroid & local gradients
        cx, cy = centroid(I)

        # **geometry weight** from trail pheromone
        # (acts like an effective metric: stronger along trails)
        # here we just use P_trail as a scalar weight on how strongly
        # drift reacts; this is the "geometry" effect you asked for.
        g_weight = 1.0 + gamma_trail_geom * P_trail[int(cx)%Nx, int(cy)%Ny]

        # sample gradients around centroid
        dFdx, dFdy = sample_grad_local(F, cx, cy, r=1)
        dNdx, dNdy = sample_grad_local(N_field, cx, cy, r=1)
        dTdx, dTdy = sample_grad_local(P_trail, cx, cy, r=1)
        dEdx, dEdy = sample_grad_local(P_explore, cx, cy, r=1)

        # drift velocity
        if st == SCOUT:
            vx = v_food*dFdx - v_avoid_explore*dEdx
            vy = v_food*dFdy - v_avoid_explore*dEdy
        else:  # RETURNING
            vx = v_nest*dNdx + v_trail*dTdx
            vy = v_nest*dNdy + v_trail*dTdy

        # geometry amplifies the trail influence
        vx *= g_weight
        vy *= g_weight

        # add noise
        vx += v_noise*np.random.randn()
        vy += v_noise*np.random.randn()

        # clamp drift
        speed = np.sqrt(vx*vx + vy*vy)
        if speed > vmax_drift:
            vx *= vmax_drift/speed
            vy *= vmax_drift/speed

        # advection term -div(I v)
        Jx = I*vx
        Jy = I*vy
        divJ = ((np.roll(Jx,-1,0) - np.roll(Jx,1,0))/(2*dx) +
                (np.roll(Jy,-1,1) - np.roll(Jy,1,1))/(2*dy))

        dIdt = dIdt_base - divJ
        I_new = I + dt*dIdt
        I_new[I_new<0] = 0.0
        nan_sanitize(I_new, 0.0, 5.0)

        mass = np.sum(I_new)
        if mass < ant_min_mass:
            continue

        # RETURNING ants drop food in nest region
        if st == RETURNING and has:
            nest_overlap = np.sum(I_new*N_field)/(mass+1e-12)
            if nest_overlap > 0.2:  # threshold
                has = False
                st  = SCOUT

        new_fields.append(I_new)
        new_state.append(st)
        new_has.append(has)
        new_age.append(age+dt)

    return new_fields, new_state, new_has, new_age


# ============================================================
# Nest-based birth (no sparks)
# ============================================================

def maybe_spawn_ants():
    global colony_food_store
    if colony_food_store < nest_birth_cost:
        return
    if len(ant_fields) >= max_ants:
        return

    num_new = min(3, int(colony_food_store / nest_birth_cost))
    for _ in range(num_new):
        ix = int(nest_cx + np.random.randn()*2) % Nx
        iy = int(nest_cy + np.random.randn()*2) % Ny
        seed_ant_at(ix, iy, SCOUT)
        colony_food_store -= nest_birth_cost
        if colony_food_store < 0:
            colony_food_store = 0.0


# ============================================================
# Visualisation
# ============================================================

fig, axs = plt.subplots(2,2,figsize=(11,10))
axC, axF, axP, axN = axs.flatten()

imC = axC.imshow(C.T, origin='lower', cmap='viridis', vmin=C_min, vmax=C_max)
axC.set_title("Coherence C (utility)")

imF = axF.imshow(F.T, origin='lower', cmap='Greens', vmin=0, vmax=8)
axF.set_title("Food F")

imP = axP.imshow(P_trail.T, origin='lower', cmap='plasma', vmin=0, vmax=4)
axP.set_title("Trail pheromone")

imN = axN.imshow(N_field.T, origin='lower', cmap='gray', vmin=0, vmax=1)
axN.set_title("Nest & ants")
ant_scatter = axN.scatter([], [], c='red', s=10)

title = fig.suptitle("")
plt.tight_layout()


# ============================================================
# Animation update
# ============================================================

def update(frame):
    global C, F, P_trail, P_explore
    global ant_fields, ant_state, ant_has_food, ant_age

    for _ in range(steps_per_frame):
        C = update_coherence(C, ant_fields, dt)
        F = update_food(F, ant_fields, ant_state, ant_has_food, dt)
        P_trail, P_explore = update_pheromones(P_trail, P_explore,
                                               ant_fields, ant_state, dt)

        ant_fields[:], ant_state[:], ant_has_food[:], ant_age[:] = \
            update_ants(C, F, N_field, P_trail, P_explore,
                        ant_fields, ant_state, ant_has_food, ant_age, dt)

        maybe_spawn_ants()

        # global sanitization
        nan_sanitize(C, C_min, C_max)
        nan_sanitize(F, 0.0, None)
        nan_sanitize(P_trail, 0.0, 10.0)
        nan_sanitize(P_explore, 0.0, 10.0)

    imC.set_data(C.T)
    imF.set_data(F.T)
    imP.set_data(P_trail.T)

    xs, ys = [], []
    for I in ant_fields:
        cx, cy = centroid(I)
        xs.append(cx)
        ys.append(cy)
    if xs:
        ant_scatter.set_offsets(np.c_[xs, ys])
    else:
        ant_scatter.set_offsets(np.empty((0,2)))

    total_food = np.sum(F)
    title.set_text(
        f"frame={frame}  ants={len(ant_fields)}  "
        f"colony_food={colony_food_store:.2f}  "
        f"food_total={total_food:.2f}"
    )
    return imC, imF, imP, imN, ant_scatter, title


anim = FuncAnimation(fig, update, interval=80, blit=False)
plt.show()

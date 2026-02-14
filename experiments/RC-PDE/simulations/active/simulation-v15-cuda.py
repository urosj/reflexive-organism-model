"""
RC-PDE v15 CUDA: core-first RC scaffolding with explicit phase routing.

What v15 iteration-1 is trying to achieve:
  - Freeze v13/v14 baselines for reproducible comparison.
  - Introduce explicit step phases: core, events, closure.
  - Add closure mode plumbing (`off|soft|full`) without changing default runtime behavior yet.

Quick run commands (copy/paste):
  # Best full-v15 run (interactive / non-headless)
  python simulations/active/simulation-v15-cuda.py --closure-mode soft --closure-softness 0.8 --spark-softness 0.08 --collapse-softness 0.5 --identity-cap-fraction 0.25 --coherence-bg-floor 0.02 --identity-birth-gate-fraction 0.9

  # Best full-v15 run (headless, reproducible artifacts)
  python simulations/active/simulation-v15-cuda.py --headless --headless-steps 5000 --nx 1024 --ny 1024 --dx 0.1 --seed 1 --storage-mode disk --snapshot-interval 50 --closure-mode soft --closure-softness 0.8 --spark-softness 0.08 --collapse-softness 0.5 --identity-cap-fraction 0.25 --coherence-bg-floor 0.02 --identity-birth-gate-fraction 0.9

  # Interactive run (VisPy if available)
  python simulations/active/simulation-v15-cuda.py

  # Force matplotlib UI
  python simulations/active/simulation-v15-cuda.py --no-vispy

  # Headless long run with disk snapshots (recommended)
  python simulations/active/simulation-v15-cuda.py --headless --headless-steps 5000 --storage-mode disk --snapshot-interval 50

  # Large grid run
  python simulations/active/simulation-v15-cuda.py --headless --storage-mode disk --snapshot-interval 100 --nx 1024 --ny 1024

  # Core-only run (L0 only; no event/closure identity layer)
  python simulations/active/simulation-v15-cuda.py --headless --closure-mode off --seed 1

  # Core+events run (L0+L1 signal, no L2 closure identities)
  python simulations/active/simulation-v15-cuda.py --headless --closure-mode off --events-control-in-core --seed 1

  # Closure routing (default behavior: closure active)
  python simulations/active/simulation-v15-cuda.py --headless --closure-mode soft --seed 1

  # Closure softness family
  python simulations/active/simulation-v15-cuda.py --headless --closure-softness 0.8 --spark-softness 0.08 --collapse-softness 0.5

  # Identity budget controls
  python simulations/active/simulation-v15-cuda.py --identity-cap-fraction 0.25 --coherence-bg-floor 0.02 --identity-birth-gate-fraction 0.9

Help:
  python simulations/active/simulation-v15-cuda.py --help
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import atexit
import json
import argparse
import torch
# CuPy removed - using pure PyTorch for identity birth selection (torch.randperm)
import numpy as np      # Still needed for some CPU operations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from configs.blob_config import load_blob_specs
from configs.activity_config import map_activity
from experiments.scripts.readback_hooks import InMemoryHookRecorder, default_telemetry_jsonl_path

# CUDA-preferred runtime with CPU fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="RC-PDE v15 CUDA simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python simulations/active/simulation-v15-cuda.py --closure-mode soft --closure-softness 0.8 --spark-softness 0.08 --collapse-softness 0.5 --identity-cap-fraction 0.25 --coherence-bg-floor 0.02 --identity-birth-gate-fraction 0.9\n"
            "  python simulations/active/simulation-v15-cuda.py --headless --headless-steps 5000 --nx 1024 --ny 1024 --dx 0.1 --seed 1 --storage-mode disk --snapshot-interval 50 --closure-mode soft --closure-softness 0.8 --spark-softness 0.08 --collapse-softness 0.5 --identity-cap-fraction 0.25 --coherence-bg-floor 0.02 --identity-birth-gate-fraction 0.9\n"
            "  python simulations/active/simulation-v15-cuda.py\n"
            "  python simulations/active/simulation-v15-cuda.py --no-vispy\n"
            "  python simulations/active/simulation-v15-cuda.py --headless --closure-mode off --seed 1\n"
            "  python simulations/active/simulation-v15-cuda.py --headless --closure-mode off --events-control-in-core --seed 1\n"
            "  python simulations/active/simulation-v15-cuda.py --headless --storage-mode disk --snapshot-interval 50\n"
            "  python simulations/active/simulation-v15-cuda.py --headless --storage-mode disk --snapshot-interval 100 --nx 1024 --ny 1024\n"
        ),
    )

    # Core grid / discretization
    parser.add_argument("--nx", type=int, default=512, help="Grid size in x")
    parser.add_argument("--ny", type=int, default=512, help="Grid size in y")
    parser.add_argument("--dx", type=float, default=0.1, help="Grid spacing (dx=dy)")

    # Runtime modes
    parser.add_argument("--headless", action="store_true", help="Run without live UI")
    parser.add_argument("--headless-steps", type=int, default=2000, help="Steps for headless mode")
    parser.add_argument("--render-interval", type=int, default=10, help="Render every N frames")

    # VisPy / Matplotlib
    vispy_group = parser.add_mutually_exclusive_group()
    vispy_group.add_argument("--use-vispy", action="store_true", help="Force VisPy viewer")
    vispy_group.add_argument("--no-vispy", action="store_true", help="Disable VisPy (use matplotlib)")

    # Snapshotting
    parser.add_argument("--snapshot-interval", type=int, default=10, help="Store snapshots every N steps")
    parser.add_argument("--downsample", type=int, default=0, help="Downsample factor (0/1 = none)")
    parser.add_argument("--storage-mode", choices=["memory", "disk"], default="memory", help="Snapshot storage mode")
    parser.add_argument("--snapshot-dir", type=str, default="./snapshots", help="Snapshot directory (disk mode)")
    parser.add_argument("--max-snapshots-memory", type=int, default=500, help="Warn after this many snapshots in memory")

    # Animation export
    parser.add_argument("--fps", type=int, default=10, help="Export FPS for offline animation")
    parser.add_argument("--animate-interval", type=int, default=100, help="Animation frame interval (ms)")
    parser.add_argument("--two-pass", action="store_true", help="Two-pass global sqrt_g scaling for streaming animation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--identity-cap-fraction", type=float, default=0.25, help="Identity mass cap as fraction of initial structured coherence mass")
    parser.add_argument("--coherence-bg-floor", type=float, default=0.02, help="Background coherence floor for structured mass estimate")
    parser.add_argument("--identity-birth-gate-fraction", type=float, default=0.9, help="Birth gate threshold as fraction of identity mass cap")
    parser.add_argument("--closure-mode", choices=["off", "soft", "full"], default="soft", help="v15 routing mode: off=L0 core-only (no events/closure identities), soft/full=closure active")
    parser.add_argument("--events-control-in-core", action="store_true", help="Ablation helper: allow L1 spark control signal in closure-mode=off (L0+L1, no L2)")
    parser.add_argument("--closure-softness", type=float, default=0.6, help="Blend from hard RC-III gates (0) to smoother field-driven closures (1)")
    parser.add_argument("--spark-softness", type=float, default=0.08, help="Soft spark transition width for continuous spark intensity")
    parser.add_argument("--collapse-softness", type=float, default=0.5, help="Soft collapse damping strength near id_min_mass")
    parser.add_argument("--activity", type=float, default=None, help="Single slider in [0, 1] that overrides closure/budget knobs (for quick regime search)")

    return parser.parse_args()


ARGS = _parse_args()

if ARGS.seed is not None:
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)

if ARGS.identity_cap_fraction <= 0.0:
    raise ValueError("--identity-cap-fraction must be > 0")
if not (0.0 < ARGS.identity_birth_gate_fraction <= 1.0):
    raise ValueError("--identity-birth-gate-fraction must be in (0, 1]")
if ARGS.coherence_bg_floor < 0.0:
    raise ValueError("--coherence-bg-floor must be >= 0")
if not (0.0 <= ARGS.closure_softness <= 1.0):
    raise ValueError("--closure-softness must be in [0, 1]")
if ARGS.spark_softness <= 0.0:
    raise ValueError("--spark-softness must be > 0")
if not (0.0 <= ARGS.collapse_softness <= 1.0):
    raise ValueError("--collapse-softness must be in [0, 1]")
if ARGS.activity is not None and not (0.0 <= ARGS.activity <= 1.0):
    raise ValueError("--activity must be in [0, 1]")

if ARGS.activity is not None:
    mapped = map_activity(float(ARGS.activity))
    ARGS.identity_cap_fraction = mapped["identity_cap_fraction"]
    ARGS.identity_birth_gate_fraction = mapped["identity_birth_gate_fraction"]
    ARGS.closure_softness = mapped["closure_softness"]
    ARGS.spark_softness = mapped["spark_softness"]
    ARGS.collapse_softness = mapped["collapse_softness"]


def _configure_vispy_backend(require_gui_backend=False):
    """Choose a GUI-capable VisPy backend for interactive mode."""
    from vispy import app as vispy_app

    preferred_backends = ("pyqt5", "pyside2", "pyqt6", "pyside6", "glfw", "pyglet", "sdl2", "tkinter")
    for backend in preferred_backends:
        try:
            app_instance = vispy_app.use_app(backend)
            print(f"[INFO] VisPy backend: {app_instance.backend_name}")
            return True
        except Exception:
            continue

    # Let VisPy pick whatever is available; reject offscreen-only backends for live UI.
    try:
        app_instance = vispy_app.use_app()
        backend_name = (app_instance.backend_name or "").lower()
    except Exception as exc:
        msg = (
            "VisPy installed but no usable GUI backend found. "
            "Install one of: PyQt5, PySide2, PyQt6, or PySide6."
        )
        if require_gui_backend:
            raise RuntimeError(msg) from exc
        print(f"[WARN] {msg} Falling back to matplotlib UI.")
        return False

    if backend_name in ("egl", "osmesa", ""):
        msg = (
            f"VisPy selected offscreen backend '{app_instance.backend_name}'. "
            "Install one of: PyQt5, PySide2, PyQt6, or PySide6 for an on-screen GUI."
        )
        if require_gui_backend:
            raise RuntimeError(msg)
        print(f"[WARN] {msg} Falling back to matplotlib UI.")
        return False

    print(f"[INFO] VisPy backend: {app_instance.backend_name}")
    return True

# ============================================================
# GLOBAL PARAMETERS - Easily scalable for large grids
# ============================================================
Nx, Ny = ARGS.nx, ARGS.ny
dx = dy = ARGS.dx

# Coherence & potential
try:
    negctrl_lam_scale = max(0.0, float(os.environ.get("RC_NEGCTRL_LAM_SCALE", "1.0")))
except ValueError:
    negctrl_lam_scale = 1.0
lam_pot     = 0.5 * negctrl_lam_scale
xi_grad     = 0.2
zeta_flux   = 0.001
try:
    readback_beta_rb = max(0.0, float(os.environ.get("RC_READBACK_BETA_RB", "1.0")))
except ValueError:
    readback_beta_rb = 1.0
zeta_flux_eff = zeta_flux * readback_beta_rb
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
I_global_mass_cap      = None    # set from initial structured coherence mass after initialization
I_cap_fraction         = ARGS.identity_cap_fraction
coherence_bg_floor     = ARGS.coherence_bg_floor
I_birth_gate_fraction  = ARGS.identity_birth_gate_fraction
closure_mode           = ARGS.closure_mode
events_control_in_core = ARGS.events_control_in_core
closure_softness       = ARGS.closure_softness
spark_softness         = ARGS.spark_softness
collapse_softness      = ARGS.collapse_softness

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

# Paper-14 iteration-2.1: in-memory stage hooks (no persistence yet).
readback_hooks = InMemoryHookRecorder(enabled=(os.environ.get("RC_READBACK_HOOKS", "1") != "0"))
hook_missing_steps = []
atexit.register(readback_hooks.close)


def _hook_step():
    return int(step_counter) if "step_counter" in globals() else None


def _hook(stage, **payload):
    if not readback_hooks.enabled:
        return
    if not getattr(_hook, "_configured", False):
        if "HEADLESS_MODE" in globals() and HEADLESS_MODE:
            telemetry_path = default_telemetry_jsonl_path(
                sim_tag=os.path.splitext(os.path.basename(__file__))[0],
                seed=ARGS.seed,
            )
            readback_hooks.enable_jsonl(
                telemetry_path,
                metadata={
                    "sim_version": os.path.splitext(os.path.basename(__file__))[0],
                    "seed": ARGS.seed,
                    "headless": True,
                    "storage_mode": STORAGE_MODE if "STORAGE_MODE" in globals() else None,
                    "beta_rb": readback_beta_rb,
                    "zeta_flux_eff": zeta_flux_eff,
                    "negctrl_lam_scale": negctrl_lam_scale,
                },
            )
            print(f"[TELEMETRY] hook_stream={telemetry_path}")
            if os.environ.get("RC_READBACK_FIELD_DUMP", "0") == "1":
                field_npz = telemetry_path.replace(".jsonl", ".fields.npz")
                readback_hooks.enable_field_dump(
                    field_npz,
                    downsample=int(os.environ.get("RC_READBACK_FIELD_DOWNSAMPLE", "4")),
                    every_n_steps=int(os.environ.get("RC_READBACK_FIELD_INTERVAL", "10")),
                )
                print(f"[TELEMETRY] field_dump={field_npz}")
        _hook._configured = True
    readback_hooks.emit(stage, step=_hook_step(), **payload)

# Optional extra potential terms
USE_SPARK_DEEPENING = True
USE_IDENTITY_TILT   = True

# Visualization throttling (for performance on large grids)
RENDER_INTERVAL = ARGS.render_interval
HEADLESS_MODE = ARGS.headless
HEADLESS_STEPS = ARGS.headless_steps
USE_VISPY = not ARGS.no_vispy
if ARGS.use_vispy:
    USE_VISPY = True

if USE_VISPY and not HEADLESS_MODE:
    try:
        if _configure_vispy_backend(require_gui_backend=ARGS.use_vispy):
            from simulations.helpers.vispy_viewer import RCVispyViewer
        else:
            USE_VISPY = False
    except ModuleNotFoundError as exc:
        if exc.name != "vispy":
            raise
        if ARGS.use_vispy:
            raise ModuleNotFoundError(
                "VisPy viewer requested but vispy is not installed. "
                "Install dependencies with: pip install -r requirements-cuda.txt"
            ) from exc
        print("[WARN] VisPy not installed; falling back to matplotlib UI.")
        USE_VISPY = False

# HEADLESS MODE SNAPSHOT & ANIMATION PARAMETERS
SNAPSHOT_INTERVAL = ARGS.snapshot_interval
DOWNSAMPLE_FACTOR = None if ARGS.downsample in (0, 1) else ARGS.downsample
STORAGE_MODE = ARGS.storage_mode
SNAPSHOT_DIR = ARGS.snapshot_dir
MAX_SNAPSHOTS_MEMORY = ARGS.max_snapshots_memory

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

def compute_intrinsic_spark_fields(C, g_xx, g_xy, g_yy):
    """Compute intrinsic spark instability fields (L1)."""
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
        zeros = torch.zeros_like(C)
        return zeros, zeros

    rel_det = abs_detH / (max_abs + eps)

    dC_dx, dC_dy = compute_gradients(C_s)
    det_g, sqrt_g, gxx_inv, gxy_inv, gyy_inv = metric_det_and_inv(g_xx, g_xy, g_yy)
    grad_up = gxx_inv*dC_dx + gxy_inv*dC_dy
    grad_vp = gxy_inv*dC_dx + gyy_inv*dC_dy
    grad_norm = grad_up**2 + grad_vp**2
    max_grad = torch.max(grad_norm)
    if max_grad < 1e-12:
        zeros = torch.zeros_like(C)
        return zeros, zeros

    rel_grad = grad_norm / (max_grad + eps)

    # Hard detector (RC-III style) and soft detector (continuous intrinsic instability field).
    spark_hard = ((rel_det < spark_rel_det_thresh) &
                  (rel_grad > spark_rel_grad_thresh)).to(torch.float32)
    det_score = torch.sigmoid((spark_rel_det_thresh - rel_det) / spark_softness)
    grad_score = torch.sigmoid((rel_grad - spark_rel_grad_thresh) / spark_softness)
    spark_soft = det_score * grad_score
    return spark_hard, spark_soft

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


def compute_intrinsic_collapse_scores(I_tensor, n_active, sqrt_g):
    """Compute intrinsic collapse-instability diagnostics (L1)."""
    if n_active <= 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    mass_weight = sqrt_g.unsqueeze(0)
    masses = torch.sum(I_tensor[:n_active] * mass_weight, dim=(1, 2)) * dx * dy
    relative_margin = (masses - id_min_mass) / (id_min_mass + eps)
    collapse_scores = torch.sigmoid(-relative_margin / 0.25)
    return torch.mean(collapse_scores), torch.max(collapse_scores)


def update_identities(C, I_tensor, n_active, spark_mask, sqrt_g, dt, step, closure_softness_local):
    """Evolve identities with Heun RK2, apply collapse, and conditionally create new ones.
    
    Args:
        C: coherence field [Nx, Ny]
        I_tensor: identity fields tensor [max_identities, Nx, Ny] (padded for unused slots)
        n_active: number of active identities
        spark_mask: spark locations mask [Nx, Ny]
        sqrt_g: metric density factor [Nx, Ny]
        dt: timestep
        step: current step count
        closure_softness_local: per-mode closure softness (0 = hard, 1 = fully soft blend)
    
    Returns:
        I_tensor (updated in-place): identity fields tensor [max_identities, Nx, Ny]
        n_active: updated number of active identities
        I_sum: combined identity richness [Nx, Ny] or None if no active identities
        closure_state: per-step closure diagnostics tensors
    """
    mass_weight = sqrt_g.unsqueeze(0)
    hard_prune_mass = id_min_mass * (1.0 - 0.5 * closure_softness_local)
    spark_threshold = 0.5 - 0.35 * closure_softness_local
    spark_activity_scale = max(1e-6, 1.0 - spark_threshold)
    spark_activity = torch.clamp(spark_mask - spark_threshold, min=0.0)

    # 1. Evolve identities (Heun / RK2) and apply softened collapse near threshold.
    if n_active > 0:
        I_active = I_tensor[:n_active]

        lapI = laplacian_3d(I_active)
        dIdt1 = g_id * C.unsqueeze(0) * I_active - d_id * I_active + D_id * lapI
        I_mid = I_active + 0.5 * dt * dIdt1

        lapI_mid = laplacian_3d(I_mid)
        dIdt2 = g_id * C.unsqueeze(0) * I_mid - d_id * I_mid + D_id * lapI_mid

        I_new = I_active + dt * dIdt2
        torch.clamp_(I_new, min=0.0)

        masses = torch.sum(I_new * mass_weight, dim=(1, 2)) * dx * dy
        if closure_softness_local > 0.0:
            mass_band = id_min_mass * (0.5 + 2.0 * collapse_softness)
            survive_weight = torch.sigmoid((masses - id_min_mass) / (mass_band + eps))
            I_new *= survive_weight.view(-1, 1, 1)
            masses = torch.sum(I_new * mass_weight, dim=(1, 2)) * dx * dy

        alive_mask = masses >= hard_prune_mass
        n_survived_t = alive_mask.sum(dtype=torch.int32)
    else:
        n_survived_t = torch.tensor(0, dtype=torch.int32, device=device)

    # 2. Birth gating with soft factors blended with hard RC-III constraints.
    effective_sparks_t = torch.sum(spark_activity / spark_activity_scale)
    sparks_hard_t = (effective_sparks_t >= spark_birth_sparks_min).to(torch.float32)
    sparks_soft_t = torch.clamp(effective_sparks_t / (spark_birth_sparks_min + eps), max=1.0)
    sparks_factor_t = (1.0 - closure_softness_local) * sparks_hard_t + closure_softness_local * sparks_soft_t

    total_I_mass_t = torch.tensor(0.0, device=device)
    if n_active > 0:
        surviving_mass = torch.sum(I_new[alive_mask] * mass_weight) * dx * dy
        total_I_mass_t = surviving_mass.detach()

    birth_gate_mass_cap = I_birth_gate_fraction * I_global_mass_cap
    mass_hard_t = (total_I_mass_t < birth_gate_mass_cap).to(torch.float32)
    mass_soft_t = torch.clamp((birth_gate_mass_cap - total_I_mass_t) / (birth_gate_mass_cap + eps), min=0.0, max=1.0)
    mass_factor_t = (1.0 - closure_softness_local) * mass_hard_t + closure_softness_local * mass_soft_t

    slots_hard_t = (n_survived_t < max_identities).to(torch.float32)
    slots_soft_t = torch.clamp(
        (max_identities - n_survived_t.to(torch.float32)) / max(1, id_birth_per_step),
        min=0.0,
        max=1.0,
    )
    slots_factor_t = (1.0 - closure_softness_local) * slots_hard_t + closure_softness_local * slots_soft_t

    interval_hard_t = torch.scalar_tensor(float(step % id_birth_interval == 0), device=device)
    if id_birth_interval <= 1:
        interval_soft_t = torch.tensor(1.0, device=device)
    else:
        phase_t = torch.scalar_tensor(float(step % id_birth_interval), device=device)
        interval_t = torch.scalar_tensor(float(id_birth_interval), device=device)
        dist_t = torch.minimum(phase_t, interval_t - phase_t) / (0.5 * interval_t + eps)
        interval_sigma = 0.25 + 0.25 * closure_softness_local
        interval_soft_t = torch.exp(-(dist_t * dist_t) / (2.0 * interval_sigma * interval_sigma))
    interval_factor_t = (1.0 - closure_softness_local) * interval_hard_t + closure_softness_local * interval_soft_t

    birth_score_t = sparks_factor_t * mass_factor_t * slots_factor_t * interval_factor_t
    n_new_cont_t = id_birth_per_step * birth_score_t
    n_new_floor_t = torch.floor(n_new_cont_t).to(torch.int32)
    fractional_t = n_new_cont_t - n_new_floor_t.to(torch.float32)
    n_new_t = n_new_floor_t + (torch.rand((), device=device) < fractional_t).to(torch.int32)
    max_new_t = torch.clamp(
        torch.scalar_tensor(max_identities, device=device, dtype=torch.int32) - n_survived_t,
        min=0,
    )
    n_new_t = torch.minimum(n_new_t, max_new_t)

    spark_weights = spark_activity.reshape(-1)
    n_candidates_t = (spark_weights > 0.0).sum(dtype=torch.int32)

    # Single host sync for prune + birth sizing decisions.
    decision = torch.stack([n_survived_t, n_new_t, n_candidates_t])
    n_survived, n_new, n_candidates = decision.cpu().tolist()

    # Apply prune results.
    if n_active > 0:
        I_tensor[:n_survived] = I_new[alive_mask]
        old_n_active = n_active
        if n_survived < old_n_active:
            I_tensor[n_survived:old_n_active].zero_()

    n_active = n_survived

    # 3. Births sampled from continuous spark intensity field.
    n_births_applied = 0
    if n_new > 0 and n_candidates > 0:
        n_new = min(n_new, n_candidates, max_identities - n_active)
        selected_flat = torch.multinomial(spark_weights, n_new, replacement=False)
        selected_ix = torch.div(selected_flat, Ny, rounding_mode='floor')
        selected_iy = selected_flat % Ny
        I_tensor[n_active:n_active+n_new] = seed_identities_at_batch(selected_ix, selected_iy)
        n_active += n_new
        n_births_applied = n_new

    # 4. Enforce global identity cap and prune tiny identities after cap rescaling.
    if n_active > 0:
        I_tensor[:n_active] = torch.nan_to_num(
            I_tensor[:n_active], nan=0.0, posinf=0.0, neginf=0.0
        )
        total_mass = torch.sum(I_tensor[:n_active] * mass_weight) * dx * dy
        scale = torch.clamp(I_global_mass_cap / (total_mass + eps), max=1.0)
        scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
        I_tensor[:n_active] *= scale

        masses_post = torch.sum(I_tensor[:n_active] * mass_weight, dim=(1, 2)) * dx * dy
        alive_post = masses_post >= hard_prune_mass
        n_post = int(alive_post.sum().item())
        if n_post < n_active:
            I_kept = I_tensor[:n_active][alive_post]
            I_tensor[:n_post] = I_kept
            I_tensor[n_post:n_active].zero_()
            n_active = n_post

    # 3. Combined identity richness - sum over active identities
    if n_active == 0:
        I_sum = None
    else:
        I_sum = torch.sum(I_tensor[:n_active], dim=0)
        torch.nan_to_num(I_sum, nan=0.0, posinf=0.0, neginf=0.0, out=I_sum)

    closure_state = {
        "birth_score_t": birth_score_t,
        "sparks_factor_t": sparks_factor_t,
        "mass_factor_t": mass_factor_t,
        "slots_factor_t": slots_factor_t,
        "interval_factor_t": interval_factor_t,
        "n_survived_t": n_survived_t.to(torch.float32),
        "n_new_t": torch.scalar_tensor(float(n_births_applied), device=device),
        "n_candidates_t": n_candidates_t.to(torch.float32),
    }

    return I_tensor, n_active, I_sum, closure_state

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
    phi = -kappa_grad * laplace_B + dV
    laplace_B_rms = torch.sqrt(torch.mean(laplace_B * laplace_B) + eps)
    dV_rms = torch.sqrt(torch.mean(dV * dV) + eps)
    l_proxy_scalar = torch.sqrt(kappa_grad * kappa_grad * laplace_B_rms * laplace_B_rms + dV_rms * dV_rms + eps)
    return phi, l_proxy_scalar

def compute_flux(C, g_xx, g_xy, g_yy, spark_mask, I_sum):
    phi, l_proxy_scalar = delta_P_over_delta_C(C, g_xx, g_xy, g_yy, spark_mask, I_sum)
    _hook(
        "post_phi",
        phi_rms=torch.sqrt(torch.mean(phi * phi) + eps),
        L_proxy_scalar=l_proxy_scalar,
    )
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
    J_mag_pre = torch.sqrt(Jx * Jx + Jy * Jy + eps)
    _hook(
        "post_J_preclamp",
        J_rms_pre=torch.sqrt(torch.mean(J_mag_pre * J_mag_pre) + eps),
        J_max_pre=torch.max(J_mag_pre),
    )
    torch.clamp_(Jx, min=-10, max=10)
    torch.clamp_(Jy, min=-10, max=10)
    J_mag = torch.sqrt(Jx * Jx + Jy * Jy + eps)
    J_mag_mean = torch.mean(J_mag)
    _hook(
        "post_J_postclamp",
        J_rms=torch.sqrt(torch.mean(J_mag * J_mag) + eps),
        J_max=torch.max(J_mag),
        J_cv=torch.std(J_mag) / (J_mag_mean + eps),
        Jx_field=Jx,
        Jy_field=Jy,
        J_mag_field=J_mag,
    )

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
    grad_sq = dC_dx * dC_dx + dC_dy * dC_dy
    _hook(
        "post_gradC",
        gradC_rms=torch.sqrt(torch.mean(grad_sq) + eps),
        gradC_max=torch.sqrt(torch.max(grad_sq) + eps),
        gradC_dx_field=dC_dx,
        gradC_dy_field=dC_dy,
    )

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
        id_term_xx = torch.zeros_like(C)
        id_term_xy = torch.zeros_like(C)
        id_term_yy = torch.zeros_like(C)

    rb_term_xx = zeta_flux_eff * Jx * Jx
    rb_term_xy = zeta_flux_eff * Jx * Jy
    rb_term_yy = zeta_flux_eff * Jy * Jy
    K_xx = lam_pot*C*g_xx + grad_term_xx + id_term_xx + rb_term_xx
    K_xy = lam_pot*C*g_xy + grad_term_xy + id_term_xy + rb_term_xy
    K_yy = lam_pot*C*g_yy + grad_term_yy + id_term_yy + rb_term_yy
    t_grad_sq = grad_term_xx * grad_term_xx + 2.0 * grad_term_xy * grad_term_xy + grad_term_yy * grad_term_yy
    t_rb_sq = rb_term_xx * rb_term_xx + 2.0 * rb_term_xy * rb_term_xy + rb_term_yy * rb_term_yy
    t_id_sq = id_term_xx * id_term_xx + 2.0 * id_term_xy * id_term_xy + id_term_yy * id_term_yy
    T_grad_rms_t = torch.sqrt(torch.mean(t_grad_sq) + eps)
    T_rb_rms_t = torch.sqrt(torch.mean(t_rb_sq) + eps)
    T_rb_mag = torch.sqrt(t_rb_sq + eps)
    T_rb_mag_mean = torch.mean(T_rb_mag)
    _hook(
        "post_K_raw",
        T_grad_rms=T_grad_rms_t,
        T_rb_rms=T_rb_rms_t,
        T_rb_trace_mean=torch.mean(rb_term_xx + rb_term_yy),
        T_rb_cv=torch.std(T_rb_mag) / (T_rb_mag_mean + eps),
        T_id_rms=torch.sqrt(torch.mean(t_id_sq) + eps),
        rb_vs_grad=T_rb_rms_t / (T_grad_rms_t + eps),
        T_grad_trace_field=(grad_term_xx + grad_term_yy),
        T_rb_trace_field=(rb_term_xx + rb_term_yy),
    )

    detK = K_xx*K_yy - K_xy*K_xy
    detK = torch.where(torch.abs(detK) < DETK_MIN, DETK_MIN, detK)
    _hook("post_K_regularized", detK_min=torch.min(detK), detK_mean=torch.mean(detK))
    inv_detK = 1.0 / detK

    g_new_xx =  K_yy * inv_detK
    g_new_xy = -K_xy * inv_detK
    g_new_yy =  K_xx * inv_detK

    torch.nan_to_num(g_new_xx, nan=1.0, out=g_new_xx)
    torch.nan_to_num(g_new_xy, nan=0.0, out=g_new_xy)
    torch.nan_to_num(g_new_yy, nan=1.0, out=g_new_yy)
    _hook(
        "post_g_preblend",
        g_new_rms=torch.sqrt(
            torch.mean(g_new_xx * g_new_xx + 2.0 * g_new_xy * g_new_xy + g_new_yy * g_new_yy) + eps
        ),
    )

    blend = 0.05
    g_xx = (1-blend)*g_xx + blend*g_new_xx
    g_xy = (1-blend)*g_xy + blend*g_new_xy
    g_yy = (1-blend)*g_yy + blend*g_new_yy

    g_xx, g_xy, g_yy, det_g_reg = regularise_metric(g_xx, g_xy, g_yy)
    _hook(
        "post_g_postblend",
        detg_min=torch.min(det_g_reg),
        detg_mean=torch.mean(det_g_reg),
        G_frob_field=torch.sqrt(g_xx * g_xx + 2.0 * g_xy * g_xy + g_yy * g_yy + eps),
    )
    return g_xx, g_xy, g_yy

# ============================================================
# COHERENCE RHS & STEP
# ============================================================
def rhs_C(C, g_xx, g_xy, g_yy, spark_mask, I_sum):
    Jx, Jy, v_up, v_vp, sqrt_g = compute_flux(C, g_xx, g_xy, g_yy, spark_mask, I_sum)
    dCdt_flux = covariant_divergence(Jx, Jy, g_xx, g_xy, g_yy)
    _hook("post_divergence", dCdt_flux_rms=torch.sqrt(torch.mean(dCdt_flux * dCdt_flux) + eps))
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

    # First projection enforces mass on the pre-update metric.
    C_new = project_to_invariant(C_new, g_xx, g_xy, g_yy, target_mass)
    g_xx, g_xy, g_yy = update_metric_from_K(C_new, Jx2, Jy2, g_xx, g_xy, g_yy, I_sum)
    # Re-project after metric update so reported mass stays invariant on the final metric.
    C_new = project_to_invariant(C_new, g_xx, g_xy, g_yy, target_mass)

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
blob_specs = load_blob_specs("configs/blobs.json")
domain_x_max = dx * max(Nx - 1, 1)
domain_y_max = dy * max(Ny - 1, 1)
# Integer index grids for identity seeding (use these in seed_identity_at to match ix/iy indices)
X_idx = torch.arange(Nx, device=device)
Y_idx = torch.arange(Ny, device=device)
Xg_idx, Yg_idx = torch.meshgrid(X_idx, Y_idx, indexing='ij')

# Initialize coherence field with Gaussian blobs + noise
C = torch.zeros(Nx, Ny, device=device)
for blob in blob_specs:
    C += gaussian_blob(
        Xg,
        Yg,
        blob["x"] * domain_x_max,
        blob["y"] * domain_y_max,
        blob["sigma"],
    )
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
# Identity mass cap follows a fraction of structured coherence mass.
structured_C0 = torch.clamp(C - coherence_bg_floor, min=0.0)
structured_mass0 = torch.sum(structured_C0 * sqrt_g0) * dx * dy
structured_mass0_value = float(structured_mass0.item())
I_global_mass_cap = I_cap_fraction * structured_mass0_value
print(f"Initial coherence mass (metric-weighted): {target_mass0_value:.4f}")
print(
    f"Initial structured coherence mass (C>{coherence_bg_floor:.3f}): "
    f"{structured_mass0_value:.4f}"
)
print(
    f"Identity mass cap: {I_global_mass_cap:.4f} "
    f"(fraction={I_cap_fraction:.3f}, birth_gate={I_birth_gate_fraction:.3f}, "
    f"closure_mode={closure_mode}, events_control_in_core={events_control_in_core}, "
    f"closure_softness={closure_softness:.2f})"
)

# Identity fields tensor [max_identities, Nx, Ny] - vectorized storage (padded with zeros)
I_tensor = torch.zeros(max_identities, Nx, Ny, device=device)
n_active = 0  # number of active identities
I_sum = None
zero_spark_mask = torch.zeros_like(C)

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
            'negctrl_lam_scale': negctrl_lam_scale,
            'xi_grad': xi_grad,
            'zeta_flux': zeta_flux,
            'zeta_flux_eff': zeta_flux_eff,
            'beta_rb': readback_beta_rb,
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
            'I_cap_fraction': I_cap_fraction,
            'coherence_bg_floor': coherence_bg_floor,
            'I_birth_gate_fraction': I_birth_gate_fraction,
            'closure_mode': closure_mode,
            'events_control_in_core': events_control_in_core,
            'closure_softness': closure_softness,
            'spark_softness': spark_softness,
            'collapse_softness': collapse_softness,
            'activity': float(ARGS.activity) if ARGS.activity is not None else None,
            'initial_structured_coherence_mass': structured_mass0_value,
            'spark_rel_det_thresh': spark_rel_det_thresh,
            'spark_rel_grad_thresh': spark_rel_grad_thresh,
            'l1_metrics': [
                'spark_score_mean',
                'spark_score_max',
                'collapse_score_mean',
                'collapse_score_max',
            ],
            'closure_metrics': [
                'closure_birth_score',
                'closure_sparks_factor',
                'closure_mass_factor',
                'closure_slots_factor',
                'closure_interval_factor',
                'closure_births',
            ],
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
                    dt_val=None, mass_val=None, I_mass_val=None, n_active_val=None,
                    spark_score_mean_val=None, spark_score_max_val=None,
                    collapse_score_mean_val=None, collapse_score_max_val=None,
                    closure_birth_score_val=None, closure_sparks_factor_val=None,
                    closure_mass_factor_val=None, closure_slots_factor_val=None,
                    closure_interval_factor_val=None, closure_births_val=None):
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
            if spark_score_mean_val is not None:
                self.diagnostics.setdefault('spark_score_mean', []).append(spark_score_mean_val)
            if spark_score_max_val is not None:
                self.diagnostics.setdefault('spark_score_max', []).append(spark_score_max_val)
            if collapse_score_mean_val is not None:
                self.diagnostics.setdefault('collapse_score_mean', []).append(collapse_score_mean_val)
            if collapse_score_max_val is not None:
                self.diagnostics.setdefault('collapse_score_max', []).append(collapse_score_max_val)
            if closure_birth_score_val is not None:
                self.diagnostics.setdefault('closure_birth_score', []).append(closure_birth_score_val)
            if closure_sparks_factor_val is not None:
                self.diagnostics.setdefault('closure_sparks_factor', []).append(closure_sparks_factor_val)
            if closure_mass_factor_val is not None:
                self.diagnostics.setdefault('closure_mass_factor', []).append(closure_mass_factor_val)
            if closure_slots_factor_val is not None:
                self.diagnostics.setdefault('closure_slots_factor', []).append(closure_slots_factor_val)
            if closure_interval_factor_val is not None:
                self.diagnostics.setdefault('closure_interval_factor', []).append(closure_interval_factor_val)
            if closure_births_val is not None:
                self.diagnostics.setdefault('closure_births', []).append(closure_births_val)
        
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
                     n_active=int(n_active_val) if n_active_val is not None else 0,
                     spark_score_mean=float(spark_score_mean_val) if spark_score_mean_val is not None else 0.0,
                     spark_score_max=float(spark_score_max_val) if spark_score_max_val is not None else 0.0,
                     collapse_score_mean=float(collapse_score_mean_val) if collapse_score_mean_val is not None else 0.0,
                     collapse_score_max=float(collapse_score_max_val) if collapse_score_max_val is not None else 0.0,
                     closure_birth_score=float(closure_birth_score_val) if closure_birth_score_val is not None else 0.0,
                     closure_sparks_factor=float(closure_sparks_factor_val) if closure_sparks_factor_val is not None else 0.0,
                     closure_mass_factor=float(closure_mass_factor_val) if closure_mass_factor_val is not None else 0.0,
                     closure_slots_factor=float(closure_slots_factor_val) if closure_slots_factor_val is not None else 0.0,
                     closure_interval_factor=float(closure_interval_factor_val) if closure_interval_factor_val is not None else 0.0,
                     closure_births=float(closure_births_val) if closure_births_val is not None else 0.0)
    
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
            diagnostics = {
                'dt': [],
                'mass': [],
                'I_mass': [],
                'ids': [],
                'spark_score_mean': [],
                'spark_score_max': [],
                'collapse_score_mean': [],
                'collapse_score_max': [],
                'closure_birth_score': [],
                'closure_sparks_factor': [],
                'closure_mass_factor': [],
                'closure_slots_factor': [],
                'closure_interval_factor': [],
                'closure_births': [],
            }
            
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
                diagnostics['spark_score_mean'].append(float(data['spark_score_mean']) if 'spark_score_mean' in data else 0.0)
                diagnostics['spark_score_max'].append(float(data['spark_score_max']) if 'spark_score_max' in data else 0.0)
                diagnostics['collapse_score_mean'].append(float(data['collapse_score_mean']) if 'collapse_score_mean' in data else 0.0)
                diagnostics['collapse_score_max'].append(float(data['collapse_score_max']) if 'collapse_score_max' in data else 0.0)
                diagnostics['closure_birth_score'].append(float(data['closure_birth_score']) if 'closure_birth_score' in data else 0.0)
                diagnostics['closure_sparks_factor'].append(float(data['closure_sparks_factor']) if 'closure_sparks_factor' in data else 0.0)
                diagnostics['closure_mass_factor'].append(float(data['closure_mass_factor']) if 'closure_mass_factor' in data else 0.0)
                diagnostics['closure_slots_factor'].append(float(data['closure_slots_factor']) if 'closure_slots_factor' in data else 0.0)
                diagnostics['closure_interval_factor'].append(float(data['closure_interval_factor']) if 'closure_interval_factor' in data else 0.0)
                diagnostics['closure_births'].append(float(data['closure_births']) if 'closure_births' in data else 0.0)
            
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

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=ARGS.animate_interval, blit=False)
    plt.tight_layout()

    output_file = 'simulation_output.mp4'
    try:
        anim.save(output_file, writer='ffmpeg', fps=ARGS.fps)
        print(f"✓ Animation saved: {output_file}")
    except Exception as e:
        print(f"FFmpeg not available ({e}), saving as GIF...")
        anim.save('simulation_output.gif', writer='pillow', fps=ARGS.fps)
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
    spark_score_mean_vals, spark_score_max_vals = [], []
    collapse_score_mean_vals, collapse_score_max_vals = [], []

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
        spark_score_mean_vals.append(float(data['spark_score_mean']) if 'spark_score_mean' in data else 0.0)
        spark_score_max_vals.append(float(data['spark_score_max']) if 'spark_score_max' in data else 0.0)
        collapse_score_mean_vals.append(float(data['collapse_score_mean']) if 'collapse_score_mean' in data else 0.0)
        collapse_score_max_vals.append(float(data['collapse_score_max']) if 'collapse_score_max' in data else 0.0)

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
fig = None
imC = imG = imS = quiv = None
line_dt = line_mass = line_Imass = line_ids = title = None
ax_dt = ax_mass = ax_ids = None
viewer = None

if not HEADLESS_MODE:
    if USE_VISPY:
        viewer = RCVispyViewer(Nx, Ny, C_min, C_max, title="RC-PDE v15 CUDA")
    else:
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
        quiv = axC.quiver(
            Xq, Yq,
            np.zeros_like(Xq, dtype=float),
            np.zeros_like(Xq, dtype=float),
            angles='xy',
            scale_units='xy',
            scale=1.0,
            width=0.002
        )

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
    # Initialize snapshot manager for headless mode
    snap_mgr = SnapshotManager(storage_mode=STORAGE_MODE, snapshot_dir=SNAPSHOT_DIR)

# History for diagnostics (collected on sampled frames)
dt_hist = []
mass_hist = []
ids_hist = []
I_mass_hist = []
closure_birth_score_hist = []
closure_sparks_factor_hist = []
closure_mass_factor_hist = []
closure_slots_factor_hist = []
closure_interval_factor_hist = []
closure_births_hist = []

step_counter = 0
last_dt = 0.0
last_mass = float(target_mass0_value)
last_I_mass = 0.0
last_collapse_score_mean = 0.0
last_collapse_score_max = 0.0
last_event_state = {
    "spark_score_mean_t": torch.tensor(0.0, device=device),
    "spark_score_max_t": torch.tensor(0.0, device=device),
}
last_closure_state = {
    "birth_score_t": torch.tensor(0.0, device=device),
    "sparks_factor_t": torch.tensor(0.0, device=device),
    "mass_factor_t": torch.tensor(0.0, device=device),
    "slots_factor_t": torch.tensor(0.0, device=device),
    "interval_factor_t": torch.tensor(0.0, device=device),
    "n_new_t": torch.tensor(0.0, device=device),
}

# ============================================================
# SIMULATION STEP
# ============================================================
def step_events(C, g_xx, g_xy, g_yy, step):
    """Event phase routing for v15 modes (L1 diagnostics + optional control signal)."""
    spark_hard, spark_soft = compute_intrinsic_spark_fields(C, g_xx, g_xy, g_yy)
    spark_mask_soft = (1.0 - closure_softness) * spark_hard + closure_softness * spark_soft
    if closure_mode == "off":
        spark_mask = spark_mask_soft if events_control_in_core else zero_spark_mask
    elif closure_mode == "full":
        spark_mask = spark_hard
    else:
        spark_mask = spark_mask_soft
    event_state = {
        "spark_score_mean_t": torch.mean(spark_soft),
        "spark_score_max_t": torch.max(spark_soft),
        "spark_pixels_t": torch.sum((spark_soft > 0.5).to(torch.int32)),
    }
    return spark_mask, event_state


def step_core(C, g_xx, g_xy, g_yy, target_mass, spark_mask, I_sum):
    """Iteration-1 scaffold: reflexive core phase."""
    return rk2_step(C, g_xx, g_xy, g_yy, target_mass, spark_mask, I_sum)


def step_closure(C, I_tensor, n_active, spark_mask, sqrt_g, dt, step):
    """Closure phase routing for v15 modes."""
    if closure_mode == "off":
        # Explicitly keep closure side effects disabled in core-only runs.
        if n_active > 0:
            I_tensor[:n_active].zero_()
        zero = torch.tensor(0.0, device=device)
        closure_state = {
            "birth_score_t": zero,
            "sparks_factor_t": zero,
            "mass_factor_t": zero,
            "slots_factor_t": zero,
            "interval_factor_t": zero,
            "n_survived_t": zero,
            "n_new_t": zero,
            "n_candidates_t": zero,
        }
        return I_tensor, 0, None, closure_state
    if closure_mode == "soft":
        return update_identities(C, I_tensor, n_active, spark_mask, sqrt_g, dt, step, closure_softness)
    if closure_mode == "full":
        # Full mode keeps hard RC-III-style closure gates.
        return update_identities(C, I_tensor, n_active, spark_mask, sqrt_g, dt, step, 0.0)
    raise ValueError(f"Unsupported closure mode: {closure_mode}")


def update(frame):
    global C, g_xx, g_xy, g_yy, I_tensor, n_active, I_sum, step_counter
    global last_dt, last_mass, last_I_mass, last_collapse_score_mean, last_collapse_score_max
    global last_event_state, last_closure_state

    sampled_frame = (frame % RENDER_INTERVAL == 0)
    render_now = (not HEADLESS_MODE) and sampled_frame
    
    # Increment counter first, then compute snapshot trigger for headless mode
    step_counter += 1
    snapshot_frame = (step_counter % SNAPSHOT_INTERVAL == 0) if HEADLESS_MODE else False

    spark_mask, event_state = step_events(C, g_xx, g_xy, g_yy, step_counter)
    core_I_sum = None if closure_mode == "off" else I_sum

    C, g_xx, g_xy, g_yy, dt, vu, vv = step_core(
        C, g_xx, g_xy, g_yy, target_mass0_value, spark_mask, core_I_sum
    )
    _hook("post_core_pre_closure", dt=dt, C_field=C)

    det_g, sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
    I_tensor, n_active, I_sum, closure_state = step_closure(
        C, I_tensor, n_active, spark_mask, sqrt_g, dt, step_counter
    )
    _hook("post_closure", n_active=n_active)
    if readback_hooks.enabled:
        missing = readback_hooks.missing_for_step(step_counter, include_closure=True)
        if missing and len(hook_missing_steps) < 20:
            hook_missing_steps.append((step_counter, missing))
    mass = torch.sum(C * sqrt_g) * dx * dy
    collapse_score_mean_t, collapse_score_max_t = compute_intrinsic_collapse_scores(I_tensor, n_active, sqrt_g)

    # Identity diagnostics
    if closure_mode != "off" and I_sum is not None:
        I_mass = torch.sum(I_sum * sqrt_g) * dx * dy
    else:
        I_mass = 0.0
    I_mass_value = I_mass.item() if isinstance(I_mass, torch.Tensor) else float(I_mass)

    last_dt = float(dt)
    last_mass = float(mass.item())
    last_I_mass = I_mass_value
    last_collapse_score_mean = float(collapse_score_mean_t.item())
    last_collapse_score_max = float(collapse_score_max_t.item())
    last_event_state = event_state
    last_closure_state = closure_state

    # Snapshot storage - immediate CPU transfer + optional disk stream
    if HEADLESS_MODE and snapshot_frame:
        snap_mgr.add_snapshot(
            step=step_counter,
            C=C, sqrt_g=sqrt_g, spark_mask=spark_mask,
            dt_val=dt, mass_val=mass.item(),
            I_mass_val=I_mass_value,
            n_active_val=n_active,
            spark_score_mean_val=event_state["spark_score_mean_t"].item(),
            spark_score_max_val=event_state["spark_score_max_t"].item(),
            collapse_score_mean_val=collapse_score_mean_t.item(),
            collapse_score_max_val=collapse_score_max_t.item(),
            closure_birth_score_val=closure_state["birth_score_t"].item(),
            closure_sparks_factor_val=closure_state["sparks_factor_t"].item(),
            closure_mass_factor_val=closure_state["mass_factor_t"].item(),
            closure_slots_factor_val=closure_state["slots_factor_t"].item(),
            closure_interval_factor_val=closure_state["interval_factor_t"].item(),
            closure_births_val=closure_state["n_new_t"].item(),
        )

    # Update diagnostics on sampled frames (rendering) or snapshot frames (headless)
    if (not HEADLESS_MODE and sampled_frame) or (HEADLESS_MODE and snapshot_frame):
        dt_hist.append(dt)
        mass_hist.append(mass.item())
        ids_hist.append(n_active)
        I_mass_hist.append(I_mass_value)
        closure_birth_score_hist.append(closure_state["birth_score_t"].item())
        closure_sparks_factor_hist.append(closure_state["sparks_factor_t"].item())
        closure_mass_factor_hist.append(closure_state["mass_factor_t"].item())
        closure_slots_factor_hist.append(closure_state["slots_factor_t"].item())
        closure_interval_factor_hist.append(closure_state["interval_factor_t"].item())
        closure_births_hist.append(closure_state["n_new_t"].item())

    # Update visualizations only on render frames
    if render_now:
        if USE_VISPY and viewer is not None:
            viewer.update_images(
                C.cpu().detach().numpy(),
                sqrt_g.cpu().detach().numpy(),
                spark_mask.cpu().detach().numpy()
            )
            viewer.update_diagnostics(
                float(dt),
                float(mass.item()),
                float(I_mass.item() if isinstance(I_mass, torch.Tensor) else I_mass),
                int(n_active)
            )
        else:
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
            vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
            vy = np.nan_to_num(vy, nan=0.0, posinf=0.0, neginf=0.0)
            quiv.set_UVC(vx, vy)

    n = len(dt_hist)

    # Update plot lines and title only on render frames (matplotlib path)
    if render_now and (not USE_VISPY):
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

    if USE_VISPY and not HEADLESS_MODE:
        # VisPy timer callback does not use a return value.
        return []

    return [imC, imG, imS, quiv, line_dt, line_mass, line_Imass, line_ids, title]

if not HEADLESS_MODE:
    if USE_VISPY:
        # VisPy handles its own event loop; keep stepping via the update callback
        from vispy import app
        timer = app.Timer(interval=0, connect=lambda ev: update(ev.count), start=True)
        app.run()
    else:
        anim = FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
else:
    # Save initial state as step 0 so headless snapshots are 0, N, 2N, ...
    if snap_mgr.has_data() is False:
        initial_spark_mask, initial_event_state = step_events(C, g_xx, g_xy, g_yy, step=0)
        initial_mass = torch.sum(C * sqrt_g0) * dx * dy
        initial_I_mass = 0.0
        initial_collapse_score_mean_t, initial_collapse_score_max_t = compute_intrinsic_collapse_scores(I_tensor, n_active, sqrt_g0)
        zero_closure = 0.0
        snap_mgr.add_snapshot(
            step=0,
            C=C, sqrt_g=sqrt_g0, spark_mask=initial_spark_mask,
            dt_val=0.0, mass_val=float(initial_mass.item()),
            I_mass_val=initial_I_mass, n_active_val=n_active,
            spark_score_mean_val=float(initial_event_state["spark_score_mean_t"].item()),
            spark_score_max_val=float(initial_event_state["spark_score_max_t"].item()),
            collapse_score_mean_val=float(initial_collapse_score_mean_t.item()),
            collapse_score_max_val=float(initial_collapse_score_max_t.item()),
            closure_birth_score_val=zero_closure,
            closure_sparks_factor_val=zero_closure,
            closure_mass_factor_val=zero_closure,
            closure_slots_factor_val=zero_closure,
            closure_interval_factor_val=zero_closure,
            closure_births_val=zero_closure,
        )
        # Keep diagnostic histories aligned with snapshots from step 0
        dt_hist.append(0.0)
        mass_hist.append(float(initial_mass.item()))
        ids_hist.append(n_active)
        I_mass_hist.append(initial_I_mass)

    for frame in range(HEADLESS_STEPS):
        update(frame)

    if step_counter > 0 and (not snap_mgr.steps or snap_mgr.steps[-1] != step_counter):
        final_spark_mask = step_events(C, g_xx, g_xy, g_yy, step_counter)[0]
        _, final_sqrt_g, *_ = metric_det_and_inv(g_xx, g_xy, g_yy)
        snap_mgr.add_snapshot(
            step=step_counter,
            C=C, sqrt_g=final_sqrt_g, spark_mask=final_spark_mask,
            dt_val=last_dt, mass_val=last_mass,
            I_mass_val=last_I_mass, n_active_val=n_active,
            spark_score_mean_val=float(last_event_state["spark_score_mean_t"].item()),
            spark_score_max_val=float(last_event_state["spark_score_max_t"].item()),
            collapse_score_mean_val=last_collapse_score_mean,
            collapse_score_max_val=last_collapse_score_max,
            closure_birth_score_val=float(last_closure_state["birth_score_t"].item()),
            closure_sparks_factor_val=float(last_closure_state["sparks_factor_t"].item()),
            closure_mass_factor_val=float(last_closure_state["mass_factor_t"].item()),
            closure_slots_factor_val=float(last_closure_state["slots_factor_t"].item()),
            closure_interval_factor_val=float(last_closure_state["interval_factor_t"].item()),
            closure_births_val=float(last_closure_state["n_new_t"].item()),
        )
        dt_hist.append(last_dt)
        mass_hist.append(last_mass)
        ids_hist.append(n_active)
        I_mass_hist.append(last_I_mass)

    if dt_hist:
        print(
            f"[HEADLESS DONE] steps={step_counter}  dt={dt_hist[-1]:.3e}  "
            f"mass={mass_hist[-1]:.4f}  I_mass={I_mass_hist[-1]:.4f}  ids={ids_hist[-1]}"
        )
        if readback_hooks.enabled:
            hook_summary = readback_hooks.summary()
            print(
                f"[HOOKS] records={hook_summary['records']}  steps={hook_summary['steps']}  "
                f"missing_steps={len(hook_missing_steps)}"
            )
            if hook_missing_steps:
                print(f"[HOOKS] first_missing={hook_missing_steps[0]}")

    # Create animation from saved snapshots
    if snap_mgr.has_data():
        print(f"[ANIMATION] Processing {len(snap_mgr.steps)} snapshots...")
        if STORAGE_MODE == 'disk':
            create_animation_streaming(SNAPSHOT_DIR, fps=ARGS.fps, interval=ARGS.animate_interval, two_pass=ARGS.two_pass)
        else:
            snapshot_data = snap_mgr.get_all()
            create_animation_from_snapshots(snapshot_data)
        snap_mgr.cleanup()

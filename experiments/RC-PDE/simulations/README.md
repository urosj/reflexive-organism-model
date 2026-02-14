# Simulations Layout

Canonical simulator paths live under:
- `simulations/active/`: maintained CUDA lanes (`simulation-v12-cuda.py` to `simulation-v16-cuda.py`).
- `simulations/legacy/`: historical/archival lanes kept for provenance.
- `simulations/helpers/`: shared runtime helpers (for example `vispy_viewer.py`).

Path policy:
- Use canonical paths only (for docs, scripts, and automation).
- Root-level `simulation-*.py` aliases were removed.

Quick run examples:
- v16 help:
  - `python simulations/active/simulation-v16-cuda.py --help`
- v16 best-feature interactive run:
  - `python simulations/active/simulation-v16-cuda.py --closure-mode soft --nonlocal-mode on --operator-diagnostics --operator-diagnostics-interval 10 --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50`
- v16 best-feature headless run:
  - `python simulations/active/simulation-v16-cuda.py --headless --headless-steps 2000 --nx 1024 --ny 1024 --dx 0.1 --seed 1 --storage-mode disk --snapshot-interval 50 --closure-mode soft --nonlocal-mode on --operator-diagnostics --operator-diagnostics-interval 10 --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50`
- v15 headless baseline:
  - `python simulations/active/simulation-v15-cuda.py --headless --headless-steps 2000 --nx 1024 --ny 1024 --dx 0.1 --seed 1 --storage-mode disk --snapshot-interval 50 --closure-mode soft`

Notes:
- `simulations/active/simulation-v12-cuda.py` through `simulations/active/simulation-v16-cuda.py` are CUDA-preferred and fall back to CPU when CUDA is unavailable.
- CPU fallback works, but large runs are much slower than CUDA.

import os
import json
from pathlib import Path


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_config_path(config_name: str | Path) -> Path:
    raw = Path(config_name)
    if raw.is_absolute():
        return raw

    module_path = Path(__file__).resolve()
    repo_root = module_path.parents[1]
    candidates = [Path.cwd() / raw]
    if raw.parent == Path("."):
        candidates.append(module_path.with_name(raw.name))
    candidates.append(repo_root / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Fall back to module-local filename behavior for clearer error locality.
    return module_path.with_name(raw.name)


def load_blob_specs(config_name="blobs.json", *, verbose=True, debug=None):
    """Load and validate Gaussian blob specs from a JSON file."""
    config_path = _resolve_config_path(config_name)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    blobs = data.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError(f"{config_path} must contain a non-empty 'blobs' list")

    parsed = []
    for idx, blob in enumerate(blobs):
        if not isinstance(blob, dict):
            raise ValueError(f"{config_path}: blob #{idx} must be an object")
        blob_id = blob.get("id")
        if not isinstance(blob_id, str) or not blob_id:
            raise ValueError(f"{config_path}: blob #{idx} must include non-empty string 'id'")

        x = blob.get("x")
        y = blob.get("y")
        sigma = blob.get("sigma")
        if not isinstance(x, (int, float)):
            raise ValueError(f"{config_path}: blob '{blob_id}' has invalid 'x'")
        if not isinstance(y, (int, float)):
            raise ValueError(f"{config_path}: blob '{blob_id}' has invalid 'y'")
        if not isinstance(sigma, (int, float)):
            raise ValueError(f"{config_path}: blob '{blob_id}' has invalid 'sigma'")
        if not (0.0 <= float(x) <= 1.0):
            raise ValueError(f"{config_path}: blob '{blob_id}' must have x in [0, 1]")
        if not (0.0 <= float(y) <= 1.0):
            raise ValueError(f"{config_path}: blob '{blob_id}' must have y in [0, 1]")
        if float(sigma) <= 0.0:
            raise ValueError(f"{config_path}: blob '{blob_id}' must have sigma > 0")

        parsed.append(
            {
                "id": blob_id,
                "x": float(x),
                "y": float(y),
                "sigma": float(sigma),
            }
        )

    if verbose:
        print(f"[INIT] loaded {len(parsed)} blobs from {config_path.name}")

    debug_enabled = _env_truthy("RC_BLOB_DEBUG") if debug is None else bool(debug)
    if debug_enabled:
        for blob in parsed:
            print(
                "[INIT][blob] "
                f"id={blob['id']} x={blob['x']:.6f} y={blob['y']:.6f} sigma={blob['sigma']:.6f}"
            )

    return parsed

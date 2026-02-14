"""Canonical telemetry contract for Paper-14 read-back auditing.

This module defines the shared `ExperienceSample` schema for v12-v16 lanes.
It includes:
- key/nullability/version contract
- alias normalization
- lightweight record validation for offline tooling
"""

from __future__ import annotations

from typing import Dict, List, Mapping, MutableMapping


SCHEMA_NAME = "ExperienceSample"
SCHEMA_VERSION = "14.1.0"


# Fallback policy for adapters:
# - non-null keys MUST be emitted with a concrete value
# - nullable keys MAY be emitted as None when the source version cannot provide them
# - optional keys MAY be omitted entirely
KEY_SPECS: Dict[str, Dict[str, object]] = {
    # Meta / identity
    "schema_version": {"dtype": "str", "required": True, "nullable": False},
    "sim_version": {"dtype": "str", "required": True, "nullable": False},
    "step": {"dtype": "int", "required": True, "nullable": False},
    "phase": {"dtype": "str", "required": True, "nullable": False},
    "dt": {"dtype": "float", "required": True, "nullable": False},
    "nx": {"dtype": "int", "required": True, "nullable": False},
    "ny": {"dtype": "int", "required": True, "nullable": False},
    "dx": {"dtype": "float", "required": True, "nullable": False},
    "seed": {"dtype": "int", "required": True, "nullable": True},
    # Read-back core
    "J_rms": {"dtype": "float", "required": True, "nullable": False},
    "J_max": {"dtype": "float", "required": True, "nullable": False},
    "J_cv": {"dtype": "float", "required": True, "nullable": True},
    "T_rb_rms": {"dtype": "float", "required": True, "nullable": False},
    "T_rb_trace_mean": {"dtype": "float", "required": True, "nullable": False},
    "T_rb_cv": {"dtype": "float", "required": True, "nullable": True},
    "gradC_rms": {"dtype": "float", "required": True, "nullable": False},
    "gradC_max": {"dtype": "float", "required": True, "nullable": False},
    "T_grad_rms": {"dtype": "float", "required": True, "nullable": False},
    "T_id_rms": {"dtype": "float", "required": False, "nullable": True},
    "T_den_rms": {"dtype": "float", "required": False, "nullable": True},
    "rb_vs_grad": {"dtype": "float", "required": True, "nullable": False},
    "rb_vs_den": {"dtype": "float", "required": True, "nullable": True},
    # Context
    "closure_mode": {"dtype": "str", "required": True, "nullable": True},
    "n_active": {"dtype": "int", "required": True, "nullable": True},
    "I_mass": {"dtype": "float", "required": True, "nullable": True},
    "spark_score_mean": {"dtype": "float", "required": True, "nullable": True},
    "spark_score_max": {"dtype": "float", "required": True, "nullable": True},
    "closure_births": {"dtype": "int", "required": True, "nullable": True},
}


# Canonical aliasing for scorer compatibility:
# if legacy `E_*` terms are present, map them to `T_*_rms` semantics.
ALIASES = {
    "E_den": "T_den_rms",
    "E_grad": "T_grad_rms",
    "E_id": "T_id_rms",
    "E_rb": "T_rb_rms",
}


def required_keys() -> List[str]:
    """Return keys that must appear in each record."""
    return [k for k, spec in KEY_SPECS.items() if bool(spec["required"])]


def nullable_keys() -> List[str]:
    """Return keys allowed to be None."""
    return [k for k, spec in KEY_SPECS.items() if bool(spec["nullable"])]


def normalize_aliases(record: MutableMapping[str, object]) -> MutableMapping[str, object]:
    """Map legacy aliases (E_*) to canonical keys in-place if canonical is absent."""
    for old_key, canonical_key in ALIASES.items():
        if old_key in record and canonical_key not in record:
            record[canonical_key] = record[old_key]
    return record


def _is_valid_type(dtype: str, value: object) -> bool:
    if dtype == "str":
        return isinstance(value, str)
    if dtype == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if dtype == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def validate_record(
    record: Mapping[str, object],
    *,
    strict_schema_version: bool = True,
) -> List[str]:
    """Validate one telemetry record against KEY_SPECS.

    Returns a list of validation errors. Empty list means valid.
    """
    errors: List[str] = []
    normalized: Dict[str, object] = dict(record)
    normalize_aliases(normalized)

    if strict_schema_version:
        version = normalized.get("schema_version")
        if version != SCHEMA_VERSION:
            errors.append(
                f"schema_version mismatch: expected {SCHEMA_VERSION!r}, got {version!r}"
            )

    for key, spec in KEY_SPECS.items():
        required = bool(spec["required"])
        nullable = bool(spec["nullable"])
        dtype = str(spec["dtype"])

        if key not in normalized:
            if required:
                errors.append(f"missing required key: {key}")
            continue

        value = normalized[key]
        if value is None:
            if not nullable:
                errors.append(f"non-nullable key is None: {key}")
            continue

        if not _is_valid_type(dtype, value):
            errors.append(
                f"type mismatch for {key}: expected {dtype}, got {type(value).__name__}"
            )

    return errors


def validate_records(
    records: List[Mapping[str, object]],
    *,
    strict_schema_version: bool = True,
) -> List[str]:
    """Validate multiple records and return flattened errors with indices."""
    all_errors: List[str] = []
    for idx, record in enumerate(records):
        errors = validate_record(record, strict_schema_version=strict_schema_version)
        for err in errors:
            all_errors.append(f"record[{idx}]: {err}")
    return all_errors

"""In-memory hook recorder for Paper-14 runtime instrumentation.

Iteration 2.1 target:
- add canonical hook points in simulation loops
- keep data in-memory only (no telemetry persistence yet)
"""

from __future__ import annotations

from collections import defaultdict, deque
import json
import os
import time
import numpy as np
from typing import Any, Deque, Dict, List, Optional, Set


CANONICAL_HOOK_POINTS: List[str] = [
    "post_gradC",
    "post_phi",
    "post_J_preclamp",
    "post_J_postclamp",
    "post_K_raw",
    "post_K_regularized",
    "post_g_preblend",
    "post_g_postblend",
    "post_divergence",
    "post_core_pre_closure",
    "post_closure",
]


def _scalarize(value: Any) -> Any:
    """Best-effort scalar conversion without retaining tensor payloads."""
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return None


class InMemoryHookRecorder:
    """Collects hook events and stage coverage by step in memory."""

    def __init__(self, *, enabled: bool = True, max_records: int = 4096):
        self.enabled = enabled
        self.records: Deque[Dict[str, Any]] = deque(maxlen=max_records)
        self.stage_counts: Dict[str, int] = defaultdict(int)
        self.stages_by_step: Dict[int, Set[str]] = defaultdict(set)
        self._jsonl_handle = None
        self._field_dump_enabled = False
        self._field_dump_npz_path: Optional[str] = None
        self._field_dump_downsample = 4
        self._field_dump_every_n_steps = 10
        self._field_dump_stages: Set[str] = {
            "post_gradC",
            "post_J_postclamp",
            "post_K_raw",
            "post_g_postblend",
            "post_core_pre_closure",
        }
        self._field_records: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._field_steps: Dict[str, List[int]] = defaultdict(list)

    def enable_jsonl(self, path: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Enable streaming of hook records to JSONL."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._jsonl_handle = open(path, "w", encoding="utf-8")
        if metadata is not None:
            header = {"record_type": "hook_stream_metadata", **metadata}
            self._jsonl_handle.write(json.dumps(header) + "\n")
            self._jsonl_handle.flush()

    def enable_field_dump(
        self,
        npz_path: str,
        *,
        downsample: int = 4,
        every_n_steps: int = 10,
        stages: Optional[Set[str]] = None,
    ) -> None:
        """Enable sparse downsampled field persistence to NPZ."""
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        self._field_dump_enabled = True
        self._field_dump_npz_path = npz_path
        self._field_dump_downsample = max(1, int(downsample))
        self._field_dump_every_n_steps = max(1, int(every_n_steps))
        if stages is not None:
            self._field_dump_stages = set(stages)

    def emit(self, stage: str, *, step: Optional[int] = None, **payload: Any) -> None:
        if not self.enabled:
            return
        record: Dict[str, Any] = {"stage": stage}
        if step is not None:
            step_int = int(step)
            record["step"] = step_int
            self.stages_by_step[step_int].add(stage)
        for key, value in payload.items():
            record[key] = _scalarize(value)
            if (
                self._field_dump_enabled
                and self._field_dump_npz_path is not None
                and step is not None
                and stage in self._field_dump_stages
                and (int(step) % self._field_dump_every_n_steps == 0)
                and key.endswith("_field")
            ):
                arr = self._extract_field_array(value)
                if arr is not None:
                    ds = self._field_dump_downsample
                    arr_ds = arr[::ds, ::ds].astype(np.float32, copy=False)
                    field_key = f"{stage}__{key}"
                    self._field_records[field_key].append(arr_ds)
                    self._field_steps[field_key].append(int(step))
        self.records.append(record)
        self.stage_counts[stage] += 1
        if self._jsonl_handle is not None:
            self._jsonl_handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.flush()
            self._jsonl_handle.close()
            self._jsonl_handle = None
        if self._field_dump_enabled and self._field_dump_npz_path is not None:
            packed: Dict[str, np.ndarray] = {}
            for key, vals in self._field_records.items():
                if vals:
                    packed[key] = np.stack(vals, axis=0)
                    packed[f"{key}__steps"] = np.asarray(self._field_steps[key], dtype=np.int64)
            if packed:
                np.savez_compressed(self._field_dump_npz_path, **packed)
            self._field_records.clear()
            self._field_steps.clear()
            self._field_dump_enabled = False

    @staticmethod
    def _extract_field_array(value: Any) -> Optional[np.ndarray]:
        """Convert tensor/array payload into a 2D numpy array for downsampled dumps."""
        arr: Optional[np.ndarray] = None
        if hasattr(value, "detach"):
            try:
                arr = value.detach().float().cpu().numpy()
            except Exception:
                return None
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            return None

        if arr.ndim != 2:
            return None
        return arr

    def missing_for_step(self, step: int, *, include_closure: bool) -> List[str]:
        required = set(CANONICAL_HOOK_POINTS)
        if not include_closure:
            required.discard("post_closure")
        seen = self.stages_by_step.get(int(step), set())
        return sorted(required - seen)

    def summary(self) -> Dict[str, Any]:
        return {
            "records": len(self.records),
            "stages": dict(self.stage_counts),
            "steps": len(self.stages_by_step),
        }


def default_telemetry_jsonl_path(
    *,
    sim_tag: str,
    seed: Optional[int],
    base_dir: str = "outputs/readback-telemetry",
) -> str:
    """Build a default JSONL path for hook telemetry."""
    seed_label = "none" if seed is None else str(seed)
    ts = int(time.time())
    pid = os.getpid()
    filename = f"{sim_tag}-seed-{seed_label}-{ts}-{pid}.jsonl"
    return os.path.join(base_dir, filename)

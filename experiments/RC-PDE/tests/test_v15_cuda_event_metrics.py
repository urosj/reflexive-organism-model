from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "simulations" / "active" / "simulation-v15-cuda.py"


def _compact_source() -> str:
    text = SRC_PATH.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text)


def test_v15_cuda_has_intrinsic_l1_metric_functions() -> None:
    src = _compact_source()
    assert "def compute_intrinsic_spark_fields(C, g_xx, g_xy, g_yy):" in src
    assert "def compute_intrinsic_collapse_scores(I_tensor, n_active, sqrt_g):" in src
    assert "\"spark_score_mean_t\": torch.mean(spark_soft)" in src
    assert "collapse_score_mean_t, collapse_score_max_t = compute_intrinsic_collapse_scores" in src


def test_v15_cuda_persists_l1_metrics_to_snapshots_and_meta() -> None:
    src = _compact_source()
    assert "'l1_metrics': [ 'spark_score_mean', 'spark_score_max', 'collapse_score_mean', 'collapse_score_max'" in src
    assert "spark_score_mean_val=event_state[\"spark_score_mean_t\"].item()" in src
    assert "spark_score_max_val=event_state[\"spark_score_max_t\"].item()" in src
    assert "collapse_score_mean_val=collapse_score_mean_t.item()" in src
    assert "collapse_score_max_val=collapse_score_max_t.item()" in src
    assert "diagnostics.setdefault('spark_score_mean', []).append" in src
    assert "diagnostics.setdefault('collapse_score_mean', []).append" in src

"""Event detection for collapse and spark phenomena in RC."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import maximum_filter

from rc.field import CoherenceField


def apply_spark(field: CoherenceField, epsilon: float = 1e-4, bump_amplitude: float = 0.1) -> bool:
    """
    Seed a new attractor where Hessian determinant is near-singular.
    """
    hess = field.hessian()
    hxx, hxy, hyx, hyy = hess[0, 0], hess[0, 1], hess[1, 0], hess[1, 1]
    det = hxx * hyy - hxy * hyx
    spark_mask = det < epsilon

    # approx algo from here 
    # but it loves one fat basin
    #C = field.C
    #Cx, Cy = np.gradient(C)
    #grad_norm = np.sqrt(Cx**2 + Cy**2)
    #
    #g_max = float(np.max(grad_norm))
    #g_med = float(np.median(grad_norm))
    #
    #grad_eps = 0.05 * g_max        # “top 5% are considered ‘large’ gradients”
    ## grad_eps = 0.1 * g_med         # “below 0.1×median ≈ flat”
    ## candidate almost-critical points
    #critical_mask = grad_norm < grad_eps
    #if not np.any(critical_mask):
    #    return False  # no candidates this step
    #
    ## Hessian
    #Cxx, _ = np.gradient(Cx)
    #_, Cyy = np.gradient(Cy)
    #Cxy, _ = np.gradient(Cy)  # or a proper mixed derivative
    #det = Cxx * Cyy - Cxy**2
    #
    #abs_det_crit = np.abs(det[critical_mask])
    #
    #if abs_det_crit.size == 0:
    #    return False
    #
    #scale = float(np.median(abs_det_crit))  # typical curvature “strength”
    #
    #det_eps = 0.05 * scale   # “near-degenerate = bottom 5% in |detH|”
    #
    #spark_mask = critical_mask & (np.abs(det) < det_eps)
    # to here 

    if not np.any(spark_mask):
        return False

    y_idx, x_idx = np.where(spark_mask)
    # Choose the location with minimum determinant
    min_idx = np.argmin(det[spark_mask])
    sy, sx = y_idx[min_idx], x_idx[min_idx]

    ny, nx = field.grid_shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    sigma = max(nx, ny) * 0.05
    peak = float(np.max(field.C))
    bump = bump_amplitude * peak * np.exp(-(((X - sx) ** 2 + (Y - sy) ** 2) / (2 * sigma**2)))
    field.C += bump
    field.clip_nonnegative()
    return True


def apply_topology_change(
    field: CoherenceField,
    *,
    spark_epsilon: float = 1e-4,
    bump_amplitude: float = 0.1,
    spark_enabled: bool = True,
    return_details: bool = False,
) -> bool:
    """
    Apply spark events; return True if field changed or a detail dict.
    """
    total_before = field.total_coherence()
    changed_spark = (
        apply_spark(field, epsilon=spark_epsilon, bump_amplitude=bump_amplitude)
        if spark_enabled
        else False
    )
    if changed_spark:
        total_after = field.total_coherence()
        if total_after > 0:
            field.C *= total_before / total_after
        field.clip_nonnegative()

        # enforce minimum C after events too
        _C_min = 1e-6
        field.C = np.maximum(field.C, _C_min)

    if return_details:
        return {"spark": changed_spark}
    return changed_spark


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    try:
        from scipy.ndimage import label
        labeled, num = label(mask)
        return labeled, int(num)
    except Exception:
        # Simple 4-connected labeling fallback
        labeled = np.zeros_like(mask, dtype=int)
        current_label = 0
        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                if mask[y, x] and labeled[y, x] == 0:
                    current_label += 1
                    stack = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        if labeled[cy, cx] != 0:
                            continue
                        labeled[cy, cx] = current_label
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labeled[ny, nx] == 0:
                                stack.append((ny, nx))
        return labeled, current_label


def find_basins(
    C: np.ndarray,
    sigma_list: Tuple[float, ...] = (1.0, 2.0, 4.0),
    min_frac_of_max: float = 0.25,
    min_size: int = 40,
    curvature_threshold: float = 1e-4,
    merge_radius: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-resolution, curvature-aware basin detector.

    Returns (y_indices, x_indices) of basin centers.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:  # pragma: no cover - fallback
        gaussian_filter = None

    if gaussian_filter is None:  # pragma: no cover - basic separable fallback
        def gaussian_filter(arr, sigma):
            from math import exp, ceil
            radius = max(1, int(ceil(3 * sigma)))
            x = np.arange(-radius, radius + 1)
            kernel = np.exp(-(x**2) / (2 * sigma**2))
            kernel /= kernel.sum()
            temp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, arr)
            return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, temp)

    centers = []
    for sigma in sigma_list:
        C_sigma = gaussian_filter(C, sigma=sigma)
        if C_sigma.max() <= 0:
            continue
        thr = min_frac_of_max * float(C_sigma.max())
        # Peak finding on smoothed field
        peaks_mask = find_local_maxima(C_sigma)
        ys, xs = np.where(peaks_mask)
        if ys.size == 0:
            continue
        # Guard against any out-of-range indices from filtering artifacts
        valid = (ys >= 0) & (ys < C_sigma.shape[0]) & (xs >= 0) & (xs < C_sigma.shape[1])
        ys, xs = ys[valid], xs[valid]
        if ys.size == 0:
            continue
        vals = C_sigma[ys, xs]
        keep = vals > thr
        ys, xs, vals = ys[keep], xs[keep], vals[keep]
        if ys.size == 0:
            continue
        for y, x in zip(ys, xs):
            centers.append((int(y), int(x)))

    # Curvature proxy from Hessian determinant of C (not smoothed)
    dCy, dCx = np.gradient(C)
    dCyy, dCyx = np.gradient(dCy)
    dCxy, dCxx = np.gradient(dCx)
    detH = dCxx * dCyy - dCxy * dCyx
    curv_strength = np.abs(detH)

    # Filter peaks by curvature strength
    filtered_centers = []
    for cy, cx in centers:
        curv_val = curv_strength[cy, cx]
        if curv_val < curvature_threshold:
            continue
        filtered_centers.append((cy, cx))

    # Merge centers across scales
    final_centers: List[Tuple[int, int]] = []
    for cy, cx in filtered_centers:
        keep = True
        for fy, fx in final_centers:
            if (cy - fy) ** 2 + (cx - fx) ** 2 <= merge_radius**2:
                keep = False
                break
        if keep:
            final_centers.append((cy, cx))

    if final_centers:
        ys = np.array([c[0] for c in final_centers], dtype=int)
        xs = np.array([c[1] for c in final_centers], dtype=int)
    else:
        ys = np.array([], dtype=int)
        xs = np.array([], dtype=int)
    return ys, xs


def find_local_maxima(C: np.ndarray, footprint_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Return coordinates of local maxima using a maximum filter (legacy)."""
    footprint = np.ones((footprint_size, footprint_size))
    filtered = maximum_filter(C, footprint=footprint, mode="nearest")
    maxima_mask = C == filtered
    return np.where(maxima_mask)


def find_basins_simple(
    C: np.ndarray,
    sigma: float = 1.0,
    min_frac_of_max: float = 0.3,
    min_distance: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple peak-based basin detector.

    Returns (y_indices, x_indices) of basin centers.
    """
    try:
        from scipy.ndimage import gaussian_filter, maximum_filter
    except Exception:  # pragma: no cover - fallback
        gaussian_filter = None
        maximum_filter = None

    if gaussian_filter is not None:
        C_smooth = gaussian_filter(C, sigma=sigma)
    else:  # pragma: no cover - simple separable fallback
        from math import exp, ceil

        radius = max(1, int(ceil(3 * sigma)))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        temp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, C)
        C_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, temp)

    max_val = float(C_smooth.max())
    if max_val <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    thr = min_frac_of_max * max_val
    mask = C_smooth >= thr

    if maximum_filter is not None:
        size = 2 * min_distance + 1
        local_max = maximum_filter(C_smooth, size=size, mode="nearest")
        peaks_mask = (C_smooth == local_max) & mask
    else:  # pragma: no cover - reuse legacy maxima
        peaks_mask = find_local_maxima(C_smooth)
        peaks_mask = peaks_mask & mask

    ys, xs = np.where(peaks_mask)
    return ys, xs

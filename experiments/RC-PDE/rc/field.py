"""Field utilities for Reflexive Coherence simulations."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np


MetricInput = Union[np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]


class CoherenceField:
    """Represents scalar coherence density C(x, t) on a regular 2-D grid."""

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        dx: float,
        init_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        *,
        init_mode: str = "gaussian",
        noise_amplitude: float = 0.05,
        base_level: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Parameters
        ----------
        grid_shape:
            Tuple of (ny, nx) grid points.
        dx:
            Grid spacing (assumed uniform in x and y).
        init_func:
            Callable that returns C values when given meshgrid arrays (X, Y).
            If omitted, `init_mode` selects a built-in initializer.
        init_mode:
            Either "gaussian" (default) or "flat_noise" to use built-in initializers.
        noise_amplitude:
            Magnitude of the random component for the flat-plus-noise initializer.
        base_level:
            Baseline value for the flat-plus-noise initializer.
        rng:
            Optional random generator or seed for reproducible noise fields.
        """
        self.grid_shape = grid_shape
        self.dx = float(dx)
        self._x_coords, self._y_coords = self._build_coords()

        if init_func is None:
            init_func = self._select_default_init(
                init_mode, noise_amplitude, base_level, rng
            )

        X, Y = np.meshgrid(self._x_coords, self._y_coords, indexing="xy")
        self.C = np.asarray(init_func(X, Y), dtype=np.float64)
        if self.C.shape != grid_shape:
            raise ValueError(
                f"Initializer produced shape {self.C.shape}, expected {grid_shape}"
            )

    def _build_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        ny, nx = self.grid_shape
        x = np.arange(nx, dtype=np.float64) * self.dx
        y = np.arange(ny, dtype=np.float64) * self.dx
        return x, y

    def _select_default_init(
        self,
        mode: str,
        noise_amplitude: float,
        base_level: float,
        rng: Optional[np.random.Generator],
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if mode == "gaussian":
            return self._gaussian_bump()
        if mode == "flat_noise":
            return self._flat_plus_noise(noise_amplitude, base_level, rng)
        raise ValueError(f"Unknown init_mode '{mode}'")

    def _gaussian_bump(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        ny, nx = self.grid_shape
        x0 = (nx - 1) * self.dx * 0.5
        y0 = (ny - 1) * self.dx * 0.5
        sigma = min(nx, ny) * self.dx * 0.15

        def init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)))

        return init

    def _flat_plus_noise(
        self,
        amplitude: float,
        base_level: float,
        rng: Optional[np.random.Generator],
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        generator = (
            rng
            if isinstance(rng, np.random.Generator)
            else np.random.default_rng(rng)
            if rng is not None
            else np.random.default_rng()
        )

        def init(x: np.ndarray, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
            noise = generator.standard_normal(self.grid_shape)
            return base_level + amplitude * noise

        return init

    def grad(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return gradient components (Cx, Cy) using central differences."""
        cy, cx = np.gradient(self.C, self.dx, edge_order=2)
        return cx, cy

    def laplacian(self, metric: Optional[MetricInput] = None) -> np.ndarray:
        """
        Compute Laplacian or Laplace–Beltrami if a metric is supplied.

        Parameters
        ----------
        metric:
            Either a callable metric(x, y) -> 2x2 array evaluated element-wise,
            or an array with shape (ny, nx, 2, 2).
        """
        if metric is None:
            return self._laplacian_euclidean()
        return self._laplace_beltrami(metric)

    def _laplacian_euclidean(self) -> np.ndarray:
        c = self.C
        dx2 = self.dx**2
        lap = (
            np.roll(c, 1, axis=0)
            + np.roll(c, -1, axis=0)
            + np.roll(c, 1, axis=1)
            + np.roll(c, -1, axis=1)
            - 4 * c
        ) / dx2
        return lap

    def _laplace_beltrami(self, metric: MetricInput) -> np.ndarray:
        g, g_inv, sqrt_det = self._metric_arrays(metric)
        cx, cy = self.grad()

        flux_x = sqrt_det * (g_inv[..., 0, 0] * cx + g_inv[..., 0, 1] * cy)
        flux_y = sqrt_det * (g_inv[..., 1, 0] * cx + g_inv[..., 1, 1] * cy)

        _, d_flux_x = np.gradient(flux_x, self.dx, edge_order=2)
        dy_flux_y, _ = np.gradient(flux_y, self.dx, edge_order=2)

        # Divergence: d/dx(flux_x) + d/dy(flux_y)
        divergence = d_flux_x + dy_flux_y
        return divergence / sqrt_det

    def _metric_arrays(
        self, metric: MetricInput
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ny, nx = self.grid_shape
        if callable(metric):
            X, Y = np.meshgrid(self._x_coords, self._y_coords, indexing="xy")
            g = np.empty((ny, nx, 2, 2), dtype=np.float64)
            g[:, :, :, :] = metric(X, Y)
        else:
            g = np.asarray(metric, dtype=np.float64)
            if g.shape != (ny, nx, 2, 2):
                raise ValueError(f"Metric array has shape {g.shape}, expected {(ny, nx, 2, 2)}")

        det = g[..., 0, 0] * g[..., 1, 1] - g[..., 0, 1] * g[..., 1, 0]
        sqrt_det = np.sqrt(det)

        g_inv = np.empty_like(g)
        g_inv[..., 0, 0] = g[..., 1, 1] / det
        g_inv[..., 1, 1] = g[..., 0, 0] / det
        g_inv[..., 0, 1] = -g[..., 0, 1] / det
        g_inv[..., 1, 0] = -g[..., 1, 0] / det

        return g, g_inv, sqrt_det

    def hessian(self) -> np.ndarray:
        """Return Hessian matrix components as array with shape (2, 2, ny, nx)."""
        cx, cy = self.grad()
        cxx = np.gradient(cx, self.dx, axis=1, edge_order=2)
        cxy = np.gradient(cx, self.dx, axis=0, edge_order=2)
        cyx = np.gradient(cy, self.dx, axis=1, edge_order=2)
        cyy = np.gradient(cy, self.dx, axis=0, edge_order=2)
        return np.array([[cxx, cxy], [cyx, cyy]])

    def total_coherence(self, metric: Optional[MetricInput] = None) -> float:
        """Compute integral of C over the domain with optional metric volume."""
        if metric is None:
            weight = 1.0
        else:
            _, _, sqrt_det = self._metric_arrays(metric)
            weight = sqrt_det
        return float(np.sum(self.C * weight) * (self.dx**2))

    def clip_nonnegative(self) -> None:
        """Clip C in-place to enforce non-negativity."""
        np.maximum(self.C, 0.0, out=self.C)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"CoherenceField(shape={self.grid_shape}, dx={self.dx})"

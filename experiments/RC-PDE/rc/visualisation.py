"""Visualization helpers for RC simulations."""

from __future__ import annotations

import pathlib
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# Default to non-interactive backend for CLI use.
matplotlib.use("Agg")

from rc.events import find_basins_simple


class RCAnimator:
    """Animator for coherence field with optional flux and curvature overlay."""

    def __init__(
        self,
        field_shape: Tuple[int, int],
        cmap: str = "viridis",
        vector_scale: float = 5.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        sample_quiver: int = 8,
        dt: float = 1.0,
        use_curvature_alpha: bool = True,
        show_overlay: bool = True,
        truth_mode: bool = False,
        use_global_clim: bool = True,
        max_clamp: Optional[float] = None,
    ) -> None:
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Coherence C")
        self.vmin = vmin
        self.vmax = vmax
        self.im_heat = self.ax.imshow(
            np.zeros(field_shape),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
            origin="lower",
            alpha=1.0,
        )
        self.im = self.ax.imshow(
            np.zeros(field_shape),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
            origin="lower",
        )
        self.cbar = self.fig.colorbar(self.im_heat, ax=self.ax, fraction=0.046, pad=0.04)
        self.cbar.set_label("Coherence C")
        self.quiver = None
        self.sample_quiver = sample_quiver
        self.vector_scale = vector_scale
        self.ax.set_axis_off()
        self.dt = dt
        self.truth_mode = truth_mode
        self.use_curvature_alpha = use_curvature_alpha and not truth_mode
        self.show_overlay = show_overlay and not truth_mode
        self.use_global_clim = use_global_clim
        self.text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            color="white",
            fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.4, pad=3),
        )
        self._frames: List[
            Tuple[int, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
        ] = []
        self._C_min: float = float("inf")
        self._C_max: float = float("-inf")
        self.max_clamp = max_clamp

    def update(
        self,
        frame: int,
        C: np.ndarray,
        Jx: Optional[np.ndarray] = None,
        Jy: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
    ) -> None:
        """Store a frame for later animation."""
        C_vis = np.clip(C, None, self.max_clamp) if self.max_clamp is not None else C
        self._C_min = min(self._C_min, float(np.min(C_vis)))
        self._C_max = max(self._C_max, float(np.max(C_vis)))
        self._frames.append(
            (
                int(frame),
                C.copy(),
                None if Jx is None else Jx.copy(),
                None if Jy is None else Jy.copy(),
                None if K is None else K.copy(),
            )
        )

    def _curvature_alpha(self, K: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if K is None or not self.use_curvature_alpha:
            return None
        detK = K[..., 0, 0] * K[..., 1, 1] - K[..., 0, 1] * K[..., 1, 0]
        strength = np.sqrt(np.abs(detK))
        norm = strength / (np.max(strength) + 1e-12)
        return np.clip(norm, 0.1, 1.0)

    def _build_animation(self) -> FuncAnimation:
        if not self._frames:
            raise RuntimeError("No frames recorded; call update() during simulation first.")

        skip = self.sample_quiver
        _, C0, Jx0, Jy0, _ = self._frames[0]
        X, Y = np.meshgrid(
            np.arange(C0.shape[1])[::skip],
            np.arange(C0.shape[0])[::skip],
        )
        # Initialize quiver with zero vectors at the sampled grid
        if not self.truth_mode:
            self.quiver = self.ax.quiver(
                X,
                Y,
                np.zeros_like(X, dtype=float),
                np.zeros_like(Y, dtype=float),
                color="white",
                pivot="mid",
                scale=self.vector_scale,
                alpha=0.4,
                linewidth=0.5,
            )
        else:
            self.quiver = None

        def _update(i):
            step, C, Jx, Jy, K = self._frames[i]
            # Compute clim per frame or globally
            vmin = self.vmin
            vmax = self.vmax
            if self.use_global_clim:
                if vmin is None:
                    vmin = self._C_min
                if vmax is None:
                    vmax = self._C_max
            else:
                if vmin is None:
                    vmin = float(np.min(C))
                if vmax is None:
                    vmax = float(np.max(C))
            if vmin == vmax:
                vmin -= 1e-6
                vmax += 1e-6
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            self.im_heat.set_norm(norm)
            self.im.set_norm(norm)
            if hasattr(self, "cbar"):
                self.cbar.update_normal(self.im_heat)

            # Base heatmap always visible
            self.im_heat.set_data(C)
            self.im_heat.set_alpha(1.0)

            artists = [self.im_heat]

            if self.show_overlay and K is not None:
                self.im.set_data(C)
                alpha = self._curvature_alpha(K, C.shape)
                if alpha is not None:
                    self.im.set_alpha(alpha)
                else:
                    self.im.set_alpha(1.0)
                artists.append(self.im)
            else:
                self.im.set_alpha(0.0)

            if self.quiver is not None and Jx is not None and Jy is not None:
                self.quiver.set_UVC(Jx[::skip, ::skip], Jy[::skip, ::skip])
                artists.append(self.quiver)

            # Diagnostics
            y_idx, x_idx = find_basins_simple(C)
            basins = len(y_idx) if C.size else 0
            self.text.set_text(f"step={step}, ⟨C⟩={C.mean():.3g}, basins={basins}")
            artists.append(self.text)
            return artists

        return FuncAnimation(self.fig, _update, frames=len(self._frames), blit=True, interval=50)

    def animate(self) -> FuncAnimation:
        """Create the FuncAnimation from collected frames."""
        self.anim = self._build_animation()
        return self.anim

    def save(self, path: str, fps: int = 5) -> pathlib.Path:
        """Save the collected frames to an animation file."""
        anim = getattr(self, "anim", None) or self._build_animation()
        path_obj = pathlib.Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        ext = path_obj.suffix.lower()
        if ext == ".gif":
            writer = PillowWriter(fps=fps)
            anim.save(path_obj, writer=writer)
        else:
            # Default to PillowWriter for robustness (MP4 needs ffmpeg)
            writer = PillowWriter(fps=fps)
            anim.save(path_obj.with_suffix(".gif"), writer=writer)
            path_obj = path_obj.with_suffix(".gif")
        return path_obj

import numpy as np
from vispy import scene


class RCVispyViewer:
    """VisPy-based viewer for RC simulation."""

    def __init__(self, nx, ny, c_min, c_max, title="RC-PDE v12"):
        self.nx = nx
        self.ny = ny
        self.c_min = c_min
        self.c_max = c_max

        self.canvas = scene.SceneCanvas(keys='interactive', size=(1200, 700), show=True, title=title)
        self.grid = self.canvas.central_widget.add_grid(margin=10, spacing=6)

        # Top-row labels
        lbl_c = scene.Label("Coherence C", color="white")
        lbl_g = scene.Label("Geometry sqrt(|g|)", color="white")
        lbl_s = scene.Label("Sparks", color="white")
        self.grid.add_widget(lbl_c, row=0, col=0)
        self.grid.add_widget(lbl_g, row=0, col=1)
        self.grid.add_widget(lbl_s, row=0, col=2)

        # Image views
        self.view_c = self.grid.add_view(row=1, col=0, camera='panzoom')
        self.view_g = self.grid.add_view(row=1, col=1, camera='panzoom')
        self.view_s = self.grid.add_view(row=1, col=2, camera='panzoom')

        self.img_c = scene.visuals.Image(np.zeros((nx, ny), dtype=np.float32), cmap='viridis', parent=self.view_c.scene)
        self.img_g = scene.visuals.Image(np.ones((nx, ny), dtype=np.float32), cmap='magma', parent=self.view_g.scene)
        self.img_s = scene.visuals.Image(np.zeros((nx, ny), dtype=np.float32), cmap='inferno', parent=self.view_s.scene)

        # Axes for image plots
        self.axis_c_x = self._make_axis(self.view_c, (0.0, 0.0), (float(nx), 0.0), (0.0, float(nx)), "x")
        self.axis_c_y = self._make_axis(self.view_c, (0.0, 0.0), (0.0, float(ny)), (0.0, float(ny)), "y")
        self.axis_g_x = self._make_axis(self.view_g, (0.0, 0.0), (float(nx), 0.0), (0.0, float(nx)), "x")
        self.axis_g_y = self._make_axis(self.view_g, (0.0, 0.0), (0.0, float(ny)), (0.0, float(ny)), "y")
        self.axis_s_x = self._make_axis(self.view_s, (0.0, 0.0), (float(nx), 0.0), (0.0, float(nx)), "x")
        self.axis_s_y = self._make_axis(self.view_s, (0.0, 0.0), (0.0, float(ny)), (0.0, float(ny)), "y")

        self.view_c.camera.set_range()
        self.view_g.camera.set_range()
        self.view_s.camera.set_range()

        # Bottom-row labels
        lbl_dt = scene.Label("dt over iterations", color="white")
        lbl_mass = scene.Label("mass C (cyan) / I_tot (orange)", color="white")
        lbl_ids = scene.Label("active identities", color="white")
        self.grid.add_widget(lbl_dt, row=2, col=0)
        self.grid.add_widget(lbl_mass, row=2, col=1)
        self.grid.add_widget(lbl_ids, row=2, col=2)

        # Diagnostic plots
        self.view_dt = self.grid.add_view(row=3, col=0, camera='panzoom')
        self.view_mass = self.grid.add_view(row=3, col=1, camera='panzoom')
        self.view_ids = self.grid.add_view(row=3, col=2, camera='panzoom')

        self.line_dt = scene.visuals.Line(color='white', parent=self.view_dt.scene)
        self.line_mass = scene.visuals.Line(color='cyan', parent=self.view_mass.scene)
        self.line_imass = scene.visuals.Line(color='orange', parent=self.view_mass.scene)
        self.line_ids = scene.visuals.Line(color='lime', parent=self.view_ids.scene)

        # Axes for diagnostic plots (gnuplot-like readability)
        self.axis_dt_x = self._make_axis(self.view_dt, (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), "iteration")
        self.axis_dt_y = self._make_axis(self.view_dt, (0.0, 0.0), (0.0, 1.0), (0.0, 1.0), "dt")
        self.axis_mass_x = self._make_axis(self.view_mass, (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), "iteration")
        self.axis_mass_y = self._make_axis(self.view_mass, (0.0, 0.0), (0.0, 1.0), (0.0, 1.0), "mass")
        self.axis_ids_x = self._make_axis(self.view_ids, (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), "iteration")
        self.axis_ids_y = self._make_axis(self.view_ids, (0.0, 0.0), (0.0, 1.0), (0.0, 1.0), "# ids")

        self.dt_vals = []
        self.mass_vals = []
        self.imass_vals = []
        self.ids_vals = []

        # Status line (mimics matplotlib suptitle diagnostics text)
        self.status_label = scene.Label("", color="lightgray")
        self.grid.add_widget(self.status_label, row=4, col=0, col_span=3)

    @staticmethod
    def _make_axis(view, p0, p1, domain, axis_label):
        return scene.Axis(
            pos=np.array([p0, p1], dtype=np.float32),
            domain=domain,
            tick_direction=(-1, 0) if p0[0] == p1[0] else (0, -1),
            axis_color="white",
            tick_color="white",
            text_color="white",
            axis_label=axis_label,
            parent=view.scene,
        )

    @staticmethod
    def _sanitize_axis_domain(v0, v1, fallback=(0.0, 1.0)):
        a = float(v0)
        b = float(v1)

        if not np.isfinite(a) or not np.isfinite(b):
            return fallback
        if a == b:
            delta = max(1e-6, abs(a) * 1e-3 + 1e-6)
            a, b = a - delta, b + delta
        if a > b:
            a, b = b, a

        # Keep domains in a numerically safe range for VisPy tick generation.
        max_abs = 1e12
        a = np.clip(a, -max_abs, max_abs)
        b = np.clip(b, -max_abs, max_abs)
        if a == b:
            return fallback
        return float(a), float(b)

    @classmethod
    def _set_line_view_range(cls, view, x_last, y_min, y_max, axis_x=None, axis_y=None):
        if x_last <= 0:
            x0, x1 = 0.0, 1.0
        else:
            x0, x1 = 0.0, float(x_last)

        y_min = float(y_min)
        y_max = float(y_max)
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            y0, y1 = 0.0, 1.0
        elif y_min == y_max:
            pad = max(1e-6, abs(y_min) * 0.05 + 1e-6)
            y0, y1 = y_min - pad, y_max + pad
        else:
            pad = 0.05 * (y_max - y_min)
            y0, y1 = y_min - pad, y_max + pad

        x0, x1 = cls._sanitize_axis_domain(x0, x1)
        y0, y1 = cls._sanitize_axis_domain(y0, y1)
        view.camera.set_range(x=(x0, x1), y=(y0, y1), margin=0.0)

        if axis_x is not None:
            try:
                axis_x.pos = np.array([[x0, y0], [x1, y0]], dtype=np.float32)
                axis_x.domain = (x0, x1)
            except Exception:
                pass
        if axis_y is not None:
            try:
                axis_y.pos = np.array([[x0, y0], [x0, y1]], dtype=np.float32)
                axis_y.domain = (y0, y1)
            except Exception:
                pass

    def update_images(self, c, sqrt_g, spark):
        self.img_c.set_data(c)
        self.img_g.set_data(sqrt_g)
        self.img_s.set_data(spark)

    def update_diagnostics(self, dt, mass, imass, ids):
        self.dt_vals.append(dt)
        self.mass_vals.append(mass)
        self.imass_vals.append(imass)
        self.ids_vals.append(ids)

        x = np.arange(len(self.dt_vals), dtype=np.float32)
        self.line_dt.set_data(np.c_[x, self.dt_vals])
        self.line_mass.set_data(np.c_[x, self.mass_vals])
        self.line_imass.set_data(np.c_[x, self.imass_vals])
        self.line_ids.set_data(np.c_[x, self.ids_vals])

        x_last = x[-1] if x.size else 0.0
        self._set_line_view_range(
            self.view_dt, x_last, np.min(self.dt_vals), np.max(self.dt_vals),
            axis_x=self.axis_dt_x, axis_y=self.axis_dt_y
        )
        mass_min = min(np.min(self.mass_vals), np.min(self.imass_vals))
        mass_max = max(np.max(self.mass_vals), np.max(self.imass_vals))
        self._set_line_view_range(
            self.view_mass, x_last, mass_min, mass_max,
            axis_x=self.axis_mass_x, axis_y=self.axis_mass_y
        )
        self._set_line_view_range(
            self.view_ids, x_last, np.min(self.ids_vals), np.max(self.ids_vals),
            axis_x=self.axis_ids_x, axis_y=self.axis_ids_y
        )

        step = len(self.dt_vals)
        self.status_label.text = (
            f"step={step}  dt={float(dt):.3e}  mass={float(mass):.4f}  "
            f"I_mass={float(imass):.4f}  ids={int(ids)}"
        )

    def process_events(self):
        # VisPy app.run() drives the event loop; avoid re-entrant process_events().
        return

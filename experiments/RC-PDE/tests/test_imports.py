"""Placeholder tests to ensure modules import."""

def test_can_import_package():
    import rc  # noqa: F401


def test_can_import_modules():
    for mod in ("field", "geometry", "flux", "pde", "events", "visualisation"):
        __import__(f"rc.{mod}")

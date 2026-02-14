def map_activity(activity: float) -> dict[str, float]:
    """
    Map a single [0, 1] activity slider into existing RC knobs.

    Design goals:
    - activity=0: quieter (lower identity budget, stronger collapse damping)
    - activity=1: hotter (higher identity budget, weaker collapse damping)
    - Does not change default behavior unless the caller opts in.
    """
    if not (0.0 <= activity <= 1.0):
        raise ValueError("activity must be in [0, 1]")

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    # Keep these conservative; the point is to provide a searchable 1D slice.
    identity_cap_fraction = lerp(0.06, 0.35, activity)
    identity_birth_gate_fraction = lerp(0.65, 0.95, activity)
    closure_softness = lerp(0.25, 0.90, activity)
    spark_softness = lerp(0.04, 0.12, activity)
    collapse_softness = lerp(0.85, 0.20, activity)

    return {
        "identity_cap_fraction": float(identity_cap_fraction),
        "identity_birth_gate_fraction": float(identity_birth_gate_fraction),
        "closure_softness": float(closure_softness),
        "spark_softness": float(spark_softness),
        "collapse_softness": float(collapse_softness),
    }

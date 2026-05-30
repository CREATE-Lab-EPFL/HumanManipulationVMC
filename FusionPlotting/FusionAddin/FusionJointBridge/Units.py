"""
Unit conversions between the bridge file's declared units and the
Fusion 360 API internal units (radians for angles, cm for lengths).

This is the ONLY place conversions are done — change units here and
nowhere else.
"""

import math


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

def angle_to_fusion(value: float, unit: str) -> float:
    """Client unit -> Fusion internal (radians)."""
    if unit == "degrees":
        return math.radians(value)
    if unit == "radians":
        return float(value)
    raise ValueError(f"Unknown angle unit: {unit!r}. Use 'degrees' or 'radians'.")


def angle_from_fusion(value: float, unit: str) -> float:
    """Fusion internal (radians) -> client unit."""
    if unit == "degrees":
        return math.degrees(value)
    if unit == "radians":
        return float(value)
    raise ValueError(f"Unknown angle unit: {unit!r}. Use 'degrees' or 'radians'.")


# ---------------------------------------------------------------------------
# Lengths
# ---------------------------------------------------------------------------

def length_to_fusion(value: float, unit: str) -> float:
    """Client unit -> Fusion internal (cm)."""
    if unit == "mm":
        return value / 10.0
    if unit == "cm":
        return float(value)
    raise ValueError(f"Unknown length unit: {unit!r}. Use 'mm' or 'cm'.")


def length_from_fusion(value: float, unit: str) -> float:
    """Fusion internal (cm) -> client unit."""
    if unit == "mm":
        return value * 10.0
    if unit == "cm":
        return float(value)
    raise ValueError(f"Unknown length unit: {unit!r}. Use 'mm' or 'cm'.")

"""
Core logic: read bridge file, apply joint targets, write back current values.
Runs on the Fusion main thread (called from the custom-event handler).
"""

import json
import pathlib
import traceback
from datetime import datetime, timezone

import Units

BRIDGE_PATH = pathlib.Path.home() / "FusionBridge" / "commands.json"
LOG_PATH    = pathlib.Path.home() / "FusionBridge" / "addin.log"

_DEFAULT_UNITS = {"angle": "degrees", "length": "mm"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}  {msg}\n")


def _find_joint(design, name: str):
    """Search the entire design tree for a joint matching name."""
    import adsk.fusion

    root = design.rootComponent

    for j in root.joints:
        if j.name == name:
            return j
    for j in root.asBuiltJoints:
        if j.name == name:
            return j
    for occ in root.allOccurrences:
        comp = occ.component
        for j in comp.joints:
            if j.name == name:
                return j
        for j in comp.asBuiltJoints:
            if j.name == name:
                return j

    return None


def _get_joint_value(joint, angle_unit: str, length_unit: str):
    """Read the current motion value in client units. Returns None for unsupported types."""
    import adsk.fusion

    jtype = joint.jointMotion.jointType

    if jtype == adsk.fusion.JointTypes.RevoluteJointType:
        return Units.angle_from_fusion(joint.jointMotion.rotationValue, angle_unit)

    if jtype == adsk.fusion.JointTypes.SliderJointType:
        return Units.length_from_fusion(joint.jointMotion.slideValue, length_unit)

    # Cylindrical has both
    if jtype == adsk.fusion.JointTypes.CylindricalJointType:
        return Units.angle_from_fusion(joint.jointMotion.rotationValue, angle_unit)

    return None


def _set_joint_value(joint, value: float, angle_unit: str, length_unit: str) -> bool:
    """Apply value in client units to joint. Returns True on success."""
    import adsk.fusion

    jtype = joint.jointMotion.jointType

    if jtype == adsk.fusion.JointTypes.RevoluteJointType:
        joint.jointMotion.rotationValue = Units.angle_to_fusion(value, angle_unit)
        return True

    if jtype == adsk.fusion.JointTypes.SliderJointType:
        joint.jointMotion.slideValue = Units.length_to_fusion(value, length_unit)
        return True

    if jtype == adsk.fusion.JointTypes.CylindricalJointType:
        joint.jointMotion.rotationValue = Units.angle_to_fusion(value, angle_unit)
        return True

    _log(f"Unsupported joint type {jtype} — skipping")
    return False


# ---------------------------------------------------------------------------
# Main entry point (called from the Fusion event handler)
# ---------------------------------------------------------------------------

def apply_and_readback(app) -> None:
    import adsk.fusion

    if not BRIDGE_PATH.exists():
        return

    try:
        with open(BRIDGE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    product = app.activeProduct
    if not isinstance(product, adsk.fusion.Design):
        return

    design  = adsk.fusion.Design.cast(product)
    units   = data.get("units", _DEFAULT_UNITS)
    angle_u = units.get("angle",  "degrees")
    length_u = units.get("length", "mm")
    targets = data.get("targets", {})
    current: dict = {}

    for name, value in targets.items():
        joint = _find_joint(design, name)
        if joint is None:
            _log(f"Joint not found: {name!r}")
            continue

        try:
            ok = _set_joint_value(joint, float(value), angle_u, length_u)
            if ok:
                read_back = _get_joint_value(joint, angle_u, length_u)
                if read_back is not None:
                    current[name] = round(read_back, 6)
        except Exception:
            _log(f"Error on joint {name!r}:\n{traceback.format_exc()}")

    data["current"]      = current
    data["last_updated"] = datetime.now(timezone.utc).isoformat()

    tmp = BRIDGE_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(BRIDGE_PATH)
    except OSError as e:
        _log(f"Could not write bridge file: {e}")

    try:
        app.activeViewport.refresh()
    except Exception:
        pass

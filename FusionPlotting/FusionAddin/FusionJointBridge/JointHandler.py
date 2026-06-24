"""
Core logic: read bridge file, apply joint targets, write back current values.
Also writes a discovery file (joints_discovery.json) listing every joint in
the active design — use it to find Fusion joint names and confirm sign conventions.

All public functions run on the Fusion main thread (called from the custom-event handler).
"""

import json
import math
import pathlib
import traceback
from datetime import datetime, timezone

import Units
import FusionConventions

BRIDGE_PATH    = pathlib.Path.home() / "FusionBridge" / "commands.json"
DISCOVERY_PATH = pathlib.Path.home() / "FusionBridge" / "joints_discovery.json"
LOG_PATH       = pathlib.Path.home() / "FusionBridge" / "addin.log"

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
    """Read current motion value in client units. Returns None for unsupported types."""
    import adsk.fusion

    jtype = joint.jointMotion.jointType
    if jtype == adsk.fusion.JointTypes.RevoluteJointType:
        return Units.angle_from_fusion(joint.jointMotion.rotationValue, angle_unit)
    if jtype == adsk.fusion.JointTypes.SliderJointType:
        return Units.length_from_fusion(joint.jointMotion.slideValue, length_unit)
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
    _log(f"Unsupported joint type {joint.jointMotion.jointType} — skipping")
    return False


# ---------------------------------------------------------------------------
# Apply targets and read back current values
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

    design   = adsk.fusion.Design.cast(product)
    units    = data.get("units", _DEFAULT_UNITS)
    angle_u  = units.get("angle",  "degrees")
    length_u = units.get("length", "mm")
    targets  = data.get("targets", {})
    current: dict = {}

    for code_name, code_value in targets.items():
        # Resolve Fusion name and sign from the conventions map
        fusion_name, sign = FusionConventions.resolve(code_name)

        joint = _find_joint(design, fusion_name)
        if joint is None:
            _log(f"Joint not found: {fusion_name!r}  (code name: {code_name!r})")
            continue

        # code convention → Fusion convention
        fusion_side_value = sign * float(code_value)

        try:
            ok = _set_joint_value(joint, fusion_side_value, angle_u, length_u)
            if ok:
                read_back = _get_joint_value(joint, angle_u, length_u)
                if read_back is not None:
                    # Fusion convention → code convention
                    current[code_name] = round(sign * read_back, 6)
        except Exception:
            _log(f"Error on joint {fusion_name!r}:\n{traceback.format_exc()}")

    data["current"]      = current
    data["last_updated"] = datetime.now(timezone.utc).isoformat()

    tmp = BRIDGE_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(BRIDGE_PATH)
    except OSError as e:
        _log(f"Could not write bridge file: {e}")



# ---------------------------------------------------------------------------
# Joint discovery — run once on add-in start
# ---------------------------------------------------------------------------

def discover_all_joints(app) -> None:
    """
    Write every joint in the active design to joints_discovery.json.

    Includes: Fusion name, type, current value, and limits (if enabled).
    Use this output to fill in the fusion_name entries in FusionConventions.py
    and to verify that sign conventions match the code (compare limits).
    """
    import adsk.fusion

    product = app.activeProduct
    if not isinstance(product, adsk.fusion.Design):
        return

    design = adsk.fusion.Design.cast(product)

    _TYPE_NAMES = {
        adsk.fusion.JointTypes.RevoluteJointType:    "revolute",
        adsk.fusion.JointTypes.SliderJointType:      "slider",
        adsk.fusion.JointTypes.CylindricalJointType: "cylindrical",
        adsk.fusion.JointTypes.PinSlotJointType:     "pin_slot",
        adsk.fusion.JointTypes.PlanarJointType:      "planar",
        adsk.fusion.JointTypes.BallJointType:        "ball",
    }

    joints_info: dict = {}

    def _collect(joint_collection, source: str) -> None:
        for j in joint_collection:
            name  = j.name
            jtype = j.jointMotion.jointType
            info  = {
                "type":   _TYPE_NAMES.get(jtype, f"unknown({jtype})"),
                "source": source,
            }

            try:
                if jtype == adsk.fusion.JointTypes.RevoluteJointType:
                    raw = j.jointMotion.rotationValue
                    info["current_deg"] = round(math.degrees(raw), 4)
                    info["current_rad"] = round(raw, 6)
                    lim = j.jointMotion.rotationLimits
                    info["min_deg"] = (
                        round(math.degrees(lim.minimumValue), 4)
                        if lim.isMinimumValueEnabled else None
                    )
                    info["max_deg"] = (
                        round(math.degrees(lim.maximumValue), 4)
                        if lim.isMaximumValueEnabled else None
                    )

                elif jtype == adsk.fusion.JointTypes.SliderJointType:
                    raw = j.jointMotion.slideValue
                    info["current_cm"] = round(raw, 6)
                    info["current_mm"] = round(raw * 10.0, 4)
                    lim = j.jointMotion.slideLimits
                    info["min_mm"] = (
                        round(lim.minimumValue * 10.0, 4)
                        if lim.isMinimumValueEnabled else None
                    )
                    info["max_mm"] = (
                        round(lim.maximumValue * 10.0, 4)
                        if lim.isMaximumValueEnabled else None
                    )

            except Exception as e:
                info["read_error"] = str(e)

            joints_info[name] = info

    _collect(design.rootComponent.joints,        "root")
    _collect(design.rootComponent.asBuiltJoints, "root_as_built")

    for occ in design.rootComponent.allOccurrences:
        comp = occ.component
        _collect(comp.joints,        comp.name)
        _collect(comp.asBuiltJoints, f"{comp.name}_as_built")

    output = {
        "design":      design.rootComponent.name,
        "joint_count": len(joints_info),
        "joints":      joints_info,
    }

    DISCOVERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DISCOVERY_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    _log(f"Discovery: {len(joints_info)} joints written to {DISCOVERY_PATH}")

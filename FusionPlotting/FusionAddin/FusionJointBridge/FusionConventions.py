"""
code_name -> (fusion_name | None, sign)
None for fusion_name uses the code name as the Fusion browser lookup key.
fusion_value = sign * code_value
"""

JOINT_MAP: dict[str, tuple] = {
    "MCP": (None, +1),
    "PIP": (None, +1),
    "DIP": (None, +1),

    "wrist_pitch": (None, -1),
    "wrist_yaw":   (None, +1),

    "thumb_CMC1": (None, -1),
    "thumb_CMC2": (None, -1),
    "thumb_MCP":  (None, -1),
    "thumb_IP":   (None, -1),

    "index_spread":  (None, +1),
    "middle_spread": (None, +1),
    "ring_spread":   (None, +1),
    "pinky_spread":  (None, +1),

    "index_MCP":  (None, -1),
    "index_PIP":  (None, -1),
    "middle_MCP": (None, -1),
    "middle_PIP": (None, -1),
    "ring_MCP":   (None, -1),
    "ring_PIP":   (None, -1),
    "pinky_MCP":  (None, -1),
    "pinky_PIP":  (None, -1),
}


def resolve(code_name: str) -> tuple[str, float]:
    entry = JOINT_MAP.get(code_name)
    if entry is None:
        return code_name, 1.0
    fusion_name, sign = entry
    return (code_name if fusion_name is None else fusion_name), float(sign)

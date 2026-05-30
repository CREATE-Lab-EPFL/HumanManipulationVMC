"""
code_name -> (fusion_name | None, sign)
None for fusion_name uses the code name as the Fusion browser lookup key.
fusion_value = sign * code_value
"""

JOINT_MAP: dict[str, tuple] = {
    "MCP": (None, +1),
    "PIP": (None, +1),
    "DIP": (None, +1),

    "wrist_pitch": ("Wrist_Pitch", -1),
    "wrist_yaw":   ("Wrist_Yaw",   +1),

    "thumb_CMC1": ("Thumb_CMC1", +1),
    "thumb_CMC2": ("Thumb_CMC2", -1),
    "thumb_MCP":  ("Thumb_MCP",  -1),
    "thumb_IP":   ("Thumb_IP",   -1),

    "index_spread":  ("Index_Spread",  +1),
    "middle_spread": ("Middle_Spread", +1),
    "ring_spread":   ("Ring_Spread",   +1),
    "pinky_spread":  ("Pinky_Spread",  +1),

    "index_MCP":  ("Index_MCP",  -1),
    "index_PIP":  ("Index_PIP",  -1),
    "middle_MCP": ("Middle_MCP", -1),
    "middle_PIP": ("Middle_PIP", -1),
    "ring_MCP":   ("Ring_MCP",   -1),
    "ring_PIP":   ("Ring_PIP",   -1),
    "pinky_MCP":  ("Pinky_MCP",  -1),
    "pinky_PIP":  ("Pinky_PIP",  -1),
}


def resolve(code_name: str) -> tuple[str, float]:
    entry = JOINT_MAP.get(code_name)
    if entry is None:
        return code_name, 1.0
    fusion_name, sign = entry
    return (code_name if fusion_name is None else fusion_name), float(sign)

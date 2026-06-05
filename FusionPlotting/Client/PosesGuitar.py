"""
FusionPlotting poses for PassiveCompliance guitar-playing experiment.

Fingers (index/middle/ring/pinky) lightly closed at FINGER_CLOSED_POSE.
Thumb and wrist at zero (wrist is rigid).
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

GUITAR_CLOSED: dict[str, float] = {
    "wrist_pitch":  0.0,
    "wrist_yaw":    0.0,

    "thumb_CMC1":   0.0,
    "thumb_CMC2":   0.0,
    "thumb_MCP":    0.0,
    "thumb_IP":     0.0,

    "index_spread":   0.0,
    "ring_spread":    0.0,
    "pinky_spread":   0.0,

    "index_MCP":  20.0,
    "index_PIP":  20.0,
    "middle_MCP": 20.0,
    "middle_PIP": 20.0,
    "ring_MCP":   20.0,
    "ring_PIP":   20.0,
    "pinky_MCP":  20.0,
    "pinky_PIP":  20.0,
}

client = JointClient()
print("Guitar playing — finger closed pose:")
for name, val in GUITAR_CLOSED.items():
    print(f"  {name}: {val}")
client.write_targets(GUITAR_CLOSED, angle_unit="degrees", length_unit="mm")
time.sleep(0.5)

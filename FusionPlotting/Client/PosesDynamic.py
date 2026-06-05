"""
FusionPlotting poses for TunableCompliance dynamic grasping experiment.

PC1 grasp pose used for dynamic grasping (lighter grasp pre-contact).
Wrist is rigid (position-controlled), always at 0.0.
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

GRASP_PC1: dict[str, float] = {
    "wrist_pitch":  0.0,
    "wrist_yaw":    0.0,

    "thumb_CMC1":  30.0,
    "thumb_CMC2":   0.0,
    "thumb_MCP":   70.0,
    "thumb_IP":    70.0,

    "index_spread":   0.0,
    "ring_spread":    0.0,
    "pinky_spread":   0.0,

    "index_MCP":  55.0,
    "index_PIP":  65.0,
    "middle_MCP": 55.0,
    "middle_PIP": 65.0,
    "ring_MCP":   55.0,
    "ring_PIP":   65.0,
    "pinky_MCP":  55.0,
    "pinky_PIP":  65.0,
}

client = JointClient()
print("TunableCompliance dynamic grasping PC1 pose:")
for name, val in GRASP_PC1.items():
    print(f"  {name}: {val}")
client.write_targets(GRASP_PC1, angle_unit="degrees", length_unit="mm")
time.sleep(0.5)

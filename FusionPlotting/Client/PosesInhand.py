"""
FusionPlotting poses for TunableCompliance in-hand manipulation experiment.

PC1 grasp pose used for in-hand manipulation (tighter grip than standard PC1).
Wrist is rigid (position-controlled), always at 0.0.
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

INHAND_PC1_GRASP: dict[str, float] = {
    "wrist_pitch":  0.0,
    "wrist_yaw":    0.0,

    "thumb_CMC1":  40.0,
    "thumb_CMC2":   0.0,
    "thumb_MCP":   90.0,
    "thumb_IP":    90.0,

    "index_spread":   0.0,
    "ring_spread":    0.0,
    "pinky_spread":   0.0,

    "index_MCP":  65.0,
    "index_PIP":  75.0,
    "middle_MCP": 65.0,
    "middle_PIP": 75.0,
    "ring_MCP":   65.0,
    "ring_PIP":   75.0,
    "pinky_MCP":  65.0,
    "pinky_PIP":  75.0,
}

client = JointClient()
print("TunableCompliance in-hand manipulation PC1 pose:")
for name, val in INHAND_PC1_GRASP.items():
    print(f"  {name}: {val}")
client.write_targets(INHAND_PC1_GRASP, angle_unit="degrees", length_unit="mm")
time.sleep(0.5)

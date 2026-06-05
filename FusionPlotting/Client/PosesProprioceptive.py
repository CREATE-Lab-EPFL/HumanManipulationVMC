"""
FusionPlotting poses for ProprioceptiveSensing/Hand experiment.

PC1 power grasp pose for proprioceptive stiffness estimation.
Wrist is rigid (position-controlled), always at 0.0.
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

PC1_GRASP: dict[str, float] = {
    "wrist_pitch":  0.0,
    "wrist_yaw":    0.0,

    "thumb_CMC1":  70.0,
    "thumb_CMC2":   0.0,
    "thumb_MCP":   90.0,
    "thumb_IP":    90.0,

    "index_spread":   0.0,
    "ring_spread":    0.0,
    "pinky_spread":   0.0,

    "index_MCP":  65.0,
    "index_PIP":  80.0,
    "middle_MCP": 65.0,
    "middle_PIP": 80.0,
    "ring_MCP":   65.0,
    "ring_PIP":   80.0,
    "pinky_MCP":  65.0,
    "pinky_PIP":  80.0,
}

client = JointClient()
print("ProprioceptiveSensing PC1 grasp pose:")
for name, val in PC1_GRASP.items():
    print(f"  {name}: {val}")
client.write_targets(PC1_GRASP, angle_unit="degrees", length_unit="mm")
time.sleep(0.5)

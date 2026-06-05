"""
FusionPlotting poses for PoseControl experiment.

PC1 (power grasp) and PC2 (precision pinch) synergy poses from Santello et al. 1998.
Wrist is rigid (position-controlled), always at 0.0.
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

PC1_POWER_GRASP: dict[str, float] = {
    "wrist_pitch":  0.0,
    "wrist_yaw":    0.0,

    "thumb_CMC1":  65.0,
    "thumb_CMC2":   0.0,
    "thumb_MCP":   50.0,
    "thumb_IP":    40.0,

    "index_spread":  -2.0,
    "ring_spread":    2.0,
    "pinky_spread":   2.0,

    "index_MCP":  55.0,
    "index_PIP":  65.0,
    "middle_MCP": 55.0,
    "middle_PIP": 65.0,
    "ring_MCP":   55.0,
    "ring_PIP":   65.0,
    "pinky_MCP":  55.0,
    "pinky_PIP":  65.0,
}

PC2_PRECISION_PINCH: dict[str, float] = {
    "wrist_pitch":  0.0,
    "wrist_yaw":    0.0,

    "thumb_CMC1":  50.0,
    "thumb_CMC2":  12.0,
    "thumb_MCP":   35.0,
    "thumb_IP":    25.0,

    "index_spread":  -1.0,
    "ring_spread":    1.0,
    "pinky_spread":   1.0,

    "index_MCP":  25.0,
    "index_PIP":  35.0,
    "middle_MCP": 25.0,
    "middle_PIP": 35.0,
    "ring_MCP":   55.0,
    "ring_PIP":   65.0,
    "pinky_MCP":  55.0,
    "pinky_PIP":  65.0,
}

POSES = {"PC1 (power grasp)": PC1_POWER_GRASP, "PC2 (precision pinch)": PC2_PRECISION_PINCH}

client = JointClient()
for label, pose in POSES.items():
    print(f"\n{label}:")
    for name, val in pose.items():
        print(f"  {name}: {val}")
    client.write_targets(pose, angle_unit="degrees", length_unit="mm")
    time.sleep(1.0)

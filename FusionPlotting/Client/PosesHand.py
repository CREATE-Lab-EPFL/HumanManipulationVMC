"""
Hardcoded poses for the full ADAPT Hand Fusion model.

Usage:
    Set POSE to the figure name you want, then run:
        python PosesHand.py

Joint names and sign conventions:
    wrist_pitch, wrist_yaw            — always 0.0 (position-controlled)
    thumb_CMC1, thumb_CMC2, thumb_MCP, thumb_IP
    index_spread                       — negative = adduction toward middle
    ring_spread, pinky_spread          — positive = adduction toward middle
    index_MCP,  index_PIP
    middle_MCP, middle_PIP
    ring_MCP,   ring_PIP
    pinky_MCP,  pinky_PIP
All angles in degrees, fingers positive = flexion.
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

# ---------------------------------------------------------------------------
# Pick which figure to send.
# ---------------------------------------------------------------------------
POSE = "flat"

# ---------------------------------------------------------------------------
# Figures: fill in figure by figure.
# ---------------------------------------------------------------------------
FIGURES: dict[str, dict[str, float]] = {
    "flat": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   0.0,
        "thumb_CMC2":   0.0,
        "thumb_MCP":    0.0,
        "thumb_IP":     0.0,
        "index_spread": 0.0,
        "ring_spread":  0.0,
        "pinky_spread": 0.0,
        "index_MCP":    0.0,
        "index_PIP":    0.0,
        "middle_MCP":   0.0,
        "middle_PIP":   0.0,
        "ring_MCP":     0.0,
        "ring_PIP":     0.0,
        "pinky_MCP":    0.0,
        "pinky_PIP":    0.0,
    },
}

# ---------------------------------------------------------------------------

if POSE not in FIGURES:
    raise ValueError(f"Unknown pose '{POSE}'. Available: {list(FIGURES)}")

joints = FIGURES[POSE]
client = JointClient()

print(f"Sending pose '{POSE}':")
for name, val in joints.items():
    print(f"  {name}: {val} deg")

client.write_targets(joints, angle_unit="degrees", length_unit="mm")
time.sleep(0.5)

current = client.read_current()
if current:
    print("\nFusion reported back:")
    for name, val in current.items():
        print(f"  {name}: {val:.2f} deg")
else:
    print("\nNo readback yet — is the add-in running in Fusion?")

"""
Hardcoded poses for the single-finger Fusion model.

Usage:
    Set POSE to the figure name you want, then run:
        python PosesFinger.py

Joint names (must match the Fusion model):
    MCP   — metacarpophalangeal
    PIP   — proximal interphalangeal
    DIP   — distal interphalangeal
All angles in degrees, positive = flexion.
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
        "MCP": 0.0,
        "PIP": 0.0,
        "DIP": 0.0,
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

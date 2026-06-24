"""
Hardcoded poses for the single-finger Fusion model.

Usage:
    python PosesFinger.py
    → shows a numbered menu, pick a pose by number.

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
# Figures: fill in figure by figure.
# ---------------------------------------------------------------------------
FIGURES: dict[str, dict[str, float]] = {
    "flat": {
        "MCP": 0.0,
        "PIP": 0.0,
        "DIP": 0.0,
    },
    "bent": {
        "MCP": 10.0,
        "PIP": 10.0,
        "DIP": 10.0,
    },
    "force_gradient": {
        "MCP": 5.0,
        "PIP": 10.0,
        "DIP": 10.0,
    },
}

# ---------------------------------------------------------------------------

names = list(FIGURES)
print("Available poses:")
for i, name in enumerate(names):
    print(f"  [{i}] {name}")

while True:
    raw = input("Select pose number: ").strip()
    if raw.isdigit() and int(raw) < len(names):
        POSE = names[int(raw)]
        break
    print(f"  Please enter a number between 0 and {len(names) - 1}.")

joints = FIGURES[POSE]
client = JointClient()

print(f"\nSending pose '{POSE}':")
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

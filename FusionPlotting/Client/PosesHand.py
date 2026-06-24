"""
Hardcoded poses for the full ADAPT Hand Fusion model.

Usage:
    python PosesHand.py
    → shows a numbered menu, pick a pose by number.

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

    # PassiveCompliance/Hand weight_compliance_hand experiment:
    # index is bent at 40° (the loaded finger), others lightly flexed at 10°.
    # Thumb held at neutral 2° (background regulation from the experiment).
    "weight_compliance": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   2.0,
        "thumb_CMC2":   2.0,
        "thumb_MCP":    2.0,
        "thumb_IP":     2.0,
        "index_spread": 0.0,
        "ring_spread":  0.0,
        "pinky_spread": 0.0,
        "index_MCP":   40.0,
        "index_PIP":   40.0,
        "middle_MCP":  10.0,
        "middle_PIP":  10.0,
        "ring_MCP":    10.0,
        "ring_PIP":    10.0,
        "pinky_MCP":   10.0,
        "pinky_PIP":   10.0,
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

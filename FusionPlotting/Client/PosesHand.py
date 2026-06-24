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
# Figures
# ---------------------------------------------------------------------------
FIGURES: dict[str, dict[str, float]] = {

    # All joints at rest.
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

    # PoseControl PC1 — power grasp (Santello et al. 1998, ~50% variance).
    # Uniform deep flexion of all fingers, thumb in full opposition.
    "power_grasp": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   65.0,
        "thumb_CMC2":   0.0,
        "thumb_MCP":    50.0,
        "thumb_IP":     40.0,
        "index_spread": -2.0,
        "ring_spread":   2.0,
        "pinky_spread":  2.0,
        "index_MCP":    55.0,
        "index_PIP":    65.0,
        "middle_MCP":   55.0,
        "middle_PIP":   65.0,
        "ring_MCP":     55.0,
        "ring_PIP":     65.0,
        "pinky_MCP":    55.0,
        "pinky_PIP":    65.0,
    },

    # PC1 lightly closed (~50% of full power-grasp angles).
    "power_grasp_light": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   35.0,
        "thumb_CMC2":   0.0,
        "thumb_MCP":    25.0,
        "thumb_IP":     20.0,
        "index_spread": -1.0,
        "ring_spread":   1.0,
        "pinky_spread":  1.0,
        "index_MCP":    30.0,
        "index_PIP":    35.0,
        "middle_MCP":   30.0,
        "middle_PIP":   35.0,
        "ring_MCP":     30.0,
        "ring_PIP":     35.0,
        "pinky_MCP":    30.0,
        "pinky_PIP":    35.0,
    },

    # PoseControl PC2 — precision pinch (Santello et al. 1998, ~30% variance).
    # Index+thumb opposed; ring/pinky remain closed.
    "precision_pinch": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   50.0,
        "thumb_CMC2":   12.0,
        "thumb_MCP":    35.0,
        "thumb_IP":     25.0,
        "index_spread": -1.0,
        "ring_spread":   1.0,
        "pinky_spread":  1.0,
        "index_MCP":    22.0,
        "index_PIP":    37.0,
        "middle_MCP":   22.0,
        "middle_PIP":   47.0,
        "ring_MCP":     58.0,
        "ring_PIP":     68.0,
        "pinky_MCP":    58.0,
        "pinky_PIP":    68.0,
    },

    # PC2 lightly closed (~50% of full precision-pinch angles).
    "precision_pinch_light": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   25.0,
        "thumb_CMC2":    6.0,
        "thumb_MCP":    17.0,
        "thumb_IP":     12.0,
        "index_spread": -0.5,
        "ring_spread":   0.5,
        "pinky_spread":  0.5,
        "index_MCP":    12.0,
        "index_PIP":    20.0,
        "middle_MCP":   12.0,
        "middle_PIP":   25.0,
        "ring_MCP":     28.0,
        "ring_PIP":     32.0,
        "pinky_MCP":    28.0,
        "pinky_PIP":    32.0,
    },

    # PassiveCompliance/Hand — guitar strumming.
    # Index, middle, ring close around the strings; pinky stays open.
    # Thumb held at neutral background (2°) as in the experiment.
    "guitar_strum": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   10.0,
        "thumb_CMC2":    5.0,
        "thumb_MCP":    10.0,
        "thumb_IP":      8.0,
        "index_spread": 0.0,
        "ring_spread":  0.0,
        "pinky_spread": 0.0,
        "index_MCP":   70.0,
        "index_PIP":   60.0,
        "middle_MCP":  70.0,
        "middle_PIP":  65.0,
        "ring_MCP":    70.0,
        "ring_PIP":    65.0,
        "pinky_MCP":   15.0,
        "pinky_PIP":   10.0,
    },

    # TunableCompliance/Hand — in-hand manipulation grasp.
    # Deep PC1 closure (all fingers at 120°) with wide spread for object rolling.
    # Thumb in full wrap configuration (CMC2 at limit, may be clamped by Fusion).
    "inhand_manipulation": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   60.0,
        "thumb_CMC2":  -180.0,
        "thumb_MCP":    90.0,
        "thumb_IP":     90.0,
        "index_spread": -10.0,
        "ring_spread":   10.0,
        "pinky_spread":  10.0,
        "index_MCP":   100.0,
        "index_PIP":    70.0,
        "middle_MCP":  100.0,
        "middle_PIP":   70.0,
        "ring_MCP":    100.0,
        "ring_PIP":     70.0,
        "pinky_MCP":   100.0,
        "pinky_PIP":    70.0,
    },

    # TunableCompliance/Hand — dynamic bottle grasping.
    # Full wrap closure (all fingers at 120°/120°), thumb at pre-grip angle.
    "dynamic_grasp": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   40.0,
        "thumb_CMC2":  -90.0,
        "thumb_MCP":    60.0,
        "thumb_IP":     60.0,
        "index_spread": -2.0,
        "ring_spread":   2.0,
        "pinky_spread":  2.0,
        "index_MCP":   120.0,
        "index_PIP":   120.0,
        "middle_MCP":  120.0,
        "middle_PIP":  120.0,
        "ring_MCP":    120.0,
        "ring_PIP":    120.0,
        "pinky_MCP":   120.0,
        "pinky_PIP":   120.0,
    },

    # ProprioceptiveSensing/Hand — compliance probing grasp.
    # Four fingers hold at PC1; thumb probes with variable stiffness.
    "proprioceptive_probing": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   100.0,
        "thumb_CMC2":  -70.0,
        "thumb_MCP":    120.0,
        "thumb_IP":     120.0,
        "index_spread":  0.0,
        "ring_spread":   0.0,
        "pinky_spread":  0.0,
        "index_MCP":    70.0,
        "index_PIP":    80.0,
        "middle_MCP":   70.0,
        "middle_PIP":   80.0,
        "ring_MCP":     70.0,
        "ring_PIP":     80.0,
        "pinky_MCP":    70.0,
        "pinky_PIP":    80.0,
    },

    # ADAPT-StiffControl — apple grasping.
    # Full-hand PC1 closure with wide spread for the apple shape.
    "apple_grasp": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   90.0,
        "thumb_CMC2":  -60.0,
        "thumb_MCP":    100.0,
        "thumb_IP":     100.0,
        "index_spread": -5.0,
        "ring_spread":   5.0,
        "pinky_spread":  5.0,
        "index_MCP":    90.0,
        "index_PIP":   110.0,
        "middle_MCP":   90.0,
        "middle_PIP":  110.0,
        "ring_MCP":     90.0,
        "ring_PIP":    110.0,
        "pinky_MCP":    90.0,
        "pinky_PIP":   110.0,
    },

    # PassiveCompliance/Hand — weight compliance.
    # Index bent at 40° (the loaded finger), others lightly flexed at 10°.
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

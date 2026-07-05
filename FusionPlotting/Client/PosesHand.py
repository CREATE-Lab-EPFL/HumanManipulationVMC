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
    index_MCP,  index_PIP,  index_DIP
    middle_MCP, middle_PIP, middle_DIP
    ring_MCP,   ring_PIP,   ring_DIP
    pinky_MCP,  pinky_PIP,  pinky_DIP
All angles in degrees, positive = flexion (DIP negation handled transparently).
Mechanical constraint: PIP = DIP always (tendon coupling).
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

# ---------------------------------------------------------------------------
# Figures — in paper order
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
        "index_DIP":    0.0,
        "middle_MCP":   0.0,
        "middle_PIP":   0.0,
        "middle_DIP":   0.0,
        "ring_MCP":     0.0,
        "ring_PIP":     0.0,
        "ring_DIP":     0.0,
        "pinky_MCP":    0.0,
        "pinky_PIP":    0.0,
        "pinky_DIP":    0.0,
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
        "index_PIP":    45.0,
        "index_DIP":    45.0,
        "middle_MCP":   55.0,
        "middle_PIP":   45.0,
        "middle_DIP":   45.0,
        "ring_MCP":     55.0,
        "ring_PIP":     45.0,
        "ring_DIP":     45.0,
        "pinky_MCP":    55.0,
        "pinky_PIP":    45.0,
        "pinky_DIP":    45.0,
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
        "index_PIP":    22.0,
        "index_DIP":    22.0,
        "middle_MCP":   30.0,
        "middle_PIP":   22.0,
        "middle_DIP":   22.0,
        "ring_MCP":     30.0,
        "ring_PIP":     22.0,
        "ring_DIP":     22.0,
        "pinky_MCP":    30.0,
        "pinky_PIP":    22.0,
        "pinky_DIP":    22.0,
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
        "index_PIP":    35.0,
        "index_DIP":    35.0,
        "middle_MCP":   22.0,
        "middle_PIP":   35.0,
        "middle_DIP":   35.0,
        "ring_MCP":     60.0,
        "ring_PIP":     48.0,
        "ring_DIP":     48.0,
        "pinky_MCP":    60.0,
        "pinky_PIP":    48.0,
        "pinky_DIP":    48.0,
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
        "index_PIP":     8.0,
        "index_DIP":     8.0,
        "middle_MCP":   12.0,
        "middle_PIP":    8.0,
        "middle_DIP":    8.0,
        "ring_MCP":     28.0,
        "ring_PIP":     20.0,
        "ring_DIP":     20.0,
        "pinky_MCP":    28.0,
        "pinky_PIP":    20.0,
        "pinky_DIP":    20.0,
    },

    # PassiveCompliance/Hand — guitar strumming.
    # Index, middle, ring close around the strings; pinky stays slightly bent.
    # Thumb held at gentle neutral as in the experiment.
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
        "index_DIP":   60.0,
        "middle_MCP":  70.0,
        "middle_PIP":  60.0,
        "middle_DIP":  60.0,
        "ring_MCP":    70.0,
        "ring_PIP":    60.0,
        "ring_DIP":    60.0,
        "pinky_MCP":   15.0,
        "pinky_PIP":   12.0,
        "pinky_DIP":   12.0,
    },

    # PassiveCompliance/Hand — weight compliance.
    # Index bent (the loaded finger), others lightly flexed.
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
        "index_MCP":   50.0,
        "index_PIP":   40.0,
        "index_DIP":   40.0,
        "middle_MCP":  10.0,
        "middle_PIP":   8.0,
        "middle_DIP":   8.0,
        "ring_MCP":    10.0,
        "ring_PIP":     8.0,
        "ring_DIP":     8.0,
        "pinky_MCP":   10.0,
        "pinky_PIP":    8.0,
        "pinky_DIP":    8.0,
    },

    # TunableCompliance/Hand — in-hand manipulation.
    # All fingers deeply closed; thumb wraps fully (CMC2 near its negative limit).
    # Wide finger spread to cradle an object being rolled in the palm.
    "inhand_manipulation": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   35.0,
        "thumb_CMC2":  -55.0,
        "thumb_MCP":    42.0,
        "thumb_IP":     35.0,
        "index_spread": -8.0,
        "ring_spread":   8.0,
        "pinky_spread":  8.0,
        "index_MCP":    60.0,
        "index_PIP":    39.0,
        "index_DIP":    39.0,
        "middle_MCP":   60.0,
        "middle_PIP":   39.0,
        "middle_DIP":   39.0,
        "ring_MCP":     60.0,
        "ring_PIP":     39.0,
        "ring_DIP":     39.0,
        "pinky_MCP":    60.0,
        "pinky_PIP":    39.0,
        "pinky_DIP":    39.0,
    },

    # TunableCompliance/Hand — dynamic bottle grasping.
    # Deep closure with thumb angled across the palm for a cylindrical grip.
    "dynamic_grasp": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   30.0,
        "thumb_CMC2":  -35.0,
        "thumb_MCP":    33.0,
        "thumb_IP":     27.0,
        "index_spread": -2.0,
        "ring_spread":   2.0,
        "pinky_spread":  2.0,
        "index_MCP":    42.0,
        "index_PIP":    35.0,
        "index_DIP":    35.0,
        "middle_MCP":   42.0,
        "middle_PIP":   35.0,
        "middle_DIP":   35.0,
        "ring_MCP":     42.0,
        "ring_PIP":     35.0,
        "ring_DIP":     35.0,
        "pinky_MCP":    42.0,
        "pinky_PIP":    35.0,
        "pinky_DIP":    35.0,
    },

    # ProprioceptiveSensing/Hand — compliance probing.
    # Fingers hold a moderate grasp; thumb extends to probe object stiffness.
    "proprioceptive_probing": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   70.0,
        "thumb_CMC2":  -30.0,
        "thumb_MCP":    50.0,
        "thumb_IP":     30.0,
        "index_spread":  0.0,
        "ring_spread":   0.0,
        "pinky_spread":  0.0,
        "index_MCP":    42.0,
        "index_PIP":    50.0,
        "index_DIP":    50.0,
        "middle_MCP":   42.0,
        "middle_PIP":   50.0,
        "middle_DIP":   50.0,
        "ring_MCP":     42.0,
        "ring_PIP":     50.0,
        "ring_DIP":     50.0,
        "pinky_MCP":    42.0,
        "pinky_PIP":    50.0,
        "pinky_DIP":    50.0,
    },

    # ADAPT-StiffControl — apple grasping.
    # Full-hand spherical grasp; fingers spread wide to conform to a round object.
    "apple_grasp": {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   75.0,
        "thumb_CMC2":  0.0,
        "thumb_MCP":    50.0,
        "thumb_IP":     40.0,
        "index_spread": -5.0,
        "ring_spread":   5.0,
        "pinky_spread":  5.0,
        "index_MCP":    52.0,
        "index_PIP":    39.0,
        "index_DIP":    39.0,
        "middle_MCP":   52.0,
        "middle_PIP":   39.0,
        "middle_DIP":   39.0,
        "ring_MCP":     52.0,
        "ring_PIP":     39.0,
        "ring_DIP":     39.0,
        "pinky_MCP":    52.0,
        "pinky_PIP":    39.0,
        "pinky_DIP":    39.0,
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

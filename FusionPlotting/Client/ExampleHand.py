"""
Example: set joint angles for the ADAPT Hand Fusion model.

Joint names and sign conventions follow the code (hand_params.py JOINT_LIMITS):
  - Finger MCP/PIP/DIP: positive = flexion  (limits 0 → 1.5 / 1.2 rad)
  - index_spread:        negative = adduction toward middle  (limits -0.3 → 0 rad)
  - ring/pinky_spread:   positive = adduction toward middle  (limits 0 → 0.3 rad)
  - wrist_pitch:         positive = flexion
  - wrist_yaw:           positive = ulnar deviation

The add-in (FusionConventions.py) handles any sign flips and name mapping
before sending values to Fusion — you never need to manually negate values here.

After running the add-in, check ~/FusionBridge/joints_discovery.json for the
actual Fusion joint names, then update FusionConventions.py accordingly.
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

# ---------------------------------------------------------------------------
# Values in code convention (degrees).  Signs match hand_params.py JOINT_LIMITS.
# ---------------------------------------------------------------------------
HAND_JOINTS: dict[str, float] = {
    # --- Wrist ---
    "wrist_pitch":    10.0,   # + = flexion
    "wrist_yaw":      10.0,   # + = ulnar deviation

    # --- Thumb ---
    "thumb_CMC1":    30.0,
    "thumb_CMC2":    20.0,
    "thumb_MCP":     15.0,
    "thumb_IP":      10.0,

    # --- Spread (index goes negative, ring/pinky go positive) ---
    "index_spread":   10.0,
    "ring_spread":    5.0,
    "pinky_spread":   5.0,

    # --- Fingers: + = flexion ---
    "index_MCP":     45.0,
    "index_PIP":     30.0,

    "middle_MCP":    45.0,
    "middle_PIP":    30.0,

    "ring_MCP":      40.0,
    "ring_PIP":      25.0,
    
    "pinky_MCP":     35.0,
    "pinky_PIP":     20.0,
}

# ---------------------------------------------------------------------------

client = JointClient()

print("Writing hand targets (code convention, degrees):")
for name, val in HAND_JOINTS.items():
    print(f"  {name}: {val}")

client.write_targets(HAND_JOINTS, angle_unit="degrees", length_unit="mm")

print(f"\nBridge file: {client.bridge_path}")
print("Waiting one poll cycle for Fusion to respond…")
time.sleep(0.5)

current = client.read_current()
if current:
    print("\nValues reported by Fusion add-in (code convention):")
    for name, val in current.items():
        print(f"  {name}: {val:.2f}")
else:
    print("\nNo values read back yet.")
    print("Check ~/FusionBridge/joints_discovery.json for actual Fusion joint names,")
    print("then update FusionConventions.py in the add-in folder.")

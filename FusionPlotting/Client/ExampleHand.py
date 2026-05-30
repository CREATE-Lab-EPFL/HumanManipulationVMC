"""
Example: set joint angles for the ADAPT Hand Fusion model.

The joint names must match exactly what appears in the Fusion 360 browser
under:  CREATE Lab > Bioinspired Robotic Hands > Lorenzo Vignoli > [Hand design]

To find the exact names: open the Joints folder in the Fusion browser, hover
over each joint — the tooltip shows its name, or right-click > Properties.
Update HAND_JOINTS below to match.
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

# ---------------------------------------------------------------------------
# Joint name -> target angle in degrees (revolute) or mm (slider/spread).
# Names below are placeholders — replace with your Fusion browser names.
# ---------------------------------------------------------------------------
HAND_JOINTS: dict[str, float] = {
    # --- Wrist (2 revolute DOF) ---
    "Wrist_Flex":    0.0,
    "Wrist_Abd":      0.0,

    # --- Thumb (4 revolute DOF) ---
    "Thumb_CMC1":    30.0,
    "Thumb_CMC2":    20.0,
    "Thumb_MCP":     15.0,
    "Thumb_IP":      10.0,

    # --- Index finger ---
    "Index_Spread":   0.0,   # abduction — revolute or slider depending on model
    "Index_MCP":     45.0,
    "Index_PIP":     30.0,

    # --- Middle finger ---
    "Middle_Spread":  0.0,
    "Middle_MCP":    45.0,
    "Middle_PIP":    30.0,

    # --- Ring finger ---
    "Ring_Spread":    0.0,
    "Ring_MCP":      40.0,
    "Ring_PIP":      25.0,

    # --- Pinky finger ---
    "Pinky_Spread":   0.0,
    "Pinky_MCP":     35.0,
    "Pinky_PIP":     20.0,
}

# ---------------------------------------------------------------------------

client = JointClient()

print("Writing hand targets:")
for name, val in HAND_JOINTS.items():
    print(f"  {name}: {val}")

client.write_targets(HAND_JOINTS, angle_unit="degrees", length_unit="mm")

print(f"\nBridge file: {client.bridge_path}")
print("Waiting one poll cycle for Fusion to respond…")
time.sleep(0.5)

current = client.read_current()
if current:
    print("\nValues reported by Fusion add-in:")
    for name, val in current.items():
        print(f"  {name}: {val:.2f}")
else:
    print("\nNo values read back yet — is the add-in running in Fusion?")

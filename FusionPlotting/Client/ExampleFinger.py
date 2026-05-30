"""
Example: set joint angles for the single-finger Fusion model.

The joint names must match exactly what appears in the Fusion 360 browser
under:  CREATE Lab > Bioinspired Robotic Hands > Lorenzo Vignoli > [Finger design]

To find the exact names: open the Joints folder in the Fusion browser, hover
over each joint — the tooltip shows its name, or right-click > Properties.
Update FINGER_JOINTS below to match.
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from JointClient import JointClient

# ---------------------------------------------------------------------------
# Joint name -> target angle (degrees).  Update names to match your model.
# ---------------------------------------------------------------------------
FINGER_JOINTS: dict[str, float] = {
    "MCP": 20.0,   # metacarpophalangeal — revolute
    "PIP": 10.0,   # proximal interphalangeal — revolute
    "DIP": 10.0,   # distal interphalangeal (mimic; may be read-only in Fusion)
}

# ---------------------------------------------------------------------------

client = JointClient()

print("Writing finger targets:")
for name, val in FINGER_JOINTS.items():
    print(f"  {name}: {val} deg")

client.write_targets(FINGER_JOINTS, angle_unit="degrees", length_unit="mm")

print(f"\nBridge file: {client.bridge_path}")
print("Waiting one poll cycle for Fusion to respond…")
time.sleep(0.5)

current = client.read_current()
if current:
    print("\nValues reported by Fusion add-in:")
    for name, val in current.items():
        print(f"  {name}: {val:.2f} deg")
else:
    print("\nNo values read back yet — is the add-in running in Fusion?")

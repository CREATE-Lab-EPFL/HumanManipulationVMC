"""
Motor Configuration for ADAPT Hand

Wrist motors (hardware IDs 13, 14) run in position mode and are invisible to
all control code. Only the 13 torque-controlled motors are listed here.

Hardware ordering: the dynamixel_node publishes positions indexed by hardware ID.
Software ordering: the canonical ordering used by all VMC/kinematics code.
"""

import numpy as np

NUM_MOTORS = 13

# Hardware motor IDs for the 13 torque-controlled motors — edit to match hardware
MOTOR_IDS = {
    'thumb_CMC1': 5,
    'thumb_CMC2': 4,
    'thumb_MCP':  7,
    'thumb_IP':   6,
    'spread':     12,
    'index_MCP':  3,
    'index_PIP':  8,
    'middle_MCP': 2,
    'middle_PIP': 9,
    'ring_MCP':   1,
    'ring_PIP':   10,
    'pinky_MCP':  0,
    'pinky_PIP':  11,
}

# Software internal ordering (fixed — do not modify)
SOFTWARE_MOTOR_ORDER = [
    'thumb_CMC1',  # [0]
    'thumb_CMC2',  # [1]
    'thumb_MCP',   # [2]
    'thumb_IP',    # [3]
    'spread',      # [4]
    'index_MCP',   # [5]
    'index_PIP',   # [6]
    'middle_MCP',  # [7]
    'middle_PIP',  # [8]
    'ring_MCP',    # [9]
    'ring_PIP',    # [10]
    'pinky_MCP',   # [11]
    'pinky_PIP',   # [12]
]

MOTOR_SLICES = {
    'thumb':  slice(0, 4),
    'spread': 4,
    'index':  slice(5, 7),
    'middle': slice(7, 9),
    'ring':   slice(9, 11),
    'pinky':  slice(11, 13),
}


def hardware_to_software(q_hardware):
    """Convert motor angles from hardware ordering (by hw ID) to software ordering."""
    q_software = np.zeros(NUM_MOTORS)
    for i, motor_name in enumerate(SOFTWARE_MOTOR_ORDER):
        q_software[i] = q_hardware[MOTOR_IDS[motor_name]]
    return q_software


def software_to_hardware(q_software):
    """Convert motor angles from software ordering to hardware ordering (by hw ID)."""
    q_hardware = np.zeros(NUM_MOTORS)
    for i, motor_name in enumerate(SOFTWARE_MOTOR_ORDER):
        q_hardware[MOTOR_IDS[motor_name]] = q_software[i]
    return q_hardware


def get_motor_slice(component):
    if component not in MOTOR_SLICES:
        raise ValueError(f"Unknown component: {component}. Valid: {list(MOTOR_SLICES.keys())}")
    return MOTOR_SLICES[component]


def unpack_motors(q_motor):
    """Unpack motor array into (thumb[4], spread, index[2], middle[2], ring[2], pinky[2])."""
    return (q_motor[MOTOR_SLICES['thumb']],
            q_motor[MOTOR_SLICES['spread']],
            q_motor[MOTOR_SLICES['index']],
            q_motor[MOTOR_SLICES['middle']],
            q_motor[MOTOR_SLICES['ring']],
            q_motor[MOTOR_SLICES['pinky']])


def validate_motor_config():
    motor_ids = list(MOTOR_IDS.values())
    if len(motor_ids) != len(set(motor_ids)):
        raise ValueError("Duplicate motor IDs found in MOTOR_IDS!")
    if min(motor_ids) < 0 or max(motor_ids) >= NUM_MOTORS:
        raise ValueError(f"Motor IDs must be in range [0, {NUM_MOTORS-1}]")
    for motor_name in SOFTWARE_MOTOR_ORDER:
        if motor_name not in MOTOR_IDS:
            raise ValueError(f"Motor '{motor_name}' not found in MOTOR_IDS")
    if set(MOTOR_IDS.keys()) != set(SOFTWARE_MOTOR_ORDER):
        raise ValueError("MOTOR_IDS and SOFTWARE_MOTOR_ORDER contain different motors!")
    print("✓ Motor configuration is valid!")


if __name__ == "__main__":
    validate_motor_config()
    print("\nMotor Configuration:")
    print("=" * 60)
    for i, motor_name in enumerate(SOFTWARE_MOTOR_ORDER):
        print(f"  Software[{i:2d}] = {motor_name:20s} -> Hardware ID {MOTOR_IDS[motor_name]:2d}")
    print("=" * 60)

    print("\nTesting conversion functions...")
    q_hw = np.arange(NUM_MOTORS) * 0.1
    q_sw = hardware_to_software(q_hw)
    q_hw_back = software_to_hardware(q_sw)
    print(f"Conversion correct: {np.allclose(q_hw, q_hw_back)}")

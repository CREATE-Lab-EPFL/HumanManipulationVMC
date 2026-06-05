"""
Motor Configuration for ADAPT Hand

Defines the mapping between physical motor IDs and software's internal representation.
Modify MOTOR_IDS to match your actual hardware setup.
"""

import numpy as np

NUM_MOTORS = 15         # total motors (position feedback from all 15)
NUM_TORQUE_MOTORS = 13  # torque-controlled motors (hw 0-12, sw indices 2-14)
WRIST_HW_IDS = {13, 14} # wrist hardware IDs — run in position mode, not torque

# Hardware motor IDs - EDIT THIS to match your hardware
MOTOR_IDS = {
    'wrist_motor1': 13,
    'wrist_motor2': 14,
    'thumb_CMC1': 5,
    'thumb_CMC2': 4,
    'thumb_MCP': 7,
    'thumb_IP': 6,
    'spread': 12,
    'index_MCP': 3,
    'index_PIP': 8,
    'middle_MCP': 2,
    'middle_PIP': 9,
    'ring_MCP': 1,
    'ring_PIP': 10,
    'pinky_MCP': 0,
    'pinky_PIP': 11,
}

# Software internal ordering (fixed - do not modify)
SOFTWARE_MOTOR_ORDER = [
    'wrist_motor1',   # [0]
    'wrist_motor2',   # [1]
    'thumb_CMC1',     # [2]
    'thumb_CMC2',     # [3]
    'thumb_MCP',      # [4]
    'thumb_IP',       # [5]
    'spread',         # [6]
    'index_MCP',      # [7]
    'index_PIP',      # [8]
    'middle_MCP',     # [9]
    'middle_PIP',     # [10]
    'ring_MCP',       # [11]
    'ring_PIP',       # [12]
    'pinky_MCP',      # [13]
    'pinky_PIP',      # [14]
]

MOTOR_SLICES = {
    'wrist': slice(0, 2),
    'thumb': slice(2, 6),
    'spread': 6,
    'index': slice(7, 9),
    'middle': slice(9, 11),
    'ring': slice(11, 13),
    'pinky': slice(13, 15),
}


def hardware_to_software(q_hardware):
    """Convert motor angles from hardware ordering to software ordering."""
    q_software = np.zeros(NUM_MOTORS)
    for i, motor_name in enumerate(SOFTWARE_MOTOR_ORDER):
        hardware_id = MOTOR_IDS[motor_name]
        q_software[i] = q_hardware[hardware_id]
    return q_software


def software_to_hardware(q_software):
    """Convert motor angles from software ordering to hardware ordering."""
    q_hardware = np.zeros(NUM_MOTORS)
    for i, motor_name in enumerate(SOFTWARE_MOTOR_ORDER):
        hardware_id = MOTOR_IDS[motor_name]
        q_hardware[hardware_id] = q_software[i]
    return q_hardware


def get_motor_slice(component):
    """Get the slice for a specific hand component."""
    if component not in MOTOR_SLICES:
        raise ValueError(f"Unknown component: {component}. Valid: {list(MOTOR_SLICES.keys())}")
    return MOTOR_SLICES[component]


def unpack_motors(q_motor):
    """Unpack motor array into separate components."""
    return (q_motor[MOTOR_SLICES['wrist']],
            q_motor[MOTOR_SLICES['thumb']],
            q_motor[MOTOR_SLICES['spread']],
            q_motor[MOTOR_SLICES['index']],
            q_motor[MOTOR_SLICES['middle']],
            q_motor[MOTOR_SLICES['ring']],
            q_motor[MOTOR_SLICES['pinky']])


def validate_motor_config():
    """Validate that the motor configuration is consistent."""
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
        hardware_id = MOTOR_IDS[motor_name]
        print(f"  Software[{i:2d}] = {motor_name:20s} -> Hardware ID {hardware_id:2d}")
    print("=" * 60)

    print("\nTesting conversion functions...")
    q_hw = np.arange(NUM_MOTORS) * 0.1
    q_sw = hardware_to_software(q_hw)
    q_hw_back = software_to_hardware(q_sw)
    print(f"Hardware input:  {q_hw}")
    print(f"Software array:  {q_sw}")
    print(f"Hardware output: {q_hw_back}")
    print(f"Conversion correct: {np.allclose(q_hw, q_hw_back)}")

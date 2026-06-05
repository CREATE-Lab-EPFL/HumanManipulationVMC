## Go Home Script
## ===============
## Sends the hand to the home/reference position using Dynamixel SDK directly.
## Uses ABSOLUTE encoder positions (not relative).
## After reaching home, switches back to torque mode (ready for VMC control).
##
## Usage:
##   python3 go_home.py          # Send to home position, then switch to torque mode
##   python3 go_home.py --read   # Read and print current absolute positions

import argparse
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS

# Dynamixel XL330 settings
BAUDRATE = 2000000
DEVICE_PORT = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0

# Control table addresses (XL330-M288)
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
ADDR_GOAL_CURRENT = 102
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112

# Position threshold for "arrived" detection (encoder ticks)
POSITION_THRESHOLD = 10

# Motor IDs
MOTOR_IDS = list(range(15))

# Home position (ABSOLUTE encoder values)
# To update: run with --read when hand is in desired pose, then copy values here
HOME_POSITION = {
    0: 813,
    1: 784,
    2: 507,
    3: 753,
    4: 1257,
    5: 384,
    6: 3357,
    7: 3167,
    8: 2857,
    9: 2845,
    10: 2931,
    11: 2766,
    12: 1862,
    13: 1656,
    14: 2299,
}

# Velocity for going home (lower = slower, safer)
PROFILE_VELOCITY = 40  # units: 0.229 rpm per unit


def init_dynamixel():
    """Initialize port and packet handler."""
    port_handler = PortHandler(DEVICE_PORT)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise RuntimeError(f"Failed to open port {DEVICE_PORT}")

    if not port_handler.setBaudRate(BAUDRATE):
        raise RuntimeError(f"Failed to set baudrate {BAUDRATE}")

    return port_handler, packet_handler


def read_positions(port_handler, packet_handler, verbose=False):
    """Read current absolute positions from all motors."""
    positions = {}
    for motor_id in MOTOR_IDS:
        pos, result, error = packet_handler.read4ByteTxRx(
            port_handler, motor_id, ADDR_PRESENT_POSITION)
        if result != COMM_SUCCESS:
            if verbose:
                print(f"Motor {motor_id}: Read failed - {packet_handler.getTxRxResult(result)}")
        elif error != 0:
            if verbose:
                print(f"Motor {motor_id}: Error - {packet_handler.getRxPacketError(error)}")
        else:
            # Handle signed 32-bit value
            if pos > 0x7FFFFFFF:
                pos = pos - 0x100000000
            positions[motor_id] = pos
    return positions


def set_position_mode(port_handler, packet_handler):
    """Set all motors to extended position control mode."""
    for motor_id in MOTOR_IDS:
        # Disable torque first (required to change operating mode)
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
        # Set extended position control mode (value = 4) for multi-turn support
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_OPERATING_MODE, 4)
        # Set profile velocity
        packet_handler.write4ByteTxRx(port_handler, motor_id, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY)
        # Enable torque
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
    print("All motors set to extended position control mode")


def go_home(port_handler, packet_handler):
    """Send all motors to home position."""
    sync_write = GroupSyncWrite(port_handler, packet_handler, ADDR_GOAL_POSITION, 4)

    for motor_id in MOTOR_IDS:
        goal = HOME_POSITION[motor_id]
        param = [
            goal & 0xFF,
            (goal >> 8) & 0xFF,
            (goal >> 16) & 0xFF,
            (goal >> 24) & 0xFF
        ]
        sync_write.addParam(motor_id, param)

    result = sync_write.txPacket()
    if result != COMM_SUCCESS:
        print(f"Sync write failed: {packet_handler.getTxRxResult(result)}")
    else:
        print("Home position command sent!")

    sync_write.clearParam()


def disable_torque(port_handler, packet_handler):
    """Disable torque on all motors."""
    for motor_id in MOTOR_IDS:
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
    print("Torque disabled on all motors")


def wait_until_reached(port_handler, packet_handler, timeout=5.0):
    """Wait until all motors reach their target positions."""
    import time

    # Initial delay to let motors start moving
    time.sleep(0.5)

    start_time = time.time()

    while time.time() - start_time < timeout:
        positions = read_positions(port_handler, packet_handler)

        # Need successful reads from all motors
        if len(positions) < len(MOTOR_IDS):
            time.sleep(0.1)
            continue

        all_reached = True
        for motor_id in MOTOR_IDS:
            error = abs(positions[motor_id] - HOME_POSITION[motor_id])
            if error > POSITION_THRESHOLD:
                all_reached = False
                break

        if all_reached:
            return True

        time.sleep(0.1)

    return False


def set_torque_mode(port_handler, packet_handler):
    """Set all motors to current (torque) control mode with zero torque."""
    for motor_id in MOTOR_IDS:
        # Disable torque first (required to change operating mode)
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
        # Set current control mode (value = 0)
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_OPERATING_MODE, 0)
        # Set goal current to 0
        packet_handler.write2ByteTxRx(port_handler, motor_id, ADDR_GOAL_CURRENT, 0)
        # Enable torque
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
    print("All motors set to torque mode (zero torque)")


def main():
    parser = argparse.ArgumentParser(description='Go to home position or read current positions')
    parser.add_argument('--read', action='store_true', help='Read and print current absolute positions')
    parser.add_argument('--disable', action='store_true', help='Disable torque after reaching home')
    args = parser.parse_args()

    port_handler, packet_handler = init_dynamixel()

    try:
        if args.read:
            positions = read_positions(port_handler, packet_handler, verbose=True)
            print("\nCurrent absolute positions (copy to HOME_POSITION):")
            print("HOME_POSITION = {")
            for motor_id in MOTOR_IDS:
                if motor_id in positions:
                    print(f"    {motor_id}: {positions[motor_id]},")
            print("}")
        else:
            set_position_mode(port_handler, packet_handler)
            go_home(port_handler, packet_handler)
            print("Moving to home position...")

            # Wait for motors to reach home
            if wait_until_reached(port_handler, packet_handler):
                print("Home position reached!")
            else:
                print("Warning: Timeout waiting for home position")

            # Switch back to torque mode (ready for VMC control)
            set_torque_mode(port_handler, packet_handler)

            if args.disable:
                disable_torque(port_handler, packet_handler)

    finally:
        port_handler.closePort()


if __name__ == '__main__':
    main()

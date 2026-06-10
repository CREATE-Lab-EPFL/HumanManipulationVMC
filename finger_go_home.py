## Go Home Script - Single Finger (2 motors)
## ==========================================
## Sends the two finger motors to the home/reference position using the
## Dynamixel SDK directly (no ROS node required).
## Uses ABSOLUTE encoder positions (not relative).
## After reaching home, switches back to current (torque) mode so the ROS
## Dynamixel node can take over.
##
## Usage:
##   python3 go_home.py          # Send to home, switch to torque mode
##   python3 go_home.py --read   # Read and print current absolute positions

import argparse
import time
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# ── Hardware settings ─────────────────────────────────────────────────────────
BAUDRATE    = 1000000
DEVICE_PORT = '/dev/ttyUSB0'
PROTOCOL    = 2.0

# Motor IDs (must match motor_ids param of dynamixel_node, default: 1, 2)
MOTOR_MCP = 1   # first motor  → joint_positions[0]
MOTOR_PIP = 2   # second motor → joint_positions[1]
MOTOR_IDS = [MOTOR_MCP, MOTOR_PIP]

# ── Control table addresses (XL330-M288) ─────────────────────────────────────
ADDR_TORQUE_ENABLE    = 64
ADDR_OPERATING_MODE   = 11
ADDR_GOAL_POSITION    = 116
ADDR_GOAL_CURRENT     = 102
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112

POSITION_THRESHOLD = 10      # ticks — "close enough" to home
PROFILE_VELOCITY   = 40      # 0.229 rpm per unit → ~9 rpm (slow, safe)

# ── Home position (ABSOLUTE encoder ticks) ───────────────────────────────────
# To update: run  python3 go_home.py --read  with the finger in the desired pose.
HOME_POSITION = {
    MOTOR_MCP: 2657,
    MOTOR_PIP: 4038,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def init():
    ph = PortHandler(DEVICE_PORT)
    pk = PacketHandler(PROTOCOL)
    if not ph.openPort():
        raise RuntimeError(f"Cannot open {DEVICE_PORT}")
    if not ph.setBaudRate(BAUDRATE):
        raise RuntimeError(f"Cannot set baudrate {BAUDRATE}")
    return ph, pk


def read_positions(ph, pk):
    positions = {}
    for mid in MOTOR_IDS:
        pos, result, error = pk.read4ByteTxRx(ph, mid, ADDR_PRESENT_POSITION)
        if result == COMM_SUCCESS and error == 0:
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
            positions[mid] = pos
    return positions


def set_position_mode(ph, pk):
    for mid in MOTOR_IDS:
        pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 0)
        pk.write1ByteTxRx(ph, mid, ADDR_OPERATING_MODE, 4)   # extended position
        pk.write4ByteTxRx(ph, mid, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY)
        pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 1)
    print("Motors set to extended position mode")


def send_home(ph, pk):
    for mid in MOTOR_IDS:
        pk.write4ByteTxRx(ph, mid, ADDR_GOAL_POSITION, HOME_POSITION[mid])
    print("Home command sent")


def wait_until_home(ph, pk, timeout=10.0):
    time.sleep(0.5)
    t0 = time.time()
    while time.time() - t0 < timeout:
        pos = read_positions(ph, pk)
        if len(pos) == len(MOTOR_IDS):
            if all(abs(pos[mid] - HOME_POSITION[mid]) <= POSITION_THRESHOLD
                   for mid in MOTOR_IDS):
                return True
        time.sleep(0.05)
    return False


def set_torque_mode(ph, pk):
    for mid in MOTOR_IDS:
        pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 0)
        pk.write1ByteTxRx(ph, mid, ADDR_OPERATING_MODE, 0)   # current control
        pk.write2ByteTxRx(ph, mid, ADDR_GOAL_CURRENT, 0)
        pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 1)
    print("Motors set to torque mode (zero torque)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true',
                        help='Read and print current absolute positions')
    args = parser.parse_args()

    ph, pk = init()
    try:
        if args.read:
            pos = read_positions(ph, pk)
            print("Current absolute positions:")
            for mid in MOTOR_IDS:
                print(f"  Motor {mid}: {pos.get(mid, 'READ ERROR')} ticks")
            print("\nCopy to HOME_POSITION if this is the desired home pose.")
        else:
            set_position_mode(ph, pk)
            send_home(ph, pk)
            print("Moving to home...")

            if wait_until_home(ph, pk):
                print("Home reached.")
            else:
                print("Warning: timeout before reaching home.")

            set_torque_mode(ph, pk)

    finally:
        ph.closePort()


if __name__ == '__main__':
    main()

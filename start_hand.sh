#!/bin/bash
# Hand startup script
# ====================
# 1. Move all motors to home position (wrist stays in position mode after)
# 2. Set USB latency timer for maximum throughput
# 3. Start finger/thumb dynamixel node in torque (current) mode
# 4. Start wrist dynamixel node in position mode (holds home position)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] Moving hand to home position ==="
python3 "$SCRIPT_DIR/hand_go_home.py"

echo "=== [2/4] Setting USB latency timer ==="
sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer

echo "=== [3/4] Starting dynamixel node: finger/thumb motors (torque mode) ==="
ros2 run dynamixel_interface dynamixel_node \
    --ros-args -p baudrate:=2000000 \
    -p motor_ids:=[0,1,2,3,4,5,6,7,8,9,10,11,12] &
TORQUE_PID=$!

echo "=== [4/4] Starting dynamixel node: wrist motors (position mode, holding home) ==="
ros2 run dynamixel_interface dynamixel_node \
    --ros-args -p baudrate:=2000000 \
    -p motor_ids:=[13,14] \
    -p control_mode:=position &
POSITION_PID=$!

echo ""
echo "Hand is ready."
echo "  Torque node PID  : $TORQUE_PID   (hw motors 0-12, finger/thumb)"
echo "  Position node PID: $POSITION_PID  (hw motors 13-14, wrist)"
echo ""
echo "Press Ctrl+C to stop."
trap "kill $TORQUE_PID $POSITION_PID 2>/dev/null; echo 'Nodes stopped.'" INT TERM
wait

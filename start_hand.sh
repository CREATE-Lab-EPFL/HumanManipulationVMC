#!/bin/bash
# Hand startup script
# ====================
# 1. Move all motors to home position (wrist stays in position mode after)
# 2. Set USB latency timer for maximum throughput
# 3. Start single dynamixel node: finger/thumb motors in torque mode,
#    wrist motors (hw 13,14) held in position mode via position_motor_ids

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] Moving hand to home position ==="
python3 "$SCRIPT_DIR/hand_go_home.py"

echo "=== [2/3] Setting USB latency timer ==="
sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer

echo "=== [3/3] Starting dynamixel node (torque + position mixed mode) ==="
ros2 run dynamixel_interface dynamixel_node \
    --ros-args -p baudrate:=2000000 \
    -p motor_ids:=[0,1,2,3,4,5,6,7,8,9,10,11,12] \
    -p position_motor_ids:=[13,14] &
NODE_PID=$!

echo ""
echo "Hand is ready."
echo "  Node PID: $NODE_PID"
echo "  Torque motors  : hw 0-12  (finger/thumb, 13 motors)"
echo "  Position motors: hw 13-14 (wrist, held at home)"
echo ""
echo "Press Ctrl+C to stop."
trap "kill $NODE_PID 2>/dev/null; echo 'Node stopped.'" INT TERM
wait

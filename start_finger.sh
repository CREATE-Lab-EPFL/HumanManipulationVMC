#!/bin/bash
# Finger startup script
# =====================
# 1. Move both finger motors to home position
# 2. Set USB latency timer for maximum throughput
# 3. Start dynamixel node: MCP and PIP motors (hw 1, 2) in torque mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] Moving finger to home position ==="
python3 "$SCRIPT_DIR/finger_go_home.py"

echo "=== [2/3] Setting USB latency timer ==="
sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer

echo "=== [3/3] Starting dynamixel node (torque mode) ==="
ros2 run dynamixel_interface dynamixel_node \
    --ros-args -p baudrate:=1000000 \
    -p motor_ids:=[1,2] &
NODE_PID=$!

echo ""
echo "Finger is ready."
echo "  Node PID: $NODE_PID"
echo "  Torque motors: hw 1 (MCP), hw 2 (PIP)"
echo ""
echo "Press Ctrl+C to stop."
trap "kill $NODE_PID 2>/dev/null; echo 'Node stopped.'" INT TERM
wait

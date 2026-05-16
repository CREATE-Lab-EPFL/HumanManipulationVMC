"""
Methods — Elastic band stiffness benchmark (finger).

Validates the VMC stiffness composition model using physical elastic bands as
ground-truth torsional springs at MCP, PIP, and DIP joints simultaneously.
Motors are fully disconnected; the elastic bands alone provide the restoring torque.
The UR5 descends and returns while the load cell records the contact force.

Protocol (per run):
  1. UR5 moves to UR5_POSE.
  2. Wait SETTLE_TIME.
  3. UR5 descends UR5_DESCENT — record Phase='descent'.
  4. UR5 returns to UR5_POSE  — record Phase='ascent'.
  5. Repeat N_RUNS times.

Outputs: MethodsElastic/outputs/elastic_band/elastic_band_{spring}_run_N.csv
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import sys
import os
import time
import csv
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from UR5_codes.UR5_config import (
    UR5_POSE, UR5_IP,
    UR5_INIT_SPEED, UR5_INIT_ACCELERATION,
    UR5_DESCENT, UR5_DESCENT_SPEED,
    N_RUNS,
)

import rtde_control
import rtde_receive

# =============================================================================
# Experiment parameters
# =============================================================================
SETTLE_TIME    = 3.0    # [s]
COLLECTED_DATA = True
SHIFT_TPU      = 0.015  # [m] — start above UR5_POSE wrt PassiveCompliance
SPRING         = 'hard' # 'soft' or 'hard'

_HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Minimal ROS2 node — load cell only
# =============================================================================
class ForceListener(Node):
    def __init__(self):
        super().__init__('elastic_band_force_listener')
        self.force_N = 0.0
        self.create_subscription(
            Float64, '/force_normal',
            lambda msg: setattr(self, 'force_N', msg.data * 0.00980665), 10)


rclpy.init()
node = ForceListener()

# =============================================================================
# UR5 init
# =============================================================================
arm  = rtde_control.RTDEControlInterface(UR5_IP)
recv = rtde_receive.RTDEReceiveInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
node.get_logger().info('UR5 connected')

# =============================================================================
# Helpers
# =============================================================================
def output_path(run):
    folder = os.path.join(_HERE, 'outputs', 'elastic_band')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'elastic_band_{SPRING}_run_{run + 1}.csv')


def open_csv(run):
    fname = output_path(run)
    f = open(fname, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        'Time_s', 'Phase',
        'Force_N', 'UR5_Z_m', 'UR5_displacement_m',
    ])
    return f, w, fname


# =============================================================================
# Main loop — sequential (no state machine needed without motor control)
# =============================================================================
node.get_logger().info(f'Moving UR5 to UR5_POSE + SHIFT_TPU ...')
init_pose = UR5_POSE.copy()
init_pose[2] += SHIFT_TPU
arm.moveL(init_pose.tolist(), UR5_INIT_SPEED, UR5_INIT_ACCELERATION)

try:
    for run in range(N_RUNS):
        node.get_logger().info(f'Run {run + 1}/{N_RUNS} — settling ...')
        t_settle = time.time()
        while time.time() - t_settle < SETTLE_TIME:
            rclpy.spin_once(node, timeout_sec=0.005)

        csv_file, csv_writer, csv_filename = (None, None, None)
        if not COLLECTED_DATA:
            csv_file, csv_writer, csv_filename = open_csv(run)
            node.get_logger().info(f'Saving to: {csv_filename}')

        descent_target    = UR5_POSE.copy()
        descent_target[2] += SHIFT_TPU - UR5_DESCENT

        # ── Descent ──────────────────────────────────────────────────────────
        node.get_logger().info('Descending ...')
        t_start = time.time()
        arm_thread = threading.Thread(
            target=arm.moveL,
            args=(descent_target.tolist(), UR5_DESCENT_SPEED, UR5_INIT_ACCELERATION),
            daemon=True)
        arm_thread.start()

        while arm_thread.is_alive():
            rclpy.spin_once(node, timeout_sec=0.005)
            if not COLLECTED_DATA:
                ur5_z = recv.getActualTCPPose()[2]
                csv_writer.writerow([
                    f'{time.time() - t_start:.4f}', 'descent',
                    f'{node.force_N:.4f}',
                    f'{ur5_z:.6f}', f'{init_pose[2] - ur5_z:.6f}',
                ])

        # ── Ascent ───────────────────────────────────────────────────────────
        node.get_logger().info('Ascending ...')
        arm_thread = threading.Thread(
            target=arm.moveL,
            args=(init_pose.tolist(), UR5_DESCENT_SPEED, UR5_INIT_ACCELERATION),
            daemon=True)
        arm_thread.start()

        while arm_thread.is_alive():
            rclpy.spin_once(node, timeout_sec=0.005)
            if not COLLECTED_DATA:
                ur5_z = recv.getActualTCPPose()[2]
                csv_writer.writerow([
                    f'{time.time() - t_start:.4f}', 'ascent',
                    f'{node.force_N:.4f}',
                    f'{ur5_z:.6f}', f'{init_pose[2] - ur5_z:.6f}',
                ])

        if csv_file and not csv_file.closed:
            csv_file.close()
            node.get_logger().info(f'Data saved: {csv_filename}')

    node.get_logger().info('=== ALL RUNS COMPLETE ===')

except KeyboardInterrupt:
    node.get_logger().info('Interrupted')
finally:
    if csv_file is not None and not csv_file.closed:
        csv_file.close()
    arm.stopScript()
    node.destroy_node()
    rclpy.shutdown()

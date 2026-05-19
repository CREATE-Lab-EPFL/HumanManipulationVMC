"""
Passive compliance shaping — glissando with the ADAPT Hand.

Index and middle fingers are positioned side-by-side above a piano keyboard.
Both fingers are held pressed down (spring pulling to PRESS_POS) while the
UR5 slides along the keyboard (Y direction) for GLISSANDO_DISTANCE at
GLISSANDO_SPEED, producing a continuous glissando motion.

Protocol:
  1. UR5 moves to UR5_POSE_GLISSANDO_START; hand settles at press pose.
  2. Wait SETTLE_TIME.
  3. UR5 slides along GLISSANDO_DIRECTION for GLISSANDO_DISTANCE.
  4. UR5 returns to start.
  5. Repeat N_RUNS times.

Controller:
  - JointVMC  (K_ROT, B_ROT) — background joint regulation for all joints.
  - TaskVMC   (K_CART, B_CART) — index and middle fingertips held at PRESS_POS
    throughout the slide.

Outputs: PassiveCompliance/Hand/outputs/piano_glissando/run_N.csv
"""

import numpy as np
import rclpy
import sys
import os
import csv
import time
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../..'))
sys.path.insert(0, os.path.join(_HERE, 'HelperPianoMIDI'))

from VMCHand.HandController    import HandController, CONTROL_FREQUENCY
from VMCHand.HandVMCJointSpace import VMC as JointVMC
from VMCHand.HandVMCTaskSpace  import VMC as TaskVMC
from VMCHand.HandGravFricLim   import GravFricLim
from KinematicsHand.FK_Hand    import FK_motor2fingerPos
from ModelIDHand.motor_config  import MOTOR_SLICES
from UR5_codes.UR5_readPose    import UR5Receiver
from piano_config import (
    UR5_POSE_GLISSANDO_START,
    GLISSANDO_DIRECTION,
    GLISSANDO_DISTANCE,
    GLISSANDO_SPEED,
    UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCEL,
    PRESS_ANGLE_DEG, PIANO_FINGERS_GLISSANDO,
    K_CART, K_ROT, B_ROT, N_RUNS,
    B_CART_GLISSANDO       as B_CART,
    GLISSANDO_SETTLE_TIME  as SETTLE_TIME,
)

import rtde_control

# =============================================================================
# Collected data
# =============================================================================
COLLECTED_DATA = False

# =============================================================================
# Joint-space press pose
# =============================================================================
Q_HOME  = np.zeros(15)

Q_PRESS = np.zeros(15)
for _f in PIANO_FINGERS_GLISSANDO:
    Q_PRESS[MOTOR_SLICES[_f]] = np.deg2rad(PRESS_ANGLE_DEG)

# =============================================================================
# Cartesian targets from FK
# =============================================================================
_R = np.zeros(3)

REST_POS  = {f: np.array(FK_motor2fingerPos(Q_HOME,  f, 'DIP', _R)) for f in PIANO_FINGERS_GLISSANDO}
PRESS_POS = {f: np.array(FK_motor2fingerPos(Q_PRESS, f, 'DIP', _R)) for f in PIANO_FINGERS_GLISSANDO}

# =============================================================================
# Glissando UR5 waypoints
# =============================================================================
_dir_unit = GLISSANDO_DIRECTION[:3] / np.linalg.norm(GLISSANDO_DIRECTION[:3])
_dir6     = np.concatenate([_dir_unit, [0.0, 0.0, 0.0]])

GLISSANDO_END = UR5_POSE_GLISSANDO_START + _dir6 * GLISSANDO_DISTANCE

# =============================================================================
# Logging
# =============================================================================
LOG_EVERY  = max(1, int(CONTROL_FREQUENCY / 50))
FIELDNAMES = (
    [f'q_{i}'    for i in range(15)] +
    [f'qdot_{i}' for i in range(15)] +
    [f'tau_{i}'  for i in range(15)] +
    ['phase']
)

def _output_path(run):
    folder = os.path.join(_HERE, 'outputs', 'piano_glissando')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'run_{run}.csv')

# =============================================================================
# ROS2 / hand initialisation
# =============================================================================
rclpy.init()
controller    = HandController()
grav_fric_lim = GravFricLim()
recv          = UR5Receiver()

vmc_joint = JointVMC()
vmc_joint.set_stiffness(K_ROT)
vmc_joint.set_damping(B_ROT)

vmc_task = TaskVMC()
vmc_task.set_stiffness(0.0)
vmc_task.set_damping(0.0)
for _f in PIANO_FINGERS_GLISSANDO:
    vmc_task.springs[_f].stiffness = np.full(3, K_CART)
    vmc_task.dampers[_f].damping   = np.full(3, B_CART)
    vmc_task.targets[_f] = PRESS_POS[_f].copy()

# =============================================================================
# Control loop
# =============================================================================
_lock       = threading.Lock()
_running    = True
_log_buffer = []
_phase      = 'settle'

def _control_loop():
    step = 0
    while _running:
        rclpy.spin_once(controller, timeout_sec=0)
        q     = controller.get_joint_positions()
        q_dot = controller.get_joint_velocities()
        tau_vmc  = vmc_joint.hand_torques(q, q_dot)
        tau_vmc += vmc_task.hand_torques(q, q_dot)
        tau_comp = grav_fric_lim.compute_compensation_torques(
            q, q_dot, tau_vmc, recv.get_tcp_rotation_matrix())
        controller.publish_torques(tau_vmc + tau_comp)

        if not COLLECTED_DATA and step % LOG_EVERY == 0:
            with _lock:
                _log_buffer.append(list(q) + list(q_dot) + list(tau_vmc + tau_comp) + [_phase])
        step += 1
        time.sleep(1.0 / CONTROL_FREQUENCY)

def _flush(writer):
    with _lock:
        rows = _log_buffer.copy()
        _log_buffer.clear()
    for row in rows:
        writer.writerow(dict(zip(FIELDNAMES, row)))

# =============================================================================
# Protocol
# =============================================================================
arm = rtde_control.RTDEControlInterface(UR5_IP)

arm.moveL(list(UR5_POSE_GLISSANDO_START), UR5_INIT_SPEED, UR5_INIT_ACCEL)

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()

try:
    for run in range(1, N_RUNS + 1):
        controller.get_logger().info(f'Glissando run {run}/{N_RUNS} — settling ...')
        _phase = 'settle'

        for _f in PIANO_FINGERS_GLISSANDO:
            vmc_task.targets[_f] = REST_POS[_f].copy()
        time.sleep(SETTLE_TIME)

        for _f in PIANO_FINGERS_GLISSANDO:
            vmc_task.targets[_f] = PRESS_POS[_f].copy()
        time.sleep(1.0)   # let fingers reach press position before slide

        if not COLLECTED_DATA:
            f      = open(_output_path(run), 'w', newline='')
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        else:
            f = writer = None

        with _lock:
            _log_buffer.clear()

        controller.get_logger().info('  sliding forward ...')
        _phase = 'slide_forward'
        arm.moveL(list(GLISSANDO_END), GLISSANDO_SPEED, UR5_INIT_ACCEL)

        controller.get_logger().info('  returning ...')
        _phase = 'return'
        arm.moveL(list(UR5_POSE_GLISSANDO_START), GLISSANDO_SPEED, UR5_INIT_ACCEL)

        if writer:
            _flush(writer)
            f.close()

        controller.get_logger().info(f'  run {run} done.')

finally:
    _running = False
    ctrl_thread.join(timeout=1.0)
    vmc_joint.set_stiffness(0.0)
    vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0)
    vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(15))
    arm.disconnect()
    recv.disconnect()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

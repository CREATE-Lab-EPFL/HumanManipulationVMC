"""
Passive compliance shaping — piano playing with the ADAPT Hand.

Two conditions, selected by CONDITION at the top of this file:

  1. 'uniform'       — index and ring press with the same cart stiffness K [N/m],
                       swept across three values: 10, 20, 100 N/m.
                       Characterises finger-key contact dynamics as a function of
                       compliance (linear spring between fingertip and key).

  2. 'heterogeneous' — index is stiff (100 N/m), ring is compliant (10 N/m).
                       Shows how heterogeneous dynamical properties across fingers
                       affect the resulting force profiles for the same motion.

Protocol (per condition / stiffness value):
  1. UR5 moves to UR5_POSE_PIANO; hand settles at home pose.
  2. Wait SETTLE_TIME.
  3. Set stiffness; start rhythmic press/lift at PRESS_FREQUENCY.
  4. Record N_CYCLES full press-lift cycles per stiffness value.

Controller:
  - JointVMC  (K_ROT, B_ROT) — all 15 joints held at home angles (background spring).
  - TaskVMC   (K_CART, B_CART) — index and ring fingertip Cartesian springs, targets
    switch between REST position (FK at home) and PRESS position (FK at 30-deg pose).

Stiffness values [N/m] match passive_stiffness_sweep_linear.py: 10, 20, 100.
Outputs: PassiveCompliance/Hand/outputs/piano_playing_hand/<condition>/K_<K>/cycle_N.csv
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
    UR5_POSE_PIANO, UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCEL,
    PRESS_ANGLE_DEG, PIANO_FINGERS_PLAYING,
    K_SWEEP, K_STIFF, K_SOFT,
    K_ROT, B_ROT,
    B_CART_PLAYING       as B_CART,
    PLAYING_SETTLE_TIME  as SETTLE_TIME,
    PRESS_FREQUENCY, N_CYCLES, REF_RAMP_DURATION,
)

import rtde_control

# =============================================================================
# SELECT CONDITION
# =============================================================================
CONDITION = 'uniform'       # 'uniform' | 'heterogeneous'

# =============================================================================
# Collected data
# =============================================================================
COLLECTED_DATA = False

# =============================================================================
# Joint-space targets
# =============================================================================
Q_HOME  = np.zeros(15)

Q_PRESS = np.zeros(15)
for _f in PIANO_FINGERS_PLAYING:
    Q_PRESS[MOTOR_SLICES[_f]] = np.deg2rad(PRESS_ANGLE_DEG)

# =============================================================================
# Cartesian targets from FK
# =============================================================================
_R = np.zeros(3)   # fingertip attachment at DIP link origin

REST_POS  = {f: np.array(FK_motor2fingerPos(Q_HOME,  f, 'DIP', _R)) for f in PIANO_FINGERS_PLAYING}
PRESS_POS = {f: np.array(FK_motor2fingerPos(Q_PRESS, f, 'DIP', _R)) for f in PIANO_FINGERS_PLAYING}

# =============================================================================
# Logging helpers
# =============================================================================
LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 50))

FIELDNAMES = (
    [f'q_{i}'    for i in range(15)] +
    [f'qdot_{i}' for i in range(15)] +
    [f'tau_{i}'  for i in range(15)]
)

def _output_path(condition, k_label, cycle):
    folder = os.path.join(_HERE, 'outputs', 'piano_playing_hand', condition, f'K_{k_label}')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'cycle_{cycle}.csv')

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
for _f in PIANO_FINGERS_PLAYING:
    vmc_task.dampers[_f].damping = np.full(3, B_CART)
    vmc_task.targets[_f] = REST_POS[_f].copy()

# =============================================================================
# Control loop (330 Hz, background thread)
# =============================================================================
_lock       = threading.Lock()
_running    = True
_log_buffer = []

_ramp_lock    = threading.Lock()
_ramp_active  = False
_ramp_t0      = None
_ramp_start   = {}
_ramp_end     = {}

def _step_target_ramp():
    global _ramp_active
    if not _ramp_active:
        return
    with _ramp_lock:
        if not _ramp_active:
            return
        alpha = min(1.0, (time.time() - _ramp_t0) / REF_RAMP_DURATION)
        for f in PIANO_FINGERS_PLAYING:
            vmc_task.targets[f] = (1 - alpha) * _ramp_start[f] + alpha * _ramp_end[f]
        if alpha >= 1.0:
            _ramp_active = False


def _control_loop():
    step = 0
    while _running:
        rclpy.spin_once(controller, timeout_sec=0)
        _step_target_ramp()
        q     = controller.get_joint_positions()
        q_dot = controller.get_joint_velocities()
        tau_vmc  = vmc_joint.hand_torques(q, q_dot)
        tau_vmc += vmc_task.hand_torques(q, q_dot)
        tau_comp = grav_fric_lim.compute_compensation_torques(
            q, q_dot, tau_vmc, recv.get_tcp_rotation_matrix())
        controller.publish_torques(tau_vmc + tau_comp)

        if not COLLECTED_DATA and step % LOG_EVERY == 0:
            with _lock:
                _log_buffer.append(list(q) + list(q_dot) + list(tau_vmc + tau_comp))
        step += 1
        time.sleep(1.0 / CONTROL_FREQUENCY)

# =============================================================================
# Helpers
# =============================================================================
def _set_stiffness(k_vals):
    for _f, k in zip(PIANO_FINGERS_PLAYING, k_vals):
        vmc_task.springs[_f].stiffness = np.full(3, k)


def _begin_target_ramp(phase):
    global _ramp_active, _ramp_t0, _ramp_start, _ramp_end
    pos = PRESS_POS if phase == 'press' else REST_POS
    with _ramp_lock:
        _ramp_start = {f: vmc_task.targets[f].copy() for f in PIANO_FINGERS_PLAYING}
        _ramp_end   = {f: pos[f].copy()              for f in PIANO_FINGERS_PLAYING}
        _ramp_t0    = time.time()
        _ramp_active = True

def _flush(writer):
    with _lock:
        rows = _log_buffer.copy()
        _log_buffer.clear()
    for row in rows:
        writer.writerow(dict(zip(FIELDNAMES, row)))

# =============================================================================
# Protocol
# =============================================================================
def run_condition(label, stiffness_pairs):
    """stiffness_pairs: list of (k_vals_list, k_label) tuples."""
    HALF = 0.5 / PRESS_FREQUENCY
    for k_vals, k_lbl in stiffness_pairs:
        _set_stiffness(k_vals)
        _begin_target_ramp('rest')
        k_info = '  '.join(f'K_{f}={k}' for f, k in zip(PIANO_FINGERS_PLAYING, k_vals))
        controller.get_logger().info(f'[{label}] {k_info}  — settling ...')
        time.sleep(SETTLE_TIME)

        for cycle in range(1, N_CYCLES + 1):
            controller.get_logger().info(f'  cycle {cycle}/{N_CYCLES}')
            if not COLLECTED_DATA:
                f      = open(_output_path(label, k_lbl, cycle), 'w', newline='')
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
            else:
                f = writer = None

            with _lock:
                _log_buffer.clear()

            _begin_target_ramp('press')
            time.sleep(HALF)
            _begin_target_ramp('rest')
            time.sleep(HALF)

            if writer:
                _flush(writer)
                f.close()

        controller.get_logger().info(f'  [{label}] K={k_lbl} done.')


arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.moveL(list(UR5_POSE_PIANO), UR5_INIT_SPEED, UR5_INIT_ACCEL)

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()

try:
    if CONDITION == 'uniform':
        run_condition('uniform', [([k, k], f'{k:.0f}') for k in K_SWEEP])

    elif CONDITION == 'heterogeneous':
        run_condition('heterogeneous',
                      [([K_STIFF, K_SOFT], f'stiff{K_STIFF:.0f}_soft{K_SOFT:.0f}')])
    else:
        raise ValueError(f'Unknown CONDITION: {CONDITION!r}')

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

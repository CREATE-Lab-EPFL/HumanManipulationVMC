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
from piano_config import (
    UR5_POSE_PIANO, UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCEL,
)

import rtde_control

# =============================================================================
# SELECT CONDITION
# =============================================================================
CONDITION = 'uniform'       # 'uniform' | 'heterogeneous'

# =============================================================================
# Experiment parameters
# =============================================================================
K_SWEEP   = [10.0, 20.0, 100.0]    # [N/m] cart stiffness sweep (condition 1)
K_STIFF   = 100.0                   # [N/m] stiff finger  (condition 2 — index)
K_SOFT    = 10.0                    # [N/m] soft finger   (condition 2 — ring)

B_CART    = 0.5                     # [N·s/m] task-space damping (both fingers)
K_ROT     = 0.1                     # [N·m/rad] background joint stiffness
B_ROT     = 0.001                   # [N·m·s/rad] background joint damping

PRESS_FREQUENCY = 1.0               # [Hz]  one full press-lift cycle per second
N_CYCLES        = 10                # cycles recorded per stiffness value
SETTLE_TIME     = 3.0               # [s]   wait before starting cycles

COLLECT_DATA = True

# =============================================================================
# Joint-space targets
# =============================================================================
Q_HOME  = np.zeros(15)

Q_PRESS = np.zeros(15)
Q_PRESS[MOTOR_SLICES['index']] = np.deg2rad(30.0)   # index MCP, PIP
Q_PRESS[MOTOR_SLICES['ring']]  = np.deg2rad(30.0)   # ring  MCP, PIP

# =============================================================================
# Cartesian targets from FK
# =============================================================================
_R = np.zeros(3)   # fingertip attachment at DIP link origin

REST_POS = {
    'index': np.array(FK_motor2fingerPos(Q_HOME,  'index', 'DIP', _R)),
    'ring':  np.array(FK_motor2fingerPos(Q_HOME,  'ring',  'DIP', _R)),
}
PRESS_POS = {
    'index': np.array(FK_motor2fingerPos(Q_PRESS, 'index', 'DIP', _R)),
    'ring':  np.array(FK_motor2fingerPos(Q_PRESS, 'ring',  'DIP', _R)),
}

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

vmc_joint = JointVMC()
vmc_joint.set_stiffness(K_ROT)
vmc_joint.set_damping(B_ROT)

vmc_task = TaskVMC()
vmc_task.set_stiffness(0.0)
vmc_task.set_damping(0.0)
for _f in ['index', 'ring']:
    vmc_task.dampers[_f].damping = np.full(3, B_CART)

vmc_task.targets['index'] = REST_POS['index'].copy()
vmc_task.targets['ring']  = REST_POS['ring'].copy()

# =============================================================================
# Control loop (330 Hz, background thread)
# =============================================================================
_lock       = threading.Lock()
_running    = True
_log_buffer = []

def _control_loop():
    step = 0
    while _running:
        q     = controller.get_joint_positions()
        q_dot = controller.get_joint_velocities()
        tau   = grav_fric_lim.hand_torques(q, q_dot)
        tau  += vmc_joint.hand_torques(q, q_dot)
        tau  += vmc_task.hand_torques(q, q_dot)
        controller.publish_torques(tau)

        if COLLECT_DATA and step % LOG_EVERY == 0:
            with _lock:
                _log_buffer.append(list(q) + list(q_dot) + list(tau))
        step += 1
        time.sleep(1.0 / CONTROL_FREQUENCY)

# =============================================================================
# Helpers
# =============================================================================
def _set_stiffness(k_index, k_ring):
    vmc_task.springs['index'].stiffness = np.full(3, k_index)
    vmc_task.springs['ring'].stiffness  = np.full(3, k_ring)

def _set_targets(phase):
    pos = PRESS_POS if phase == 'press' else REST_POS
    vmc_task.targets['index'] = pos['index'].copy()
    vmc_task.targets['ring']  = pos['ring'].copy()

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
    """stiffness_pairs: list of (k_index, k_ring, k_label) tuples."""
    HALF = 0.5 / PRESS_FREQUENCY
    for k_idx, k_rng, k_lbl in stiffness_pairs:
        _set_stiffness(k_idx, k_rng)
        _set_targets('rest')
        controller.get_logger().info(
            f'[{label}] K_index={k_idx}  K_ring={k_rng}  — settling ...')
        time.sleep(SETTLE_TIME)

        for cycle in range(1, N_CYCLES + 1):
            controller.get_logger().info(f'  cycle {cycle}/{N_CYCLES}')
            if COLLECT_DATA:
                f      = open(_output_path(label, k_lbl, cycle), 'w', newline='')
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
            else:
                f = writer = None

            with _lock:
                _log_buffer.clear()

            _set_targets('press')
            time.sleep(HALF)
            _set_targets('rest')
            time.sleep(HALF)

            if writer:
                _flush(writer)
                f.close()

        controller.get_logger().info(f'  [{label}] K={k_lbl} done.')


arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.moveL(list(UR5_POSE_PIANO), UR5_INIT_SPEED, UR5_INIT_ACCEL)

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()
spin_thread = threading.Thread(target=lambda: rclpy.spin(controller), daemon=True)
spin_thread.start()

try:
    if CONDITION == 'uniform':
        run_condition('uniform', [(k, k, f'{k:.0f}') for k in K_SWEEP])

    elif CONDITION == 'heterogeneous':
        run_condition('heterogeneous',
                      [(K_STIFF, K_SOFT, f'stiff{K_STIFF:.0f}_soft{K_SOFT:.0f}')])
    else:
        raise ValueError(f'Unknown CONDITION: {CONDITION!r}')

finally:
    _running = False
    ctrl_thread.join(timeout=1.0)
    arm.disconnect()
    controller.destroy_node()
    rclpy.shutdown()

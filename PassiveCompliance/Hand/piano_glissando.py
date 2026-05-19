"""
Passive compliance shaping — glissando with the ADAPT Hand.

Index and middle fingers are held at the press pose while the UR5 slides along
the keyboard for GLISSANDO_DISTANCE, then returns.  Repeated N_RUNS times.

Prerequisite — run in a separate terminal:
    python3 HelperPianoMIDI/midi_publisher.py

Outputs:
  outputs/piano_glissando/run_N.csv
  outputs/piano_glissando/midi_N.csv
"""

import numpy as np
import rclpy
import sys
import os
import csv
import time
import threading

from std_msgs.msg import String

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
STATE_FIELDNAMES = (
    [f'q_{i}'    for i in range(15)] +
    [f'qdot_{i}' for i in range(15)] +
    [f'tau_{i}'  for i in range(15)] +
    ['phase']
)
MIDI_FIELDNAMES = ['time_s', 'note', 'velocity']

def _state_path(run):
    folder = os.path.join(_HERE, 'outputs', 'piano_glissando')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'run_{run}.csv')

def _midi_path(run):
    folder = os.path.join(_HERE, 'outputs', 'piano_glissando')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'midi_{run}.csv')

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
# MIDI subscriber
# =============================================================================
_midi_lock   = threading.Lock()
_midi_events = []

def _midi_note_on_cb(msg):
    fields = dict(pair.split('=') for pair in msg.data.split())
    with _midi_lock:
        _midi_events.append({
            'time_s':   time.time(),
            'note':     int(fields['note']),
            'velocity': int(fields['velocity']),
        })

controller.create_subscription(String, '/midi/note_on', _midi_note_on_cb, 10)
controller.get_logger().info(
    'Subscribed to /midi/note_on — start midi_publisher.py if not already running.')

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

def _flush_state(writer):
    with _lock:
        rows = _log_buffer.copy()
        _log_buffer.clear()
    for row in rows:
        writer.writerow(dict(zip(STATE_FIELDNAMES, row)))

def _flush_midi(writer, run_t0):
    with _midi_lock:
        events = _midi_events.copy()
        _midi_events.clear()
    for ev in events:
        writer.writerow({
            'time_s':   f'{ev["time_s"] - run_t0:.4f}',
            'note':     ev['note'],
            'velocity': ev['velocity'],
        })

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
            sf = open(_state_path(run), 'w', newline='')
            sw = csv.DictWriter(sf, fieldnames=STATE_FIELDNAMES)
            sw.writeheader()
            mf = open(_midi_path(run), 'w', newline='')
            mw = csv.DictWriter(mf, fieldnames=MIDI_FIELDNAMES)
            mw.writeheader()
        else:
            sf = sw = mf = mw = None

        with _lock:
            _log_buffer.clear()
        with _midi_lock:
            _midi_events.clear()
        run_t0 = time.time()

        controller.get_logger().info('  sliding forward ...')
        _phase = 'slide_forward'
        arm.moveL(list(GLISSANDO_END), GLISSANDO_SPEED, UR5_INIT_ACCEL)

        controller.get_logger().info('  returning ...')
        _phase = 'return'
        arm.moveL(list(UR5_POSE_GLISSANDO_START), GLISSANDO_SPEED, UR5_INIT_ACCEL)

        if not COLLECTED_DATA:
            _flush_state(sw)
            sf.close()
            _flush_midi(mw, run_t0)
            mf.close()

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

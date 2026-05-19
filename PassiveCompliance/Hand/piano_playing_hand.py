"""
Passive compliance shaping — piano playing with the ADAPT Hand.

Hand holds a fixed press pose (index+ring, stiffness K); UR5 performs N_CYCLES
rhythmic press-lift strokes per stiffness value.  MIDI note-on velocity is the
contact-force proxy.

Conditions (CONDITION):
  'uniform'       — index and ring at the same K (K_SWEEP = 10, 20, 100 N/m).
  'heterogeneous' — index at K_STIFF, ring at K_SOFT.

Prerequisite — run in a separate terminal:
    python3 HelperPianoMIDI/midi_publisher.py

Outputs:
  outputs/piano_playing_hand/<condition>/K_<K>/state_N.csv
  outputs/piano_playing_hand/<condition>/K_<K>/midi_N.csv
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
    UR5_POSE_PIANO, UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCEL,
    PRESS_ANGLE_DEG, PRESS_DEPTH, PRESS_SPEED,
    PIANO_FINGERS_PLAYING,
    K_SWEEP, K_STIFF, K_SOFT,
    K_ROT, B_ROT,
    B_CART_PLAYING       as B_CART,
    PLAYING_SETTLE_TIME  as SETTLE_TIME,
    N_CYCLES,
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
# Hand pose — fingers bent to PRESS_ANGLE_DEG; this is the fixed VMC target
# =============================================================================
Q_PRESS = np.zeros(15)
for _f in PIANO_FINGERS_PLAYING:
    Q_PRESS[MOTOR_SLICES[_f]] = np.deg2rad(PRESS_ANGLE_DEG)

_R = np.zeros(3)
PRESS_POS = {f: np.array(FK_motor2fingerPos(Q_PRESS, f, 'DIP', _R))
             for f in PIANO_FINGERS_PLAYING}

# UR5 press position: descend PRESS_DEPTH from the hover pose along base-frame Z
UR5_POSE_PRESS = UR5_POSE_PIANO.copy()
UR5_POSE_PRESS[2] -= PRESS_DEPTH

# =============================================================================
# Logging
# =============================================================================
LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 50))

STATE_FIELDNAMES = (
    ['time_s'] +
    [f'q_{i}'    for i in range(15)] +
    [f'qdot_{i}' for i in range(15)] +
    [f'tau_{i}'  for i in range(15)]
)
MIDI_FIELDNAMES = ['time_s', 'note', 'velocity']


def _state_path(condition, k_label, cycle):
    folder = os.path.join(_HERE, 'outputs', 'piano_playing_hand', condition, f'K_{k_label}')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'state_{cycle}.csv')


def _midi_path(condition, k_label, cycle):
    folder = os.path.join(_HERE, 'outputs', 'piano_playing_hand', condition, f'K_{k_label}')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'midi_{cycle}.csv')


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
vmc_task.set_stiffness(0.0)   # stiffness set per condition below
for _f in PIANO_FINGERS_PLAYING:
    vmc_task.dampers[_f].damping = np.full(3, B_CART)
    vmc_task.targets[_f] = PRESS_POS[_f].copy()   # fixed for the whole experiment

# =============================================================================
# MIDI subscriber (note_on only — velocity is the force proxy)
# =============================================================================
_midi_lock   = threading.Lock()
_midi_events = []   # dicts: {'time_s': float, 'note': int, 'velocity': int}


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
# Control loop (330 Hz, background thread)
# =============================================================================
_lock       = threading.Lock()
_running    = True
_log_buffer = []   # list of rows: [time_s, q×15, qdot×15, tau×15]
_t0_log     = time.time()


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
                _log_buffer.append(
                    [f'{time.time() - _t0_log:.4f}'] +
                    list(q) + list(q_dot) + list(tau_vmc + tau_comp)
                )
        step += 1
        time.sleep(1.0 / CONTROL_FREQUENCY)


# =============================================================================
# Helpers
# =============================================================================
def _set_stiffness(k_vals):
    for _f, k in zip(PIANO_FINGERS_PLAYING, k_vals):
        vmc_task.springs[_f].stiffness = np.full(3, k)


def _flush_state(writer):
    with _lock:
        rows = _log_buffer.copy()
        _log_buffer.clear()
    for row in rows:
        writer.writerow(dict(zip(STATE_FIELDNAMES, row)))


def _flush_midi(writer, cycle_t0):
    with _midi_lock:
        events = _midi_events.copy()
        _midi_events.clear()
    for ev in events:
        writer.writerow({
            'time_s':   f'{ev["time_s"] - cycle_t0:.4f}',
            'note':     ev['note'],
            'velocity': ev['velocity'],
        })


# =============================================================================
# Protocol
# =============================================================================
def run_condition(label, stiffness_pairs):
    for k_vals, k_lbl in stiffness_pairs:
        _set_stiffness(k_vals)
        k_info = '  '.join(f'K_{f}={k}' for f, k in zip(PIANO_FINGERS_PLAYING, k_vals))
        controller.get_logger().info(f'[{label}] {k_info} — settling {SETTLE_TIME:.0f} s …')
        time.sleep(SETTLE_TIME)

        for cycle in range(1, N_CYCLES + 1):
            controller.get_logger().info(f'  cycle {cycle}/{N_CYCLES}')

            if not COLLECTED_DATA:
                sf = open(_state_path(label, k_lbl, cycle), 'w', newline='')
                sw = csv.DictWriter(sf, fieldnames=STATE_FIELDNAMES)
                sw.writeheader()
                mf = open(_midi_path(label, k_lbl, cycle), 'w', newline='')
                mw = csv.DictWriter(mf, fieldnames=MIDI_FIELDNAMES)
                mw.writeheader()
            else:
                sf = sw = mf = mw = None

            # Clear buffers just before the stroke
            with _lock:
                _log_buffer.clear()
            with _midi_lock:
                _midi_events.clear()
            cycle_t0 = time.time()

            # UR5 stroke: descend to press keys, then ascend to hover position
            arm.moveL(UR5_POSE_PRESS.tolist(), PRESS_SPEED, UR5_INIT_ACCEL)
            arm.moveL(UR5_POSE_PIANO.tolist(), PRESS_SPEED, UR5_INIT_ACCEL)

            if not COLLECTED_DATA:
                _flush_state(sw)
                sf.close()
                _flush_midi(mw, cycle_t0)
                mf.close()

        controller.get_logger().info(f'  [{label}] K={k_lbl} done.')


# =============================================================================
# Run
# =============================================================================
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

"""
Passive compliance shaping — piano playing with the ADAPT Hand.

Hand holds a fixed press pose (index+ring, stiffness K); UR5 performs N_CYCLES
rhythmic press-lift strokes per stiffness value.  MIDI note-on velocity is the
contact-force proxy.

Conditions (CONDITION):
  'uniform'       — both fingers at the same K (K_SWEEP = 10, 20, 100 N/m).
  'heterogeneous' — index at K_STIFF, ring at K_SOFT.

Prerequisite — run in a separate terminal:
    python3 HelperPianoMIDI/midi_publisher.py

Output: outputs/piano_playing_hand/<condition>/data.csv
"""

import numpy as np
import rclpy
import sys, os, csv, time, threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../..'))
sys.path.insert(0, os.path.join(_HERE, 'HelperPianoMIDI'))

from midi_controller import MidiController

from VMCHand.HandController    import HandController, CONTROL_FREQUENCY
from VMCHand.HandVMCJointSpace import VMC as JointVMC
from VMCHand.HandVMCTaskSpace  import VMC as TaskVMC
from VMCHand.HandGravFricLim   import GravFricLim
from KinematicsHand.FK_Hand    import FK_motor2fingerPos
from ModelIDHand.motor_config  import MOTOR_SLICES
from UR5_codes.UR5_readPose    import UR5Receiver
from piano_config import (
    UR5_POSE_PIANO, UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCEL,
    PRESS_ANGLE_DEG, SPREAD_ANGLE_DEG, PRESS_DEPTH, PRESS_SPEED,
    PIANO_FINGERS_PLAYING, K_SWEEP, K_STIFF, K_SOFT,
    K_ROT, K_ROT_PRESS, B_ROT, B_CART_PLAYING as B_CART,
    PLAYING_SETTLE_TIME as SETTLE_TIME, N_CYCLES,
)
import rtde_control

# =============================================================================
# Select condition
# =============================================================================
CONDITION      = 'uniform'    # 'uniform' | 'heterogeneous'
COLLECTED_DATA = False

# =============================================================================
# Poses
# =============================================================================
Q_PRESS = np.zeros(15)
Q_PRESS[6] = np.deg2rad(SPREAD_ANGLE_DEG)
for _f in PIANO_FINGERS_PLAYING:
    Q_PRESS[MOTOR_SLICES[_f]] = np.deg2rad(PRESS_ANGLE_DEG)

PRESS_POS = {f: np.array(FK_motor2fingerPos(Q_PRESS, f, 'DIP', np.zeros(3)))
             for f in PIANO_FINGERS_PLAYING}

UR5_POSE_PRESS     = UR5_POSE_PIANO.copy()
UR5_POSE_PRESS[2] -= PRESS_DEPTH

# =============================================================================
# CSV schema
# =============================================================================
_S_COLS = ([f'q_{i}'    for i in range(15)] +
           [f'qdot_{i}' for i in range(15)] +
           [f'tau_{i}'  for i in range(15)])
FIELDS  = ['time_s', 'type', 'k_index', 'k_ring', 'cycle'] + _S_COLS + ['note', 'velocity']

def _out_path(cond):
    d = os.path.join(_HERE, 'outputs', 'piano_playing_hand', cond)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'data.csv')

# =============================================================================
# Hand initialisation
# =============================================================================
rclpy.init()
controller    = HandController()
grav_fric_lim = GravFricLim()
recv          = UR5Receiver()

vmc_joint = JointVMC()
vmc_joint.set_stiffness(K_ROT)
vmc_joint.set_damping(B_ROT)
for _k in ['index', 'middle', 'ring', 'pinky']:
    vmc_joint.spread[_k] = np.array([np.deg2rad(SPREAD_ANGLE_DEG)])
for _f in PIANO_FINGERS_PLAYING:
    vmc_joint.stiffness[_f] = np.full(3, K_ROT_PRESS)

vmc_task = TaskVMC()
vmc_task.set_stiffness(0.0)
for _f in PIANO_FINGERS_PLAYING:
    vmc_task.dampers[_f].damping = np.full(3, B_CART)
    vmc_task.targets[_f]         = PRESS_POS[_f].copy()

# =============================================================================
# MIDI input (direct rtmidi — no ROS2 bridge needed)
# =============================================================================
_midi_lock   = threading.Lock()
_midi_events = []

def _on_note_on(note: int, velocity: int) -> None:
    with _midi_lock:
        _midi_events.append({'time_s': time.time(), 'note': note, 'velocity': velocity})

midi_ctrl = MidiController(on_note_on=_on_note_on)

# =============================================================================
# Control loop
# =============================================================================
_lock     = threading.Lock()
_running  = True
_buf      = []
_t0       = time.time()
LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 50))

def _control_loop():
    step = 0
    while _running:
        rclpy.spin_once(controller, timeout_sec=0)
        q, qd = controller.get_joint_positions(), controller.get_joint_velocities()
        tau   = vmc_joint.hand_torques(q, qd) + vmc_task.hand_torques(q, qd)
        tau  += grav_fric_lim.compute_compensation_torques(q, qd, tau, recv.get_tcp_rotation_matrix())
        controller.publish_torques(tau)
        if not COLLECTED_DATA and step % LOG_EVERY == 0:
            with _lock:
                _buf.append([f'{time.time()-_t0:.4f}'] + list(q) + list(qd) + list(tau))
        step += 1
        time.sleep(1.0 / CONTROL_FREQUENCY)

# =============================================================================
# Flush: merge state rows and MIDI events into one sorted time series
# =============================================================================
def _flush(writer, k_index, k_ring, cycle):
    with _lock:      state = _buf.copy();        _buf.clear()
    with _midi_lock: midi  = _midi_events.copy(); _midi_events.clear()
    rows = []
    for r in state:
        rows.append({'time_s': r[0], 'type': 'state', 'k_index': k_index, 'k_ring': k_ring,
                     'cycle': cycle, **dict(zip(_S_COLS, r[1:])), 'note': '', 'velocity': ''})
    for e in midi:
        rows.append({'time_s': f'{e["time_s"]-_t0:.4f}', 'type': 'midi',
                     'k_index': k_index, 'k_ring': k_ring, 'cycle': cycle,
                     **{c: '' for c in _S_COLS}, 'note': e['note'], 'velocity': e['velocity']})
    rows.sort(key=lambda r: float(r['time_s']))
    for r in rows:
        writer.writerow(r)

# =============================================================================
# Protocol
# =============================================================================
def run_condition(label, stiffness_pairs):
    f      = open(_out_path(label), 'w', newline='') if not COLLECTED_DATA else None
    writer = csv.DictWriter(f, fieldnames=FIELDS) if f else None
    if writer: writer.writeheader()
    try:
        for k_vals, _ in stiffness_pairs:
            for _f, k in zip(PIANO_FINGERS_PLAYING, k_vals):
                vmc_task.springs[_f].stiffness = np.full(3, k)
            controller.get_logger().info(f'[{label}] K={k_vals} — settling {SETTLE_TIME:.0f}s ...')
            time.sleep(SETTLE_TIME)
            for cycle in range(1, N_CYCLES + 1):
                with _lock:      _buf.clear()
                with _midi_lock: _midi_events.clear()
                arm.moveL(UR5_POSE_PRESS.tolist(), PRESS_SPEED, UR5_INIT_ACCEL)
                arm.moveL(UR5_POSE_PIANO.tolist(),  PRESS_SPEED, UR5_INIT_ACCEL)
                if not COLLECTED_DATA: _flush(writer, k_vals[0], k_vals[1], cycle)
    finally:
        if f: f.close()

# =============================================================================
# Run
# =============================================================================
arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.moveL(list(UR5_POSE_PIANO), UR5_INIT_SPEED, UR5_INIT_ACCEL)

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()

try:
    if CONDITION == 'uniform':
        run_condition('uniform', [([k, k], None) for k in K_SWEEP])
    elif CONDITION == 'heterogeneous':
        run_condition('heterogeneous', [([K_STIFF, K_SOFT], None)])
    else:
        raise ValueError(f'Unknown CONDITION: {CONDITION!r}')
except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    midi_ctrl.close()
    arm.stopScript()
    _running = False
    ctrl_thread.join(timeout=1.0)
    vmc_joint.set_stiffness(0.0); vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0);  vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(15))
    arm.disconnect(); recv.disconnect()
    controller.destroy_node()
    if rclpy.ok(): rclpy.shutdown()

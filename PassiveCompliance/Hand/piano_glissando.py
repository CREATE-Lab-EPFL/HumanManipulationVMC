"""
Passive compliance shaping — glissando with the ADAPT Hand.

Index and middle fingers are held at press pose while the UR5 slides along
the keyboard for GLISSANDO_DISTANCE, then returns.  N_RUNS times per stiffness
value (K_SWEEP = 10, 20, 100 N/m).

Prerequisite — run in a separate terminal:
    python3 HelperPianoMIDI/midi_publisher.py

Output: outputs/piano_glissando/data.csv
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
    UR5_POSE_GLISSANDO_START, GLISSANDO_DIRECTION, GLISSANDO_DISTANCE, GLISSANDO_SPEED,
    UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCEL,
    PRESS_ANGLE_DEG, SPREAD_ANGLE_DEG, PIANO_FINGERS_GLISSANDO,
    K_SWEEP, K_ROT, K_ROT_PRESS, B_ROT, N_RUNS,
    B_CART_GLISSANDO as B_CART,
    GLISSANDO_SETTLE_TIME as SETTLE_TIME,
)
import rtde_control

COLLECTED_DATA = False

# =============================================================================
# Poses
# =============================================================================
Q_HOME  = np.zeros(15)
Q_PRESS = np.zeros(15)
Q_PRESS[6] = np.deg2rad(SPREAD_ANGLE_DEG)
for _f in PIANO_FINGERS_GLISSANDO:
    Q_PRESS[MOTOR_SLICES[_f]] = np.deg2rad(PRESS_ANGLE_DEG)

_R        = np.zeros(3)
REST_POS  = {f: np.array(FK_motor2fingerPos(Q_HOME,  f, 'DIP', _R)) for f in PIANO_FINGERS_GLISSANDO}
PRESS_POS = {f: np.array(FK_motor2fingerPos(Q_PRESS, f, 'DIP', _R)) for f in PIANO_FINGERS_GLISSANDO}

_dir          = GLISSANDO_DIRECTION[:3] / np.linalg.norm(GLISSANDO_DIRECTION[:3])
GLISSANDO_END = UR5_POSE_GLISSANDO_START + np.concatenate([_dir, [0, 0, 0]]) * GLISSANDO_DISTANCE

# =============================================================================
# CSV schema
# =============================================================================
_S_COLS = ([f'q_{i}'    for i in range(15)] +
           [f'qdot_{i}' for i in range(15)] +
           [f'tau_{i}'  for i in range(15)])
FIELDS  = ['time_s', 'type', 'k', 'run', 'phase'] + _S_COLS + ['note', 'velocity']

def _out_path():
    d = os.path.join(_HERE, 'outputs', 'piano_glissando')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'data_glissando.csv')

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
for _f in PIANO_FINGERS_GLISSANDO:
    vmc_joint.stiffness[_f] = np.full(3, K_ROT_PRESS)
vmc_joint.stiffness['thumb']  = np.zeros(4)
vmc_joint.stiffness['middle'] = np.zeros(3)
vmc_joint.stiffness['pinky']  = np.zeros(3)
vmc_joint.ring_pinky_target   = np.zeros(3)

vmc_task = TaskVMC()
vmc_task.set_stiffness(0.0)
vmc_task.set_damping(0.0)
for _f in PIANO_FINGERS_GLISSANDO:
    vmc_task.dampers[_f].damping = np.full(3, B_CART)
    vmc_task.targets[_f]         = PRESS_POS[_f].copy()

# =============================================================================
# MIDI input (direct rtmidi — no ROS2 bridge needed)
# =============================================================================
_midi_lock   = threading.Lock()
_midi_events = []
_phase       = 'settle'

def _on_note_on(note: int, velocity: int) -> None:
    with _midi_lock:
        _midi_events.append({'time_s':   time.time(),
                             'note':     note,
                             'velocity': velocity,
                             'phase':    _phase})

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
                _buf.append([f'{time.time()-_t0:.4f}'] + list(q) + list(qd) + list(tau) + [_phase])
        step += 1
        time.sleep(1.0 / CONTROL_FREQUENCY)

# =============================================================================
# Flush: merge state rows and MIDI events into one sorted time series
# =============================================================================
def _flush(writer, k, run):
    with _lock:      state = _buf.copy();        _buf.clear()
    with _midi_lock: midi  = _midi_events.copy(); _midi_events.clear()
    rows = []
    for r in state:
        rows.append({'time_s': r[0], 'type': 'state', 'k': k, 'run': run, 'phase': r[-1],
                     **dict(zip(_S_COLS, r[1:-1])), 'note': '', 'velocity': ''})
    for e in midi:
        rows.append({'time_s': f'{e["time_s"]-_t0:.4f}', 'type': 'midi',
                     'k': k, 'run': run, 'phase': e['phase'],
                     **{c: '' for c in _S_COLS}, 'note': e['note'], 'velocity': e['velocity']})
    rows.sort(key=lambda r: float(r['time_s']))
    for r in rows:
        writer.writerow(r)

# =============================================================================
# Protocol
# =============================================================================
input('Press ENTER to start…')
arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.moveL(list(UR5_POSE_GLISSANDO_START), UR5_INIT_SPEED, UR5_INIT_ACCEL)

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()

f      = open(_out_path(), 'w', newline='') if not COLLECTED_DATA else None
writer = csv.DictWriter(f, fieldnames=FIELDS) if f else None
if writer: writer.writeheader()

try:
    for k in K_SWEEP:
        for _f in PIANO_FINGERS_GLISSANDO:
            vmc_task.springs[_f].stiffness = np.full(3, k)
        for run in range(1, N_RUNS + 1):
            controller.get_logger().info(f'K={k:.0f} run {run}/{N_RUNS} — settling ...')
            _phase = 'settle'
            for _f in PIANO_FINGERS_GLISSANDO:
                vmc_task.targets[_f] = REST_POS[_f].copy()
            time.sleep(SETTLE_TIME)
            for _f in PIANO_FINGERS_GLISSANDO:
                vmc_task.targets[_f] = PRESS_POS[_f].copy()
            time.sleep(1.0)

            with _lock:      _buf.clear()
            with _midi_lock: _midi_events.clear()

            _phase = 'slide_forward'
            arm.moveL(list(GLISSANDO_END), GLISSANDO_SPEED, UR5_INIT_ACCEL)
            _phase = 'return'
            arm.moveL(list(UR5_POSE_GLISSANDO_START), GLISSANDO_SPEED, UR5_INIT_ACCEL)

            if not COLLECTED_DATA: _flush(writer, k, run)

except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    midi_ctrl.close()
    arm.stopScript()
    if f: f.close()
    _running = False
    ctrl_thread.join(timeout=1.0)
    vmc_joint.set_stiffness(0.0); vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0);  vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(15))
    arm.disconnect(); recv.disconnect()
    controller.destroy_node()
    if rclpy.ok(): rclpy.shutdown()

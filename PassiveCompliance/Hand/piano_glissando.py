"""
Passive compliance shaping — glissando with the ADAPT Hand.

Index and ring fingers are held at press pose while the UR5 slides along
the keyboard for GLISSANDO_DISTANCE, then returns.  N_RUNS times per stiffness
value.

Prerequisite — verify MIDI connectivity:
    python3 HelperPianoMIDI/midi_listener.py

Output: outputs/piano_glissando/data_glissando.csv
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
    GLISSANDO_DEPTH,
    UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCEL,
    PRESS_POSE, SPREAD_ANGLE_DEG, PIANO_FINGERS_GLISSANDO,
    K_SWEEP, K_ROT, K_ROT_PRESS, K_MCP_PRESS, B_ROT, B_ROT_HOLD, N_RUNS,
    WRIST_PITCH_DEG, WRIST_K_FIX, WRIST_B_FIX, FRICTION_TAU_MAX,
    B_CART_GLISSANDO as B_CART,
    GLISSANDO_SETTLE_TIME as SETTLE_TIME, RAMP_DURATION,
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
    Q_PRESS[MOTOR_SLICES[_f]] = PRESS_POSE[:2]

_R        = np.zeros(3)
REST_POS  = {f: np.array(FK_motor2fingerPos(Q_HOME,  f, 'DIP', _R)) for f in PIANO_FINGERS_GLISSANDO}
PRESS_POS = {f: np.array(FK_motor2fingerPos(Q_PRESS, f, 'DIP', _R)) for f in PIANO_FINGERS_GLISSANDO}

_dir          = GLISSANDO_DIRECTION[:3] / np.linalg.norm(GLISSANDO_DIRECTION[:3])
GLISSANDO_END = UR5_POSE_GLISSANDO_START + np.concatenate([_dir, [0, 0, 0]]) * GLISSANDO_DISTANCE

# Lowered (key-engaged) poses: descend GLISSANDO_DEPTH along base-frame Z
_DOWN                = np.array([0.0, 0.0, GLISSANDO_DEPTH, 0.0, 0.0, 0.0])
GLISSANDO_START_DOWN = UR5_POSE_GLISSANDO_START - _DOWN
GLISSANDO_END_DOWN   = GLISSANDO_END - _DOWN

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
grav_fric_lim.friction_max = FRICTION_TAU_MAX   # task-specific stiction comp (see piano_config)
recv          = UR5Receiver()

vmc_joint = JointVMC()
vmc_joint.set_stiffness(K_ROT)
vmc_joint.set_damping(B_ROT)
# Non-playing fingers: no spring (K=0), damping only — pure dissipation, no oscillation
for _hold in ['thumb', 'index', 'ring', 'pinky',
              'spread_index', 'spread_middle', 'spread_ring', 'spread_pinky']:
    vmc_joint.stiffness[_hold][:] = 0.0
    vmc_joint.damping[_hold][:]   = B_ROT_HOLD
# Wrist held (near-)RIGID via a stiff PD so the measured compliance is finger-only
vmc_joint.wrist = np.deg2rad([WRIST_PITCH_DEG, 0.0])   # [pitch, yaw]
vmc_joint.stiffness['wrist'][:] = WRIST_K_FIX
vmc_joint.damping['wrist'][:]   = WRIST_B_FIX
for _k in ['index', 'middle', 'ring', 'pinky']:
    vmc_joint.spread[_k] = np.array([np.deg2rad(SPREAD_ANGLE_DEG)])
_PRESS_JOINTS = PRESS_POSE.copy()
for _f in PIANO_FINGERS_GLISSANDO:
    # [MCP, PIP, DIP]: MCP held firmer (fixed K_MCP_PRESS); PIP/DIP stay soft
    vmc_joint.stiffness[_f] = np.array([K_MCP_PRESS, K_ROT_PRESS, K_ROT_PRESS])
# Playing finger (middle only): soft joint spring toward press pose (task spring dominates)
vmc_joint.middle_target = _PRESS_JOINTS.copy()
# Unused fingers: targets unused (K=0); damping-only, set above
vmc_joint.thumb        = np.zeros(4)
vmc_joint.index_target = np.zeros(3)
vmc_joint.ring_target  = np.zeros(3)
vmc_joint.pinky_target = np.zeros(3)

vmc_task = TaskVMC()
vmc_task.set_stiffness(0.0)
vmc_task.set_damping(0.0)
for _f in PIANO_FINGERS_GLISSANDO:
    vmc_task.dampers[_f].damping = np.full(3, B_CART)
    vmc_task.targets[_f]         = REST_POS[_f].copy()   # start at rest; ramp will bring to press

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
# Ramp helper (main-thread blocking; control loop runs throughout)
# =============================================================================

def _ramp_targets(end_pos, k_end=None):
    """Ramp task targets to end_pos; also ramp stiffness to k_end if given."""
    start_pos = {f: vmc_task.targets[f].copy() for f in PIANO_FINGERS_GLISSANDO}
    start_k   = {f: float(vmc_task.springs[f].stiffness.flat[0])
                 for f in PIANO_FINGERS_GLISSANDO} if k_end is not None else None
    dt = 1.0 / CONTROL_FREQUENCY
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        for _f in PIANO_FINGERS_GLISSANDO:
            vmc_task.targets[_f] = (1 - alpha) * start_pos[_f] + alpha * end_pos[_f]
            if k_end is not None:
                vmc_task.springs[_f].stiffness = np.full(3,
                    (1 - alpha) * start_k[_f] + alpha * k_end)
        if alpha >= 1.0:
            break
        time.sleep(dt)


# =============================================================================
# Protocol
# =============================================================================
input('Press ENTER to connect the UR5 and approach the keyboard…')
arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.moveL(list(UR5_POSE_GLISSANDO_START), UR5_INIT_SPEED, UR5_INIT_ACCEL)

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()

_ramp_targets(PRESS_POS, k_end=K_SWEEP[0])   # gradual initial approach

f      = open(_out_path(), 'w', newline='') if not COLLECTED_DATA else None
writer = csv.DictWriter(f, fieldnames=FIELDS) if f else None
if writer: writer.writeheader()

try:
    for k in K_SWEEP:
        for run in range(1, N_RUNS + 1):
            print(f'\n=== Glissando  K={k:.0f}  run {run}/{N_RUNS} ===')
            input('    Press ENTER to start this run…')
            controller.get_logger().info(f'K={k:.0f} run {run}/{N_RUNS} — settling ...')
            _phase = 'settle'
            # Ramp stiffness to the new K level during the first lift; subsequent runs are no-ops
            _ramp_targets(REST_POS, k_end=k if run == 1 else None)
            time.sleep(SETTLE_TIME)
            _ramp_targets(PRESS_POS)      # gradual press, stiffness unchanged

            with _lock:      _buf.clear()
            with _midi_lock: _midi_events.clear()

            # Descend GLISSANDO_DEPTH so the pressed finger actually engages the keys
            _phase = 'press_down'
            arm.moveL(list(GLISSANDO_START_DOWN), UR5_INIT_SPEED, UR5_INIT_ACCEL)
            _phase = 'slide_forward'
            arm.moveL(list(GLISSANDO_END_DOWN), GLISSANDO_SPEED, UR5_INIT_ACCEL)
            # Rise back up BEFORE coming home, then lift the finger off the keys, so it
            # never drags backward across them (that would damage the hand)
            _phase = 'lift_up'
            arm.moveL(list(GLISSANDO_END), UR5_INIT_SPEED, UR5_INIT_ACCEL)
            _phase = 'lift'
            _ramp_targets(REST_POS)
            _phase = 'return'
            arm.moveL(list(UR5_POSE_GLISSANDO_START), GLISSANDO_SPEED, UR5_INIT_ACCEL)
            # Next run presses the finger down again (top of the loop)

            if not COLLECTED_DATA: _flush(writer, k, run)

except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    midi_ctrl.close()
    arm.stopScript()
    _ramp_targets(REST_POS, k_end=0.0)   # gradual return home, control loop still active
    if f: f.close()
    _running = False
    ctrl_thread.join(timeout=1.0)
    vmc_joint.set_stiffness(0.0); vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0);  vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(15))
    arm.disconnect(); recv.disconnect()
    controller.destroy_node()
    if rclpy.ok(): rclpy.shutdown()

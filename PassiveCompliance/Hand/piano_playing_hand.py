"""
Passive compliance shaping — piano playing with the ADAPT Hand.

Hand holds a fixed press pose (index+ring, stiffness K); UR5 performs N_CYCLES
rhythmic press-lift strokes per stiffness value.  MIDI note-on velocity is the
contact-force proxy.

Conditions (CONDITION):
  'uniform'       — both fingers at the same K (K_SWEEP = 5, 20 N/m).
  'heterogeneous' — index at K_STIFF, ring at K_SOFT.

Prerequisite — verify MIDI connectivity:
    python3 HelperPianoMIDI/midi_listener.py

Output: outputs/piano_playing_hand/data_<condition>.csv
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
    PRESS_POSE, SPREAD_ANGLE_DEG, PINKY_BEND_DEG, PRESS_DEPTH, PRESS_SPEED, PRESS_ACCEL,
    PIANO_FINGERS_PLAYING, K_SWEEP, K_STIFF, K_SOFT,
    K_ROT, K_ROT_PRESS, K_MCP_PRESS, B_ROT, B_ROT_HOLD, WRIST_PITCH_DEG, FRICTION_TAU_MAX,
    B_CART_PLAYING as B_CART,
    PLAYING_SETTLE_TIME as SETTLE_TIME, N_CYCLES, RAMP_DURATION,
)
import rtde_control

# =============================================================================
# Data collection toggle (all three tasks run in sequence — see TASKS below)
# =============================================================================
COLLECTED_DATA = False

# =============================================================================
# Poses
# =============================================================================
Q_PRESS = np.zeros(15)
Q_PRESS[6] = np.deg2rad(SPREAD_ANGLE_DEG)
for _f in PIANO_FINGERS_PLAYING:
    Q_PRESS[MOTOR_SLICES[_f]] = PRESS_POSE[:2]

PRESS_POS = {f: np.array(FK_motor2fingerPos(Q_PRESS, f, 'DIP', np.zeros(3)))
             for f in PIANO_FINGERS_PLAYING}
REST_POS  = {f: np.array(FK_motor2fingerPos(np.zeros(15), f, 'DIP', np.zeros(3)))
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
    d = os.path.join(_HERE, 'outputs', 'piano_playing_hand')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'data_{cond}.csv')

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
for _hold in ['thumb', 'middle', 'pinky',
              'spread_index', 'spread_middle', 'spread_ring', 'spread_pinky']:
    vmc_joint.stiffness[_hold][:] = 0.0
    vmc_joint.damping[_hold][:]   = B_ROT_HOLD
# Wrist keeps its spring so it holds the chosen pitch
vmc_joint.wrist = np.deg2rad([WRIST_PITCH_DEG, 0.0])   # [pitch, yaw]
for _k in ['index', 'middle', 'ring', 'pinky']:
    vmc_joint.spread[_k] = np.array([np.deg2rad(SPREAD_ANGLE_DEG)])
_PRESS_JOINTS = PRESS_POSE.copy()
for _f in PIANO_FINGERS_PLAYING:
    # [MCP, PIP, DIP]: MCP held firmer (fixed K_MCP_PRESS); PIP/DIP stay soft
    vmc_joint.stiffness[_f] = np.array([K_MCP_PRESS, K_ROT_PRESS, K_ROT_PRESS])
# Playing fingers: soft joint spring toward press pose (task spring dominates)
vmc_joint.index_target = _PRESS_JOINTS.copy()
vmc_joint.ring_target  = _PRESS_JOINTS.copy()
# Unused fingers: targets unused (K=0); damping-only, set above
vmc_joint.thumb         = np.zeros(4)
vmc_joint.middle_target = np.zeros(3)
vmc_joint.pinky_target  = np.zeros(3)

vmc_task = TaskVMC()
vmc_task.set_stiffness(0.0)
for _f in PIANO_FINGERS_PLAYING:
    vmc_task.dampers[_f].damping = np.full(3, B_CART)
    vmc_task.targets[_f]         = REST_POS[_f].copy()

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
def run_task(k_vals, writer):
    """Ramp to k_vals = [k_index, k_ring], settle, then N_CYCLES press-lift strokes."""
    _ramp_stiffness(k_vals)   # gradual stiffness transition
    controller.get_logger().info(f'K={k_vals} — settling {SETTLE_TIME:.0f}s ...')
    time.sleep(SETTLE_TIME)
    for cycle in range(1, N_CYCLES + 1):
        with _lock:      _buf.clear()
        with _midi_lock: _midi_events.clear()
        arm.moveL(UR5_POSE_PRESS.tolist(), PRESS_SPEED, PRESS_ACCEL)
        arm.moveL(UR5_POSE_PIANO.tolist(),  PRESS_SPEED, PRESS_ACCEL)
        if writer is not None: _flush(writer, k_vals[0], k_vals[1], cycle)

# =============================================================================
# Ramp helpers (main-thread blocking; control loop runs throughout)
# =============================================================================

def _ramp_stiffness(k_vals):
    """Gradually change task spring stiffness to k_vals while holding position targets."""
    start_k = [float(vmc_task.springs[f].stiffness.flat[0]) for f in PIANO_FINGERS_PLAYING]
    if max(abs(start_k[i] - k_vals[i]) for i in range(len(k_vals))) < 0.1:
        return
    dt = 1.0 / CONTROL_FREQUENCY
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        for i, _f in enumerate(PIANO_FINGERS_PLAYING):
            vmc_task.springs[_f].stiffness = np.full(3, (1 - alpha) * start_k[i] + alpha * k_vals[i])
        if alpha >= 1.0:
            break
        time.sleep(dt)


def _ramp_to_press():
    """Gradually move task targets REST→PRESS and ramp stiffness 0→K_SWEEP[0]."""
    start_pos = {f: REST_POS[f].copy() for f in PIANO_FINGERS_PLAYING}
    dt = 1.0 / CONTROL_FREQUENCY
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        for _f in PIANO_FINGERS_PLAYING:
            vmc_task.targets[_f]           = (1 - alpha) * start_pos[_f] + alpha * PRESS_POS[_f]
            vmc_task.springs[_f].stiffness = np.full(3, alpha * K_SWEEP[0])
        if alpha >= 1.0:
            break
        time.sleep(dt)


def _ramp_to_home():
    """Gradually open the hand back to home: task targets PRESS→REST with task
    stiffness→0, joint-space finger/wrist targets PRESS→home, and every joint's
    rotational stiffness ramped up to the proper background K_ROT so the hand
    returns firmly (the playing/held fingers were soft/zero during the task)."""
    start_pos   = {f: vmc_task.targets[f].copy() for f in PIANO_FINGERS_PLAYING}
    start_k     = {f: float(vmc_task.springs[f].stiffness.flat[0]) for f in PIANO_FINGERS_PLAYING}
    start_jt    = {'index': vmc_joint.index_target.copy(), 'ring': vmc_joint.ring_target.copy()}
    start_wrist = vmc_joint.wrist.copy()
    start_ks    = {g: vmc_joint.stiffness[g].copy() for g in vmc_joint.stiffness}
    home_joint  = np.zeros(3)
    dt = 1.0 / CONTROL_FREQUENCY
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        for _f in PIANO_FINGERS_PLAYING:
            vmc_task.targets[_f]           = (1 - alpha) * start_pos[_f] + alpha * REST_POS[_f]
            vmc_task.springs[_f].stiffness = np.full(3, (1 - alpha) * start_k[_f])
        vmc_joint.index_target = (1 - alpha) * start_jt['index'] + alpha * home_joint
        vmc_joint.ring_target  = (1 - alpha) * start_jt['ring']  + alpha * home_joint
        vmc_joint.wrist        = (1 - alpha) * start_wrist       + alpha * np.zeros(2)
        for g in start_ks:                                   # restore rotational stiffness → K_ROT
            vmc_joint.stiffness[g][:] = (1 - alpha) * start_ks[g] + alpha * K_ROT
        if alpha >= 1.0:
            break
        time.sleep(dt)


# =============================================================================
# Run
# =============================================================================
# (label, [k_index, k_ring], csv_name, description)
TASKS = [
    ('uniform — low K',  [K_SWEEP[0], K_SWEEP[0]], 'uniform',
     f'Uniform compliance, LOW stiffness  — index & ring both K={K_SWEEP[0]:.0f} N/m'),
    ('uniform — high K', [K_SWEEP[1], K_SWEEP[1]], 'uniform',
     f'Uniform compliance, HIGH stiffness — index & ring both K={K_SWEEP[1]:.0f} N/m'),
    ('heterogeneous',    [K_STIFF, K_SOFT],        'heterogeneous',
     f'Heterogeneous compliance — index STIFF K={K_STIFF:.0f}, ring SOFT K={K_SOFT:.0f} N/m'),
]

input('Press ENTER to connect the UR5 and approach the keyboard…')
arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.moveL(list(UR5_POSE_PIANO), UR5_INIT_SPEED, UR5_INIT_ACCEL)

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()

_ramp_to_press()

writers = {}   # csv_name -> (file_handle, DictWriter)
try:
    for i, (label, k_vals, csv_name, desc) in enumerate(TASKS, 1):
        print(f'\n=== Task {i}/{len(TASKS)}: {label} ===\n    {desc}')
        input('    Press ENTER to start this task…')
        controller.get_logger().info(f'[{label}] {desc}')
        if not COLLECTED_DATA and csv_name not in writers:
            fh = open(_out_path(csv_name), 'w', newline='')
            w  = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
            writers[csv_name] = (fh, w)
        run_task(k_vals, writers[csv_name][1] if not COLLECTED_DATA else None)
except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    for fh, _ in writers.values(): fh.close()
    midi_ctrl.close()
    arm.stopScript()
    _ramp_to_home()    # gradual return while control loop is still active
    _running = False
    ctrl_thread.join(timeout=1.0)
    vmc_joint.set_stiffness(0.0); vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0);  vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(15))
    arm.disconnect(); recv.disconnect()
    controller.destroy_node()
    if rclpy.ok(): rclpy.shutdown()

"""
Tunable compliance — in-hand object manipulation (ADAPT Hand).

The hand closes on an object to a PC1 grasp pose with uniform tip stiffness,
then the stiffness is shifted asymmetrically between hand sides to induce
controlled object rolling/sliding:

  UNIFORM  — all five fingertips at K_UNIFORM
  ASYM_A   — pinky + ring at K_LOW;  thumb + index + middle at K_HIGH
  ASYM_B   — thumb + index + middle at K_LOW; pinky + ring at K_HIGH

Each phase is held until convergence, then recorded for RECORD_DURATION seconds.

Outputs: TunableCompliance/Hand/outputs/inhand_manipulation/inhand_run_N.csv
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

from VMCHand.HandController     import HandController, CONTROL_FREQUENCY
from VMCHand.HandVMCJointSpace  import VMC as JointVMC
from VMCHand.HandVMCTaskSpace   import VMC as TaskVMC
from VMCHand.HandGravFricLim    import GravFricLim
from KinematicsHand.FK_Hand import (
    FK_motor2wrist, FK_motor2thumb, FK_motor2finger, FK_motor2spread,
    FK_motor2thumbPos, FK_motor2fingerPos, FK_motor2palm,
    joint_to_motor,
)
from ModelIDHand.hand_params import FINGER_TIP_OFFSETS
from StiffnessModelHand.stiffness2mixedspace import tip_stiffness_MixedSpace
from UR5_codes.UR5_config import (
    UR5_IP, UR5_POSE_SQUEEZING,
    UR5_INIT_SPEED, UR5_INIT_ACCELERATION,
)
from UR5_codes.UR5_readPose import UR5Receiver
import rtde_control

# =============================================================================
# Custom parameters
# =============================================================================
K_UNIFORM    = 30.0    # [N/m]  uniform tip stiffness for baseline grasp
K_HIGH       = 150.0   # [N/m]  stiff-side stiffness
K_LOW        = 5.0     # [N/m]  compliant-side stiffness
COLLECT_DATA = False

# =============================================================================
# Fixed parameters
# =============================================================================
K_ROT       = 0.1     # [N·m/rad]
B_ROT       = 0.0001  # [N·m·s/rad]
B_TIP       = 0.001   # [N·s/m]
K_RETURN    = 0.2     # [N·m/rad]
B_FLEX_DAMP = B_ROT

ROT_DAMPING_PER_K = B_ROT / K_ROT if K_ROT else 0.0
B_RETURN          = K_RETURN * ROT_DAMPING_PER_K

SETTLE_TIME      = 3.0   # [s]
RAMP_DURATION    = 5.0   # [s]
CONVERGE_VEL_THR = 0.02  # [rad/s]
CONVERGE_HOLD    = 1.0   # [s]
CONVERGE_TIMEOUT = 12.0  # [s]
RECORD_DURATION  = 5.0   # [s]

LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 30))

# =============================================================================
# PC1 target pose
# =============================================================================
PC1_WRIST  = np.deg2rad([0.0,  0.0])
PC1_THUMB  = np.deg2rad([70.0, 0.0, 80.0, 80.0])
PC1_SPREAD = {
    'index':  np.deg2rad(-2.0),
    'middle': 0.0,
    'ring':   np.deg2rad(2.0),
    'pinky':  np.deg2rad(2.0),
}
PC1_INDEX  = np.deg2rad([80.0, 85.0, 85.0])
PC1_MIDDLE = np.deg2rad([80.0, 85.0, 85.0])
PC1_RING   = np.deg2rad([80.0, 85.0, 85.0])
PC1_PINKY  = np.deg2rad([80.0, 85.0, 85.0])

HOME_WRIST  = np.zeros(2)
HOME_THUMB  = np.zeros(4)
HOME_SPREAD = {f: np.zeros(1) for f in ['index', 'middle', 'ring', 'pinky']}
HOME_FINGER = np.zeros(3)

FINGERTIPS  = ['thumb', 'index', 'middle', 'ring', 'pinky']

# Pinky + ring are the soft side in ASYM_A; thumb + index + middle in ASYM_B.
SIDE_A_SOFT = ['pinky', 'ring']
SIDE_B_SOFT = ['thumb', 'index', 'middle']

# =============================================================================
# Derived quantities
# =============================================================================
Q_TARGET = joint_to_motor(
    PC1_WRIST,
    PC1_THUMB,
    PC1_SPREAD['index'],
    PC1_INDEX[:2],
    PC1_MIDDLE[:2],
    PC1_RING[:2],
    PC1_PINKY[:2],
)

D_REF = {
    'thumb':  np.array(FK_motor2thumbPos(Q_TARGET, 'IP',  FINGER_TIP_OFFSETS['thumb'])),
    'index':  np.array(FK_motor2fingerPos(Q_TARGET, 'index',  'DIP', FINGER_TIP_OFFSETS['index'])),
    'middle': np.array(FK_motor2fingerPos(Q_TARGET, 'middle', 'DIP', FINGER_TIP_OFFSETS['middle'])),
    'ring':   np.array(FK_motor2fingerPos(Q_TARGET, 'ring',   'DIP', FINGER_TIP_OFFSETS['ring'])),
    'pinky':  np.array(FK_motor2fingerPos(Q_TARGET, 'pinky',  'DIP', FINGER_TIP_OFFSETS['pinky'])),
    'palm':   np.array(FK_motor2palm(Q_TARGET, np.zeros(3))[1]),
}

THETA_REF_DEG = np.degrees(np.concatenate([
    PC1_WRIST,
    PC1_THUMB,
    [PC1_SPREAD['index']],
    [PC1_SPREAD['middle']],
    [PC1_SPREAD['ring']],
    [PC1_SPREAD['pinky']],
    PC1_INDEX,
    PC1_MIDDLE,
    PC1_RING,
    PC1_PINKY,
]))

K_JOINT_DICT_MODEL = {
    'wrist':         K_ROT * np.eye(2),
    'thumb':         K_ROT * np.diag([1.0, 1.0, 0.0, 0.0]),
    'spread_index':  K_ROT * np.eye(1),
    'spread_middle': K_ROT * np.eye(1),
    'spread_ring':   K_ROT * np.eye(1),
    'spread_pinky':  K_ROT * np.eye(1),
    'index':         np.zeros((3, 3)),
    'middle':        np.zeros((3, 3)),
    'ring':          np.zeros((3, 3)),
    'pinky':         np.zeros((3, 3)),
}

# Stiffness dicts for each experimental phase.
K_DICT_UNIFORM = {f: K_UNIFORM for f in FINGERTIPS}
K_DICT_ASYM_A  = {f: (K_LOW  if f in SIDE_A_SOFT else K_HIGH) for f in FINGERTIPS}
K_DICT_ASYM_B  = {f: (K_LOW  if f in SIDE_B_SOFT else K_HIGH) for f in FINGERTIPS}

# =============================================================================
# ROS2 + controller initialisation
# =============================================================================
rclpy.init()
controller = HandController()

vmc_joint = JointVMC()

vmc_joint.stiffness['wrist'] = np.full(2, K_ROT)
vmc_joint.stiffness['thumb'] = np.full(4, K_ROT)
vmc_joint.damping['wrist']   = np.full(2, B_ROT)
vmc_joint.damping['thumb']   = np.full(4, B_ROT)
for _f in ['index', 'middle', 'ring', 'pinky']:
    vmc_joint.stiffness[f'spread_{_f}'] = np.array([K_ROT])
    vmc_joint.damping[f'spread_{_f}']   = np.array([B_ROT])
    vmc_joint.stiffness[_f]             = np.full(3, K_ROT)
    vmc_joint.damping[_f]               = np.full(3, B_ROT)

vmc_joint.wrist             = HOME_WRIST.copy()
vmc_joint.thumb             = HOME_THUMB.copy()
vmc_joint.spread            = {f: HOME_SPREAD[f].copy() for f in ['index', 'middle', 'ring', 'pinky']}
vmc_joint.index_target      = HOME_FINGER.copy()
vmc_joint.middle_target     = HOME_FINGER.copy()
vmc_joint.ring_pinky_target = HOME_FINGER.copy()

vmc_task = TaskVMC()

for _f in FINGERTIPS:
    vmc_task.springs[_f].stiffness  = np.zeros(3)
    vmc_task.dampers[_f].damping    = np.full(3, B_TIP)
    vmc_task.targets[_f]            = D_REF[_f].copy()
    vmc_task.attachment_points[_f]  = FINGER_TIP_OFFSETS[_f].copy()

vmc_task.springs['palm'].stiffness = np.full(3, K_UNIFORM)
vmc_task.dampers['palm'].damping   = np.full(3, B_TIP)
vmc_task.targets['palm']           = D_REF['palm'].copy()

grav_lim = GravFricLim()
recv     = UR5Receiver()

print('Initialising stiffness model (HandHessians)…')
_t0 = time.time()
stiff_model = tip_stiffness_MixedSpace(mode='normal')
print(f'  done in {time.time() - _t0:.1f} s')
input('Press ENTER to continue…')
print()

arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info('UR5 connected')

# =============================================================================
# CSV
# =============================================================================

def _output_path():
    folder = os.path.join(_HERE, 'outputs', 'inhand_manipulation')
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder)
                if f.startswith('inhand_run_') and f.endswith('.csv')]
    n = max((int(f.replace('inhand_run_', '').replace('.csv', ''))
             for f in existing), default=0) + 1
    return os.path.join(folder, f'inhand_run_{n}.csv')


def _csv_header():
    cols = ['time_s', 'phase', 'converged']
    for _f in FINGERTIPS:
        cols.append(f'k_{_f}_Npm')
    for i in range(15):
        cols.append(f'q_motor_{i}_rad')
    for i in range(15):
        cols.append(f'q_dot_motor_{i}_rads')
    cols += ['joint_wrist_pitch_rad', 'joint_wrist_yaw_rad']
    cols += ['joint_thumb_CMC1_rad', 'joint_thumb_CMC2_rad',
             'joint_thumb_MCP_rad',  'joint_thumb_IP_rad']
    for _f in ['index', 'middle', 'ring', 'pinky']:
        cols.append(f'joint_spread_{_f}_rad')
    for _f in ['index', 'middle', 'ring', 'pinky']:
        cols += [f'joint_{_f}_MCP_rad', f'joint_{_f}_PIP_rad', f'joint_{_f}_DIP_rad']
    for _f in FINGERTIPS:
        for ax in 'xyz':
            cols.append(f'tip_{_f}_{ax}_m')
        for ax in 'xyz':
            cols.append(f'ref_{_f}_{ax}_m')
        for ax in 'xyz':
            cols.append(f'disp_{_f}_{ax}_m')
        for ax in 'xyz':
            cols.append(f'force_1st_{_f}_{ax}_N')
        cols.append(f'force_1st_{_f}_mag_N')
        for r in range(3):
            for c in range(3):
                cols.append(f'stiff_1st_{_f}_{r}{c}_Npm')
        for k in range(3):
            cols.append(f'stiff_1st_{_f}_eig_{k}_Npm')
        for ax in 'xyz':
            cols.append(f'force_2nd_{_f}_{ax}_N')
        cols.append(f'force_2nd_{_f}_mag_N')
        for r in range(3):
            for c in range(3):
                cols.append(f'stiff_2nd_{_f}_{r}{c}_Npm')
        for k in range(3):
            cols.append(f'stiff_2nd_{_f}_eig_{k}_Npm')
    return cols


_csv_path   = None
_csv_file   = None
_csv_writer = None
if COLLECT_DATA:
    _csv_path   = _output_path()
    _csv_file   = open(_csv_path, 'w', newline='')
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(_csv_header())
    controller.get_logger().info(f'Saving to: {_csv_path}')
else:
    controller.get_logger().info('Data collection disabled (COLLECT_DATA = False).')

# =============================================================================
# Computation helpers
# =============================================================================

def _tip_pos(finger, q):
    r = FINGER_TIP_OFFSETS[finger]
    if finger == 'thumb':
        return np.array(FK_motor2thumbPos(q, 'IP', r))
    return np.array(FK_motor2fingerPos(q, finger, 'DIP', r))


def _compute_row(q, q_dot, phase, k_dict, converged):
    row = [f'{time.time() - _experiment_start:.4f}', phase, int(converged)]
    for _f in FINGERTIPS:
        row.append(f'{k_dict[_f]:.1f}')

    row += [f'{v:.6f}' for v in q]
    row += [f'{v:.6f}' for v in q_dot]

    w = FK_motor2wrist(q)
    t = FK_motor2thumb(q)
    row += [f'{w[0]:.6f}', f'{w[1]:.6f}']
    row += [f'{t[i]:.6f}' for i in range(4)]
    for _f in ['index', 'middle', 'ring', 'pinky']:
        row.append(f'{FK_motor2spread(q, _f):.6f}')
    for _f in ['index', 'middle', 'ring', 'pinky']:
        ang = FK_motor2finger(q, _f)
        row += [f'{ang[i]:.6f}' for i in range(3)]

    K_task_now = {f: k_dict[f] * np.eye(3) for f in FINGERTIPS}
    K_task_now['palm'] = K_UNIFORM * np.eye(3)

    for _f in FINGERTIPS:
        pos  = _tip_pos(_f, q)
        ref  = D_REF[_f]
        disp = pos - ref

        f1   = stiff_model.tip_force(_f, q, THETA_REF_DEG, D_REF,
                                      K_JOINT_DICT_MODEL, K_task_now)
        K1   = stiff_model.tip_stiffness(_f, q, K_JOINT_DICT_MODEL, K_task_now)
        eig1 = np.linalg.eigvalsh(K1)

        # Force is linear in spring deflections; CCT modifies only stiffness.
        f2   = f1
        K2   = stiff_model.tip_stiffness(_f, q, K_JOINT_DICT_MODEL, K_task_now,
                                          d_ref_dict=D_REF, f_ext=f1)
        eig2 = np.linalg.eigvalsh(K2)

        row += [f'{v:.6f}' for v in pos]
        row += [f'{v:.6f}' for v in ref]
        row += [f'{v:.6f}' for v in disp]
        row += [f'{v:.6f}' for v in f1]
        row.append(f'{float(np.linalg.norm(f1)):.6f}')
        row += [f'{K1[r, c]:.6f}' for r in range(3) for c in range(3)]
        row += [f'{v:.6f}' for v in eig1]
        row += [f'{v:.6f}' for v in f2]
        row.append(f'{float(np.linalg.norm(f2)):.6f}')
        row += [f'{K2[r, c]:.6f}' for r in range(3) for c in range(3)]
        row += [f'{v:.6f}' for v in eig2]

    return row

# =============================================================================
# Pose dictionaries
# =============================================================================
HOME_POSE_TARGETS = {
    'wrist':             HOME_WRIST.copy(),
    'thumb':             HOME_THUMB.copy(),
    'spread':            {f: HOME_SPREAD[f].copy() for f in ['index', 'middle', 'ring', 'pinky']},
    'index_target':      HOME_FINGER.copy(),
    'middle_target':     HOME_FINGER.copy(),
    'ring_pinky_target': HOME_FINGER.copy(),
}
PC1_POSE_TARGETS = {
    'wrist':             PC1_WRIST.copy(),
    'thumb':             PC1_THUMB.copy(),
    'spread':            {f: np.array([PC1_SPREAD[f]]) for f in ['index', 'middle', 'ring', 'pinky']},
    'index_target':      PC1_INDEX.copy(),
    'middle_target':     PC1_MIDDLE.copy(),
    'ring_pinky_target': PC1_RING.copy(),
}

# =============================================================================
# State machine
# =============================================================================
STATE_INIT_ARM     = 0
STATE_SETTLE_ARM   = 1
STATE_RAMP_TO_PC1  = 2
STATE_UNIFORM_CONV = 3
STATE_UNIFORM_REC  = 4
STATE_ASYM_A_RAMP  = 5
STATE_ASYM_A_CONV  = 6
STATE_ASYM_A_REC   = 7
STATE_ASYM_B_RAMP  = 8
STATE_ASYM_B_CONV  = 9
STATE_ASYM_B_REC   = 10
STATE_UNLOAD       = 11
STATE_RAMP_TO_HOME = 12
STATE_RETURN       = 13
STATE_DONE         = 14

state             = STATE_INIT_ARM
_state_start      = time.time()
_experiment_start = time.time()
_arm_moving       = False
_converge_ticks   = 0
_CONVERGE_TICKS   = int(CONVERGE_HOLD * CONTROL_FREQUENCY)
_log_tick         = 0
_converged        = False
_use_task_vmc     = False

_ramp_t0            = None
_ramp_start_targets = None
_ramp_end_targets   = None

_k_ramp_t0    = None
_k_ramp_start = {}
_k_ramp_end   = {}
_k_ramp_after = None

_current_k_dict = {f: K_UNIFORM for f in FINGERTIPS}


def _move_arm_async(target_pose, speed, done_state):
    global state, _arm_moving, _state_start
    def _run():
        global state, _arm_moving, _state_start
        arm.moveL(target_pose.tolist(), speed, UR5_INIT_ACCELERATION)
        _state_start = time.time()
        state        = done_state
        _arm_moving  = False
    _arm_moving = True
    threading.Thread(target=_run, daemon=True).start()


def _set_task_stiffness(k_dict):
    for _f in FINGERTIPS:
        vmc_task.springs[_f].stiffness = np.full(3, k_dict[_f])


def _set_joint_stiffness_uniform(k_rot, b_rot):
    vmc_joint.stiffness['wrist'] = np.full(2, k_rot)
    vmc_joint.stiffness['thumb'] = np.full(4, k_rot)
    vmc_joint.damping['wrist']   = np.full(2, b_rot)
    vmc_joint.damping['thumb']   = np.full(4, b_rot)
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.stiffness[f'spread_{_f}'] = np.array([k_rot])
        vmc_joint.damping[f'spread_{_f}']   = np.array([b_rot])
        vmc_joint.stiffness[_f]             = np.full(3, k_rot)
        vmc_joint.damping[_f]               = np.full(3, b_rot)


def _set_joint_stiffness_experiment():
    vmc_joint.stiffness['wrist'] = np.full(2, K_ROT)
    vmc_joint.stiffness['thumb'] = np.array([K_ROT, K_ROT, 0.0, 0.0])
    vmc_joint.damping['wrist']   = np.full(2, B_ROT)
    vmc_joint.damping['thumb']   = np.full(4, B_ROT)
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.stiffness[f'spread_{_f}'] = np.array([K_ROT])
        vmc_joint.damping[f'spread_{_f}']   = np.array([B_ROT])
        vmc_joint.stiffness[_f]             = np.zeros(3)
        vmc_joint.damping[_f]               = np.full(3, B_FLEX_DAMP)


def _begin_ramp(end_targets):
    global _ramp_t0, _ramp_start_targets, _ramp_end_targets
    _ramp_t0 = time.time()
    _ramp_start_targets = {
        'wrist':             vmc_joint.wrist.copy(),
        'thumb':             vmc_joint.thumb.copy(),
        'spread':            {f: vmc_joint.spread[f].copy() for f in ['index', 'middle', 'ring', 'pinky']},
        'index_target':      vmc_joint.index_target.copy(),
        'middle_target':     vmc_joint.middle_target.copy(),
        'ring_pinky_target': vmc_joint.ring_pinky_target.copy(),
    }
    _ramp_end_targets = end_targets


def _step_ramp(now):
    alpha = min(1.0, (now - _ramp_t0) / RAMP_DURATION)
    s, e = _ramp_start_targets, _ramp_end_targets
    vmc_joint.wrist             = (1 - alpha) * s['wrist']             + alpha * e['wrist']
    vmc_joint.thumb             = (1 - alpha) * s['thumb']             + alpha * e['thumb']
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.spread[_f]    = (1 - alpha) * s['spread'][_f]        + alpha * e['spread'][_f]
    vmc_joint.index_target      = (1 - alpha) * s['index_target']      + alpha * e['index_target']
    vmc_joint.middle_target     = (1 - alpha) * s['middle_target']     + alpha * e['middle_target']
    vmc_joint.ring_pinky_target = (1 - alpha) * s['ring_pinky_target'] + alpha * e['ring_pinky_target']
    return alpha >= 1.0


def _begin_k_ramp(start_dict, end_dict, after_state):
    global _k_ramp_t0, _k_ramp_start, _k_ramp_end, _k_ramp_after
    _k_ramp_t0    = time.time()
    _k_ramp_start = {f: start_dict[f] for f in FINGERTIPS}
    _k_ramp_end   = {f: end_dict[f]   for f in FINGERTIPS}
    _k_ramp_after = after_state


def _step_k_ramp(now):
    global _current_k_dict
    alpha = min(1.0, (now - _k_ramp_t0) / RAMP_DURATION)
    for _f in FINGERTIPS:
        _current_k_dict[_f] = (1 - alpha) * _k_ramp_start[_f] + alpha * _k_ramp_end[_f]
    _set_task_stiffness(_current_k_dict)
    return alpha >= 1.0

# =============================================================================
# Control callback
# =============================================================================

def control_callback():
    global state, _state_start, _arm_moving
    global _converge_ticks, _log_tick, _converged
    global _use_task_vmc

    q     = controller.get_joint_positions()
    q_dot = controller.get_joint_velocities()

    tau_joint = vmc_joint.hand_torques(q, q_dot)
    tau_task  = vmc_task.hand_torques(q, q_dot) if _use_task_vmc else np.zeros(15)
    tau_vmc   = tau_joint + tau_task
    tau_comp  = grav_lim.compute_compensation_torques(
        q, q_dot, tau_vmc, recv.get_tcp_rotation_matrix())
    controller.publish_torques(tau_vmc + tau_comp)

    now     = time.time()
    elapsed = now - _state_start

    if state == STATE_INIT_ARM:
        if not _arm_moving:
            controller.get_logger().info('Moving UR5 to squeezing pose …')
            _move_arm_async(UR5_POSE_SQUEEZING, UR5_INIT_SPEED, STATE_SETTLE_ARM)

    elif state == STATE_SETTLE_ARM:
        if elapsed >= SETTLE_TIME:
            controller.get_logger().info(
                f'UR5 settled. Ramping HOME → PC1 over {RAMP_DURATION:.1f} s …')
            _begin_ramp(PC1_POSE_TARGETS)
            _state_start = now
            state        = STATE_RAMP_TO_PC1

    elif state == STATE_RAMP_TO_PC1:
        if _step_ramp(now):
            _set_joint_stiffness_experiment()
            _set_task_stiffness(K_DICT_UNIFORM)
            for _f in FINGERTIPS:
                vmc_task.targets[_f] = D_REF[_f].copy()
            _use_task_vmc   = True
            _converge_ticks = 0
            _converged      = False
            _state_start    = now
            state           = STATE_UNIFORM_CONV
            controller.get_logger().info(
                f'Ramp done. Engaging tip springs at K_UNIFORM = {K_UNIFORM} N/m …')

    elif state == STATE_UNIFORM_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged   = True
                _log_tick    = 0
                _state_start = now
                state        = STATE_UNIFORM_REC
                controller.get_logger().info(
                    f'Uniform grasp converged at {elapsed:.1f} s. '
                    f'Recording {RECORD_DURATION:.0f} s …')

    elif state == STATE_UNIFORM_REC:
        _log_tick += 1
        if COLLECT_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'uniform', _current_k_dict, True))
        if elapsed >= RECORD_DURATION:
            if COLLECT_DATA:
                _csv_file.flush()
            _state_start = now
            _begin_k_ramp(K_DICT_UNIFORM, K_DICT_ASYM_A, STATE_ASYM_A_CONV)
            state = STATE_ASYM_A_RAMP
            controller.get_logger().info(
                f'Ramping to ASYM_A (pinky+ring → {K_LOW} N/m, '
                f'thumb+index+middle → {K_HIGH} N/m) over {RAMP_DURATION:.1f} s …')

    elif state == STATE_ASYM_A_RAMP:
        if _step_k_ramp(now):
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = _k_ramp_after

    elif state == STATE_ASYM_A_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged   = True
                _log_tick    = 0
                _state_start = now
                state        = STATE_ASYM_A_REC
                controller.get_logger().info(
                    f'ASYM_A converged at {elapsed:.1f} s. '
                    f'Recording {RECORD_DURATION:.0f} s …')

    elif state == STATE_ASYM_A_REC:
        _log_tick += 1
        if COLLECT_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'asym_a', _current_k_dict, True))
        if elapsed >= RECORD_DURATION:
            if COLLECT_DATA:
                _csv_file.flush()
            _state_start = now
            _begin_k_ramp(K_DICT_ASYM_A, K_DICT_ASYM_B, STATE_ASYM_B_CONV)
            state = STATE_ASYM_B_RAMP
            controller.get_logger().info(
                f'Ramping to ASYM_B (thumb+index+middle → {K_LOW} N/m, '
                f'pinky+ring → {K_HIGH} N/m) over {RAMP_DURATION:.1f} s …')

    elif state == STATE_ASYM_B_RAMP:
        if _step_k_ramp(now):
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = _k_ramp_after

    elif state == STATE_ASYM_B_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged   = True
                _log_tick    = 0
                _state_start = now
                state        = STATE_ASYM_B_REC
                controller.get_logger().info(
                    f'ASYM_B converged at {elapsed:.1f} s. '
                    f'Recording {RECORD_DURATION:.0f} s …')

    elif state == STATE_ASYM_B_REC:
        _log_tick += 1
        if COLLECT_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'asym_b', _current_k_dict, True))
        if elapsed >= RECORD_DURATION:
            if COLLECT_DATA:
                _csv_file.flush()
            controller.get_logger().info(
                'All phases recorded. Releasing contact before returning home …')
            _set_task_stiffness({f: 0.0 for f in FINGERTIPS})
            _use_task_vmc = False
            _state_start  = now
            state         = STATE_UNLOAD

    elif state == STATE_UNLOAD:
        if elapsed >= CONVERGE_HOLD:
            controller.get_logger().info(
                f'Contact released. Ramping PC1 → HOME over {RAMP_DURATION:.1f} s …')
            _set_joint_stiffness_uniform(K_RETURN, B_RETURN)
            _begin_ramp(HOME_POSE_TARGETS)
            _state_start = now
            state        = STATE_RAMP_TO_HOME

    elif state == STATE_RAMP_TO_HOME:
        if _step_ramp(now):
            _converge_ticks = 0
            _converged      = False
            _state_start    = now
            state           = STATE_RETURN
            controller.get_logger().info('Ramp to HOME done. Waiting for convergence …')

    elif state == STATE_RETURN:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            state = STATE_DONE
            controller.get_logger().info('Home reached. Experiment complete.')

    elif state == STATE_DONE:
        pass

# =============================================================================
# Run
# =============================================================================
timer_period = 1.0 / CONTROL_FREQUENCY
controller.create_timer(timer_period, control_callback)
controller.get_logger().info(
    f'In-hand manipulation | '
    f'K_UNIFORM = {K_UNIFORM} N/m | '
    f'K_HIGH = {K_HIGH} N/m | '
    f'K_LOW = {K_LOW} N/m | '
    f'k_rot = {K_ROT} N·m/rad')

try:
    while rclpy.ok() and state != STATE_DONE:
        rclpy.spin_once(controller, timeout_sec=0.01)
except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    vmc_joint.set_stiffness(0.0)
    vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0)
    vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(15))
    controller.get_logger().info('Stiffness zeroed (safe shutdown).')

    if _csv_file is not None and not _csv_file.closed:
        _csv_file.close()
        controller.get_logger().info(f'Data saved to: {_csv_path}')

    recv.disconnect()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

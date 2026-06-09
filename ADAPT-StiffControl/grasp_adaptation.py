"""
ADAPT Hand — closed-loop force adaptation via stiffness-descent GD.

stiffness_descent: K_task for all five fingertips updated each tick to drive
analytic tip force toward f_des (hardcoded: hard / medium / soft force level).

Protocol:
  1. UR5 → GRASP_POSE; settle SETTLE_TIME s; hand ramps HOME → PC1 at K_TIP_GENTLE
  2. Wait velocity convergence at K_TIP_GENTLE
  3. Run GD on all five fingers until mean |f_meas − f_des| < F_CONVERGE_THR (no timeout)
  4. UR5 presses −PRESS_HEIGHT (elastic band shows exerted force)
  5. Hold HOLD_TIME s; release (k=0); hand returns HOME; arm stays at PRESS_POSE
"""

import numpy as np
import rclpy
import sys
import os
import csv
import time
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from VMCHand.HandController     import HandController, CONTROL_FREQUENCY
from VMCHand.HandVMCJointSpace  import VMC as JointVMC
from VMCHand.HandVMCTaskSpace   import VMC as TaskVMC
from VMCHand.HandGravFricLim    import GravFricLim
from KinematicsHand.FK_Hand import (
    FK_motor2thumb, FK_motor2finger, FK_motor2spread,
    FK_motor2thumbPos, FK_motor2fingerPos, FK_motor2palm,
    joint_to_motor,
)
from ModelIDHand.hand_params import FINGER_TIP_OFFSETS, eta
from StiffnessModelHand.stiffness2mixedspace import tip_stiffness_MixedSpace
from UR5_codes.UR5_config import UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCELERATION
from UR5_codes.UR5_readPose import UR5Receiver
from hand_config import (
    UR5_POSE_GRASP,
    PC1_THUMB, PC1_SPREAD, PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
    HOME_THUMB, HOME_SPREAD, HOME_FINGER,
    FINGERTIPS, FORCE_LEVELS, F_DES,
    K_TIP_GENTLE, GD_LR, GD_KTASK_STEP, F_CONVERGE_THR,
    K_ROT, B_ROT, B_TIP, K_RETURN, B_FLEX_DAMP,
    FRICTION_TAU_MAX,
    PRESS_HEIGHT,
    SETTLE_TIME, RAMP_DURATION, CONVERGE_VEL_THR, CONVERGE_HOLD,
    CONVERGE_TIMEOUT, HOLD_TIME,
)
import rtde_control

# =============================================================================
# Force level selection
# =============================================================================
print('\nSelect force level:')
for i, level in enumerate(FORCE_LEVELS):
    print(f'  {i + 1}) {level}  (f_des = {F_DES[level]:.2f} N per finger)')
_sel = int(input('Enter number: ')) - 1
assert 0 <= _sel < len(FORCE_LEVELS), 'Invalid selection'
FORCE_LEVEL = FORCE_LEVELS[_sel]
F_DES_MAG   = F_DES[FORCE_LEVEL]
print(f'Selected: {FORCE_LEVEL}  →  f_des = {F_DES_MAG:.2f} N\n')

B_RETURN  = K_RETURN * (B_ROT / K_ROT if K_ROT else 0.0)
LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 30))

_z_press   = np.array([0.0, 0.0, -PRESS_HEIGHT, 0.0, 0.0, 0.0])
GRASP_POSE = UR5_POSE_GRASP.copy()
PRESS_POSE = UR5_POSE_GRASP + _z_press

# =============================================================================
# PC1 motor targets and task-space references
# =============================================================================
Q_TARGET = joint_to_motor(
    PC1_THUMB,
    PC1_SPREAD['index'],
    PC1_INDEX[:2], PC1_MIDDLE[:2], PC1_RING[:2], PC1_PINKY[:2],
)

D_REF = {
    'thumb':  np.array(FK_motor2thumbPos(Q_TARGET, 'IP',  FINGER_TIP_OFFSETS['thumb'])),
    'index':  np.array(FK_motor2fingerPos(Q_TARGET, 'index',  'DIP', FINGER_TIP_OFFSETS['index'])),
    'middle': np.array(FK_motor2fingerPos(Q_TARGET, 'middle', 'DIP', FINGER_TIP_OFFSETS['middle'])),
    'ring':   np.array(FK_motor2fingerPos(Q_TARGET, 'ring',   'DIP', FINGER_TIP_OFFSETS['ring'])),
    'pinky':  np.array(FK_motor2fingerPos(Q_TARGET, 'pinky',  'DIP', FINGER_TIP_OFFSETS['pinky'])),
    'palm':   np.array(FK_motor2palm(np.zeros(3))[1]),
}

_d_ref_model_dict = {f: D_REF[f].copy() for f in FINGERTIPS}
_d_ref_model_dict['palm'] = D_REF['palm'].copy()

THETA_REF_DEG = np.degrees(np.concatenate([
    PC1_THUMB,
    [PC1_SPREAD['index']], [PC1_SPREAD['middle']],
    [PC1_SPREAD['ring']],  [PC1_SPREAD['pinky']],
    PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
]))

# Joint stiffness for model logging (thumb CMC active, fingers flex-passive)
K_JOINT_DICT_MODEL = {
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

# GD initial K_joint per finger
_GD_K_JOINT_INIT = {
    'thumb':  K_ROT * np.diag([1.0, 1.0, 0.0, 0.0]),
    'index':  np.zeros((3, 3)),
    'middle': np.zeros((3, 3)),
    'ring':   np.zeros((3, 3)),
    'pinky':  np.zeros((3, 3)),
}

# Per-finger theta_ref arrays whose ordering MATCHES the GD K_joint_dict keys.
# tip_force / stiffness_descent index theta_ref sequentially per dict key,
# so each array must list angles in the same order as the dict.
#   thumb:      [CMC1, CMC2, MCP, IP]                  (4 angles)
#   non-thumb:  [spread, MCP, PIP, DIP]                 (4 angles — spread first)
_GD_THETA_REF = {
    'thumb':  np.degrees(PC1_THUMB),
    'index':  np.degrees(np.concatenate([[PC1_SPREAD['index']],  PC1_INDEX])),
    'middle': np.degrees(np.concatenate([[PC1_SPREAD['middle']], PC1_MIDDLE])),
    'ring':   np.degrees(np.concatenate([[PC1_SPREAD['ring']],   PC1_RING])),
    'pinky':  np.degrees(np.concatenate([[PC1_SPREAD['pinky']],  PC1_PINKY])),
}

# GD K_joint dicts that include spread K_ROT for non-thumb fingers so the model
# accounts for the background spread-joint torque present in the actual VMC.
# The spread entry is fixed (not optimised); only the flex entry is updated by GD.
_GD_K_JOINT_DICT_INIT = {
    'thumb':  {'thumb':          K_ROT * np.diag([1.0, 1.0, 0.0, 0.0])},
    'index':  {'spread_index':   K_ROT * np.eye(1), 'index':  np.zeros((3, 3))},
    'middle': {'spread_middle':  K_ROT * np.eye(1), 'middle': np.zeros((3, 3))},
    'ring':   {'spread_ring':    K_ROT * np.eye(1), 'ring':   np.zeros((3, 3))},
    'pinky':  {'spread_pinky':   K_ROT * np.eye(1), 'pinky':  np.zeros((3, 3))},
}

# =============================================================================
# ROS2 + controller
# =============================================================================
rclpy.init()
controller = HandController()

vmc_joint = JointVMC()
vmc_joint.set_stiffness(K_ROT)
vmc_joint.set_damping(B_ROT)
vmc_joint.thumb             = HOME_THUMB.copy()
vmc_joint.spread            = {f: HOME_SPREAD[f].copy() for f in ['index', 'middle', 'ring', 'pinky']}
vmc_joint.index_target      = HOME_FINGER.copy()
vmc_joint.middle_target     = HOME_FINGER.copy()
vmc_joint.ring_pinky_target = HOME_FINGER.copy()

vmc_task = TaskVMC()
for _f in FINGERTIPS:
    vmc_task.dampers[_f].damping   = np.full(3, B_TIP)
    vmc_task.targets[_f]           = D_REF[_f].copy()
    vmc_task.attachment_points[_f] = FINGER_TIP_OFFSETS[_f].copy()

vmc_task.springs['palm'].stiffness = np.full(3, K_TIP_GENTLE)
vmc_task.dampers['palm'].damping   = np.full(3, B_TIP)
vmc_task.targets['palm']           = D_REF['palm'].copy()

grav_lim = GravFricLim()
grav_lim.friction_max = FRICTION_TAU_MAX
recv = UR5Receiver()

print('Initialising stiffness model (HandHessians)…')
_t0 = time.time()
stiff_model = tip_stiffness_MixedSpace(eta=eta, mode='normal')
print(f'  done in {time.time() - _t0:.1f} s')
input('Press ENTER to continue…')
print()

arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info('UR5 connected')

# Move to press position first so the user can verify the workspace, then lift to grasp pose
controller.get_logger().info('Moving to press position …')
arm.moveL(PRESS_POSE.tolist(), UR5_INIT_SPEED, UR5_INIT_ACCELERATION)
input('Arm at press position. Press ENTER to lift to grasp pose …')
controller.get_logger().info('Lifting to grasp pose …')
arm.moveL(GRASP_POSE.tolist(), UR5_INIT_SPEED, UR5_INIT_ACCELERATION)
controller.get_logger().info('At grasp pose. Starting experiment …')

# =============================================================================
# CSV
# =============================================================================
def _output_path():
    folder = os.path.join(_HERE, 'outputs', 'grasp_adaptation')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'grasp_{FORCE_LEVEL}.csv')


def _csv_header():
    cols = ['time_s', 'phase', 'force_level', 'f_des_mag_N', 'converged']
    # Per-finger GD: measured force and task stiffness diagonal
    for _f in FINGERTIPS:
        for ax in 'xyz':
            cols.append(f'f_meas_gd_{_f}_{ax}_N')
        cols.append(f'f_meas_gd_{_f}_mag_N')
        for ax in range(3):
            cols.append(f'K_task_{_f}_{ax}{ax}_Npm')
    # Motor angles and velocities
    for i in range(13):
        cols.append(f'q_motor_{i}_rad')
    for i in range(13):
        cols.append(f'q_dot_motor_{i}_rads')
    # FK joint angles
    cols += ['joint_thumb_CMC1_rad', 'joint_thumb_CMC2_rad',
             'joint_thumb_MCP_rad',  'joint_thumb_IP_rad']
    for _f in ['index', 'middle', 'ring', 'pinky']:
        cols.append(f'joint_spread_{_f}_rad')
    for _f in ['index', 'middle', 'ring', 'pinky']:
        cols += [f'joint_{_f}_MCP_rad', f'joint_{_f}_PIP_rad', f'joint_{_f}_DIP_rad']
    # Per-finger tip state and model forces
    for _f in FINGERTIPS:
        for ax in 'xyz':
            cols.append(f'tip_{_f}_{ax}_m')
        for ax in 'xyz':
            cols.append(f'ref_{_f}_{ax}_m')
        for ax in 'xyz':
            cols.append(f'disp_{_f}_{ax}_m')
        cols.append(f'disp_{_f}_mag_m')
        for ax in 'xyz':
            cols.append(f'force_1st_{_f}_{ax}_N')
        cols.append(f'force_1st_{_f}_mag_N')
        for r in range(3):
            for c in range(3):
                cols.append(f'stiff_1st_{_f}_{r}{c}_Npm')
        for k in range(3):
            cols.append(f'stiff_1st_{_f}_eig_{k}_Npm')
    return cols


_csv_path   = _output_path()
_csv_file   = open(_csv_path, 'w', newline='')
_csv_writer = csv.writer(_csv_file)
_csv_writer.writerow(_csv_header())
controller.get_logger().info(f'Saving to: {_csv_path}')

# =============================================================================
# Computation helpers
# =============================================================================
def _tip_pos(finger, q):
    r = FINGER_TIP_OFFSETS[finger]
    if finger == 'thumb':
        return np.array(FK_motor2thumbPos(q, 'IP', r))
    return np.array(FK_motor2fingerPos(q, finger, 'DIP', r))


def _compute_row(q, q_dot, phase, converged):
    K_task_now = {f: _gd_K_task[f] for f in FINGERTIPS}
    K_task_now['palm'] = K_TIP_GENTLE * np.eye(3)

    # Use GD-updated thumb K_joint so the logged stiffness reflects the actual controller state
    _K_joint_log = dict(K_JOINT_DICT_MODEL)
    _K_joint_log['thumb'] = _gd_K_joint['thumb']

    row = [f'{time.time() - _experiment_start:.4f}', phase,
           FORCE_LEVEL, f'{F_DES_MAG:.4f}', int(converged)]

    for _f in FINGERTIPS:
        fm = _gd_f_meas[_f]
        row += [f'{v:.6f}' for v in fm]
        row.append(f'{float(np.linalg.norm(fm)):.6f}')
        row += [f'{_gd_K_task[_f][i, i]:.6f}' for i in range(3)]

    row += [f'{v:.6f}' for v in q]
    row += [f'{v:.6f}' for v in q_dot]

    t = FK_motor2thumb(q)
    row += [f'{t[i]:.6f}' for i in range(4)]
    for _f in ['index', 'middle', 'ring', 'pinky']:
        row.append(f'{FK_motor2spread(q, _f):.6f}')
    for _f in ['index', 'middle', 'ring', 'pinky']:
        ang = FK_motor2finger(q, _f)
        row += [f'{ang[i]:.6f}' for i in range(3)]

    for _f in FINGERTIPS:
        pos  = _tip_pos(_f, q)
        ref  = _d_ref_model_dict[_f]
        disp = pos - ref
        mag  = float(np.linalg.norm(disp))
        f1   = stiff_model.tip_force(_f, q, THETA_REF_DEG, _d_ref_model_dict,
                                      _K_joint_log, K_task_now)
        K1   = stiff_model.tip_stiffness(_f, q, _K_joint_log, K_task_now)
        eig1 = np.linalg.eigvalsh(K1)

        row += [f'{v:.6f}' for v in pos]
        row += [f'{v:.6f}' for v in ref]
        row += [f'{v:.6f}' for v in disp]
        row.append(f'{mag:.6f}')
        row += [f'{v:.6f}' for v in f1]
        row.append(f'{float(np.linalg.norm(f1)):.6f}')
        row += [f'{K1[r, c]:.6f}' for r in range(3) for c in range(3)]
        row += [f'{v:.6f}' for v in eig1]

    return row


# =============================================================================
# Pose dictionaries
# =============================================================================
HOME_POSE_TARGETS = {
    'thumb':             HOME_THUMB.copy(),
    'spread':            {f: HOME_SPREAD[f].copy() for f in ['index', 'middle', 'ring', 'pinky']},
    'index_target':      HOME_FINGER.copy(),
    'middle_target':     HOME_FINGER.copy(),
    'ring_pinky_target': HOME_FINGER.copy(),
}
PC1_POSE_TARGETS = {
    'thumb':             PC1_THUMB.copy(),
    'spread':            {f: np.array([PC1_SPREAD[f]]) for f in ['index', 'middle', 'ring', 'pinky']},
    'index_target':      PC1_INDEX.copy(),
    'middle_target':     PC1_MIDDLE.copy(),
    'ring_pinky_target': PC1_RING.copy(),
}

# =============================================================================
# State machine
# =============================================================================
STATE_APPROACH   = 0   # trigger UR5 → GRASP_POSE
STATE_SETTLE     = 1   # wait SETTLE_TIME
STATE_RAMP_CLOSE = 2   # hand ramps HOME → PC1 at K_TIP_GENTLE
STATE_CONV       = 3   # wait velocity convergence at K_TIP_GENTLE
STATE_ADAPT_GD   = 4   # GD on all fingers until converged (no timeout)
STATE_PRESS      = 5   # UR5 pressing down; hold at PRESS_POSE
STATE_RELEASE    = 6   # k=0 at PRESS_POSE; wait HOLD_TIME
STATE_RAMP_HOME  = 7   # snap joint targets, ramp to HOME
STATE_RETRACT    = 8   # hand ramping to HOME; arm stays at PRESS_POSE
STATE_DONE       = 9

state             = STATE_SETTLE
_state_start      = time.time()
_experiment_start = time.time()
_arm_moving       = False
_converge_ticks   = 0
_CONVERGE_TICKS   = int(CONVERGE_HOLD * CONTROL_FREQUENCY)
_log_tick         = 0
_use_task_vmc     = False

_ramp_t0            = None
_ramp_start_targets = None
_ramp_end_targets   = None

# Gradient-descent state (initialised in STATE_CONV)
_gd_K_joint        = {f: v.copy() for f, v in _GD_K_JOINT_INIT.items()}
_gd_K_task         = {f: K_TIP_GENTLE * np.eye(3) for f in FINGERTIPS}
_gd_d_ref          = {f: D_REF[f].copy() for f in FINGERTIPS}
_gd_f_des          = {f: np.zeros(3) for f in FINGERTIPS}
_gd_f_meas         = {f: np.zeros(3) for f in FINGERTIPS}
_gd_converge_ticks = 0
_gd_last_print     = 0.0


def _move_arm_async(target_pose, speed, done_state):
    global _arm_moving
    def _run():
        global state, _arm_moving, _state_start
        arm.moveL(target_pose.tolist(), speed, UR5_INIT_ACCELERATION)
        _state_start = time.time()
        state        = done_state
        _arm_moving  = False
    _arm_moving = True
    threading.Thread(target=_run, daemon=True).start()


def _set_task_stiffness(k):
    for _f in FINGERTIPS:
        vmc_task.springs[_f].stiffness = np.full(3, k)
    vmc_task.springs['palm'].stiffness = np.full(3, k)


def _set_joint_stiffness_uniform(k_rot, b_rot):
    vmc_joint.set_stiffness(k_rot)
    vmc_joint.set_damping(b_rot)


def _set_joint_stiffness_experiment():
    vmc_joint.set_stiffness(K_ROT)
    vmc_joint.set_damping(B_ROT)
    vmc_joint.stiffness['thumb'] = np.array([K_ROT, K_ROT, 0.0, 0.0])
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.stiffness[_f] = np.zeros(3)
        vmc_joint.damping[_f]   = np.full(3, B_FLEX_DAMP)


def _begin_ramp(end_targets):
    global _ramp_t0, _ramp_start_targets, _ramp_end_targets
    _ramp_t0 = time.time()
    _ramp_start_targets = {
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
    vmc_joint.thumb             = (1 - alpha) * s['thumb']             + alpha * e['thumb']
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.spread[_f]    = (1 - alpha) * s['spread'][_f]        + alpha * e['spread'][_f]
    vmc_joint.index_target      = (1 - alpha) * s['index_target']      + alpha * e['index_target']
    vmc_joint.middle_target     = (1 - alpha) * s['middle_target']     + alpha * e['middle_target']
    vmc_joint.ring_pinky_target = (1 - alpha) * s['ring_pinky_target'] + alpha * e['ring_pinky_target']
    return alpha >= 1.0


# =============================================================================
# Control callback
# =============================================================================
def control_callback():
    global state, _state_start
    global _converge_ticks, _log_tick
    global _use_task_vmc
    global _gd_K_joint, _gd_K_task, _gd_d_ref
    global _gd_f_des, _gd_f_meas, _gd_converge_ticks, _gd_last_print

    q     = controller.get_joint_positions()
    q_dot = controller.get_joint_velocities()

    tau_joint = vmc_joint.hand_torques(q, q_dot)
    tau_task  = vmc_task.hand_torques(q, q_dot) if _use_task_vmc else np.zeros(13)
    tau_vmc   = tau_joint + tau_task
    tau_comp  = grav_lim.compute_compensation_torques(
        q, q_dot, tau_vmc, recv.get_tcp_rotation_matrix())
    controller.publish_torques(tau_vmc + tau_comp)

    now     = time.time()
    elapsed = now - _state_start

    if state == STATE_SETTLE:
        if elapsed >= SETTLE_TIME:
            controller.get_logger().info(
                f'Settled. Ramping HOME → PC1 over {RAMP_DURATION:.1f} s …')
            _begin_ramp(PC1_POSE_TARGETS)
            _state_start = now
            state        = STATE_RAMP_CLOSE

    elif state == STATE_RAMP_CLOSE:
        if _step_ramp(now):
            _set_joint_stiffness_experiment()
            _set_task_stiffness(K_TIP_GENTLE)
            for _f in FINGERTIPS:
                vmc_task.targets[_f]  = D_REF[_f].copy()
                _d_ref_model_dict[_f] = D_REF[_f].copy()
            _use_task_vmc   = True
            _converge_ticks = 0
            _state_start    = now
            state           = STATE_CONV
            controller.get_logger().info(
                f'Closing at K_TIP_GENTLE = {K_TIP_GENTLE} N/m …')

    elif state == STATE_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            # Initialise GD: f_des direction from initial contact force per finger
            for _f in FINGERTIPS:
                _kj_init = {**_GD_K_JOINT_DICT_INIT[_f]}
                _kj_init[_f] = _GD_K_JOINT_INIT[_f]
                f_init = np.asarray(stiff_model.tip_force(
                    _f, q, _GD_THETA_REF[_f], {_f: _d_ref_model_dict[_f]},
                    _kj_init, {_f: K_TIP_GENTLE * np.eye(3)}))
                f_init_mag = float(np.linalg.norm(f_init))
                if f_init_mag > 1e-9:
                    _gd_f_des[_f] = F_DES_MAG * f_init / f_init_mag
                else:
                    _gd_f_des[_f] = np.array([0.0, 0.0, -F_DES_MAG])
                _gd_K_joint[_f] = _GD_K_JOINT_INIT[_f].copy()
                _gd_K_task[_f]  = K_TIP_GENTLE * np.eye(3)
                _gd_d_ref[_f]   = _d_ref_model_dict[_f].copy()
            _gd_converge_ticks = 0
            _log_tick          = 0
            _state_start       = now
            state              = STATE_ADAPT_GD
            controller.get_logger().info(
                f'Converged at {elapsed:.1f} s. '
                f'Starting GD → f_des = {F_DES_MAG:.2f} N (level: {FORCE_LEVEL}) …')

    elif state == STATE_ADAPT_GD:
        f_errs = []
        for _f in FINGERTIPS:
            _kj_dict = {**_GD_K_JOINT_DICT_INIT[_f]}  # includes spread K_ROT for non-thumb
            _kj_dict[_f] = _gd_K_joint[_f]
            _gd_f_meas[_f] = np.asarray(stiff_model.tip_force(
                _f, q, _GD_THETA_REF[_f], {_f: _gd_d_ref[_f]},
                _kj_dict, {_f: _gd_K_task[_f]}))

            # Multiplicative update for all fingers: scale K_task (and K_joint for thumb)
            # by f_des/f_meas each tick, clamped to ±GD_KTASK_STEP fractional change.
            _f_meas_mag = float(np.linalg.norm(_gd_f_meas[_f]))
            if _f_meas_mag > 1e-6:
                _ratio = float(np.clip(F_DES_MAG / _f_meas_mag,
                                       1.0 - GD_KTASK_STEP, 1.0 + GD_KTASK_STEP))
                _gd_K_task[_f] = np.maximum(_gd_K_task[_f] * _ratio, 0.0)

            vmc_task.springs[_f].stiffness = np.diag(_gd_K_task[_f])
            f_errs.append(float(np.linalg.norm(_gd_f_meas[_f] - _gd_f_des[_f])))

        k_jt = np.diag(_gd_K_joint['thumb'])
        vmc_joint.stiffness['thumb'] = np.maximum(
            np.array([k_jt[0], k_jt[1], 0.0, 0.0]), 0.0)

        f_err_mean = float(np.mean(f_errs))
        if f_err_mean < F_CONVERGE_THR:
            _gd_converge_ticks += 1
        else:
            _gd_converge_ticks = 0
        gd_converged = _gd_converge_ticks >= _CONVERGE_TICKS

        if not gd_converged and now - _gd_last_print >= 1.0:
            _gd_last_print = now
            errs_str = '  '.join(f'{f}: {e:.3f}' for f, e in zip(FINGERTIPS, f_errs))
            controller.get_logger().info(
                f'GD t={elapsed:.1f}s  |f_err| per finger: {errs_str}  mean: {f_err_mean:.3f} N')

        _log_tick += 1
        if _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(_compute_row(q, q_dot, 'adapt_gd', gd_converged))

        if gd_converged:
            _csv_file.flush()
            controller.get_logger().info(
                f'GD converged at {elapsed:.1f} s. '
                f'mean |f_err| = {f_err_mean:.4f} N. Pressing down …')
            _log_tick    = 0
            _state_start = now
            _move_arm_async(PRESS_POSE, UR5_INIT_SPEED, STATE_RELEASE)
            state        = STATE_PRESS

    elif state == STATE_PRESS:
        _log_tick += 1
        if _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(_compute_row(q, q_dot, 'press', True))

    elif state == STATE_RELEASE:
        _log_tick += 1
        if _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(_compute_row(q, q_dot, 'press', True))
        if elapsed >= HOLD_TIME:
            controller.get_logger().info('Releasing grasp …')
            _set_task_stiffness(0.0)
            _use_task_vmc = False
            _log_tick     = 0
            _state_start  = now
            state         = STATE_RAMP_HOME

    elif state == STATE_RAMP_HOME:
        if elapsed >= CONVERGE_HOLD:
            controller.get_logger().info(
                f'Contact released. Ramping to HOME over {RAMP_DURATION:.1f} s …')
            vmc_joint.thumb             = FK_motor2thumb(q)
            for _f in ['index', 'middle', 'ring', 'pinky']:
                vmc_joint.spread[_f]    = np.array([FK_motor2spread(q, _f)])
            vmc_joint.index_target      = FK_motor2finger(q, 'index')
            vmc_joint.middle_target     = FK_motor2finger(q, 'middle')
            vmc_joint.ring_pinky_target = FK_motor2finger(q, 'ring')
            _set_joint_stiffness_uniform(K_RETURN, B_RETURN)
            _begin_ramp(HOME_POSE_TARGETS)
            _state_start = now
            state        = STATE_RETRACT

    elif state == STATE_RETRACT:
        if _step_ramp(now):
            controller.get_logger().info('Hand home.')
            state = STATE_DONE

    elif state == STATE_DONE:
        pass


# =============================================================================
# Run
# =============================================================================
timer_period = 1.0 / CONTROL_FREQUENCY
controller.create_timer(timer_period, control_callback)
controller.get_logger().info(
    f'Grasp adaptation | force level: {FORCE_LEVEL} | f_des = {F_DES_MAG} N | '
    f'K_TIP_GENTLE={K_TIP_GENTLE} N/m | GD_LR={GD_LR} | F_CONVERGE_THR={F_CONVERGE_THR} N')

try:
    while rclpy.ok() and state != STATE_DONE:
        rclpy.spin_once(controller, timeout_sec=0.01)
except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    arm.stopScript()

    vmc_joint.set_stiffness(0.0)
    vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0)
    vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(13))
    controller.get_logger().info('Stiffness zeroed (safe shutdown).')

    if not _csv_file.closed:
        _csv_file.close()
        controller.get_logger().info(f'Data saved to: {_csv_path}')

    arm.disconnect()
    recv.disconnect()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

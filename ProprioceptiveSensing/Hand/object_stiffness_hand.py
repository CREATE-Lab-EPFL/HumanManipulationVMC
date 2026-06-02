"""
Proprioceptive sensing — object stiffness estimation with the ADAPT Hand.

The hand squeezes objects of known compliance (C_0) and estimates their stiffness
by comparing the theoretical compliance C_A (from VMC) with the observed
deformation C_A + C_0.
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

from VMCHand.HandController import HandController, CONTROL_FREQUENCY
from VMCHand.HandVMCJointSpace import VMC as JointVMC
from VMCHand.HandVMCTaskSpace  import VMC as TaskVMC
from VMCHand.HandGravFricLim   import GravFricLim
from KinematicsHand.FK_Hand import (
    FK_motor2wrist, FK_motor2thumb, FK_motor2finger, FK_motor2spread,
    FK_motor2thumbPos, FK_motor2fingerPos, FK_motor2palm,
    joint_to_motor,
)
from ModelIDHand.hand_params import FINGER_TIP_OFFSETS, eta
from StiffnessModelHand.stiffness2mixedspace import tip_stiffness_MixedSpace
from UR5_codes.UR5_config import UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCELERATION
from hand_config import (
    UR5_POSE_SQUEEZING,
    PC1_WRIST, PC1_THUMB, PC1_SPREAD, PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
    HOME_WRIST, HOME_THUMB, HOME_SPREAD, HOME_FINGER,
    FINGERTIPS, OBJECTS,
    K_TIP_GENTLE, K_TIP_SWEEP,
    K_ROT, B_ROT, B_TIP, K_RETURN, B_FLEX_DAMP,
    SETTLE_TIME, RAMP_DURATION,
    CONVERGE_VEL_THR, CONVERGE_HOLD, CONVERGE_TIMEOUT, RECORD_DURATION,
)
from UR5_codes.UR5_readPose import UR5Receiver
import rtde_control

# =============================================================================
# Collected data
# =============================================================================
COLLECTED_DATA = False

B_RETURN  = K_RETURN * (B_ROT / K_ROT if K_ROT else 0.0)
LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 30))


# =============================================================================
# Derived quantities
# =============================================================================
# joint2motor_finger takes only [MCP, PIP]; DIP is a mimic of PIP.
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

# Zero-stiffness groups are kept so theta_ref indexing stays consistent with K_JOINT_DICT_MODEL.
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

# =============================================================================
# Object sequencing (all objects run back-to-back in one session)
# =============================================================================
_obj_idx    = 0
OBJECT_NAME = OBJECTS[_obj_idx]

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
    vmc_task.springs[_f].stiffness    = np.zeros(3)
    vmc_task.dampers[_f].damping      = np.full(3, B_TIP)
    vmc_task.targets[_f]              = D_REF[_f].copy()
    vmc_task.attachment_points[_f]    = FINGER_TIP_OFFSETS[_f].copy()

vmc_task.springs['palm'].stiffness = np.full(3, K_TIP_GENTLE)
vmc_task.dampers['palm'].damping   = np.full(3, B_TIP)
vmc_task.targets['palm']           = D_REF['palm'].copy()

grav_lim = GravFricLim()
recv     = UR5Receiver()

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

# =============================================================================
# CSV
# =============================================================================

def _output_path():
    folder = os.path.join(_HERE, 'outputs', 'object_stiffness_hand')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'object_stiffness_hand_{OBJECT_NAME}.csv')


def _csv_header():
    cols = ['time_s', 'phase', 'k_tip_Npm', 'converged']
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


_csv_path = None
_csv_file = None
_csv_writer = None
if not COLLECTED_DATA:
    _csv_path   = _output_path()
    _csv_file   = open(_csv_path, 'w', newline='')
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(_csv_header())
    controller.get_logger().info(f'Saving to: {_csv_path}')
else:
    controller.get_logger().info('Data collection disabled (COLLECTED_DATA = True).')

# =============================================================================
# Computation helpers
# =============================================================================

def _tip_pos(finger, q):
    r = FINGER_TIP_OFFSETS[finger]
    if finger == 'thumb':
        return np.array(FK_motor2thumbPos(q, 'IP', r))
    return np.array(FK_motor2fingerPos(q, finger, 'DIP', r))


def _compute_row(q, q_dot, phase, k_tip, converged):
    row = [f'{time.time() - _experiment_start:.4f}', phase, f'{k_tip:.1f}', int(converged)]

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

    K_task_now = {f: k_tip * np.eye(3) for f in FINGERTIPS}
    K_task_now['palm'] = k_tip * np.eye(3)

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
STATE_GENTLE_CONV  = 3
STATE_GENTLE_REC   = 4
STATE_K_TIP_RAMP   = 5
STATE_SWEEP_CONV   = 6
STATE_SWEEP_REC    = 7
STATE_UNLOAD       = 8
STATE_RAMP_TO_HOME = 9
STATE_RETURN       = 10
STATE_CONFIRM_NEXT = 11
STATE_DONE         = 12

state             = STATE_INIT_ARM
_state_start      = time.time()
_experiment_start = time.time()
_arm_moving       = False
_converge_ticks   = 0
_CONVERGE_TICKS   = int(CONVERGE_HOLD * CONTROL_FREQUENCY)
_log_tick         = 0
_converged        = False
_sweep_idx        = 0
_current_k_tip    = K_TIP_GENTLE
_use_task_vmc     = False
_confirm_ready    = False
_confirm_pending  = False

_ramp_t0            = None
_ramp_start_targets = None
_ramp_end_targets   = None

_k_ramp_start = None
_k_ramp_end   = None
_k_ramp_after = None


def _ask_confirm_async(prompt):
    global _confirm_ready, _confirm_pending
    _confirm_ready   = False
    _confirm_pending = True
    def _run():
        global _confirm_ready
        input(prompt)
        _confirm_ready = True
    threading.Thread(target=_run, daemon=True).start()


def _reset_trial():
    global _obj_idx, OBJECT_NAME, _experiment_start
    global _csv_path, _csv_file, _csv_writer
    global _converge_ticks, _converged, _sweep_idx, _current_k_tip
    global _use_task_vmc, _log_tick, _confirm_ready, _confirm_pending

    if _csv_file is not None and not _csv_file.closed:
        _csv_file.close()

    _obj_idx       += 1
    OBJECT_NAME     = OBJECTS[_obj_idx]
    _experiment_start = time.time()
    _converge_ticks = 0
    _converged      = False
    _sweep_idx      = 0
    _current_k_tip  = K_TIP_GENTLE
    _use_task_vmc   = False
    _log_tick       = 0
    _confirm_ready  = False
    _confirm_pending = False

    if not COLLECTED_DATA:
        _csv_path   = _output_path()
        _csv_file   = open(_csv_path, 'w', newline='')
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(_csv_header())
        controller.get_logger().info(f'Saving to: {_csv_path}')


def _move_arm_async(target_pose, speed, done_state):
    global _arm_moving
    def _run():
        global state, _arm_moving, _state_start
        arm.moveL(target_pose.tolist(), speed, UR5_INIT_ACCELERATION)
        _state_start = time.time()
        state = done_state
        _arm_moving = False
    _arm_moving = True
    threading.Thread(target=_run, daemon=True).start()


def _set_task_stiffness(k_tip):
    for _f in FINGERTIPS:
        vmc_task.springs[_f].stiffness = np.full(3, k_tip)
    vmc_task.springs['palm'].stiffness = np.full(3, k_tip)


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


def _begin_k_tip_ramp(k_start, k_end, after_state):
    global _ramp_t0, _k_ramp_start, _k_ramp_end, _k_ramp_after
    _ramp_t0      = time.time()
    _k_ramp_start = k_start
    _k_ramp_end   = k_end
    _k_ramp_after = after_state


def _step_k_tip_ramp(now):
    alpha = min(1.0, (now - _ramp_t0) / RAMP_DURATION)
    k_now = (1 - alpha) * _k_ramp_start + alpha * _k_ramp_end
    _set_task_stiffness(k_now)
    return alpha >= 1.0, k_now


# =============================================================================
# Control callback
# =============================================================================

def control_callback():
    global state, _state_start
    global _converge_ticks, _log_tick, _converged
    global _sweep_idx, _current_k_tip, _use_task_vmc
    global _obj_idx, _confirm_ready, _confirm_pending

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
            _set_task_stiffness(K_TIP_GENTLE)
            for _f in FINGERTIPS:
                vmc_task.targets[_f] = D_REF[_f].copy()
            _use_task_vmc   = True
            _converge_ticks = 0
            _converged      = False
            _state_start    = now
            state           = STATE_GENTLE_CONV
            controller.get_logger().info(
                f'Ramp done. Engaging tip springs at k_tip = {K_TIP_GENTLE} N/m …')

    elif state == STATE_GENTLE_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged   = True
                _state_start = now
                state        = STATE_GENTLE_REC
                controller.get_logger().info(
                    f'Gentle grasp converged at {elapsed:.1f} s. '
                    f'Recording {RECORD_DURATION:.0f} s …')

    elif state == STATE_GENTLE_REC:
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'gentle', K_TIP_GENTLE, True))
        if elapsed >= RECORD_DURATION:
            if not COLLECTED_DATA:
                _csv_file.flush()
            _sweep_idx   = 0
            _state_start = now
            _begin_k_tip_ramp(K_TIP_GENTLE, K_TIP_SWEEP[0], STATE_SWEEP_CONV)
            state = STATE_K_TIP_RAMP
            controller.get_logger().info(
                f'Baseline recorded. Ramping k_tip {K_TIP_GENTLE:.1f} → '
                f'{K_TIP_SWEEP[0]} N/m over {RAMP_DURATION:.1f} s '
                f'(sweep step 1/{len(K_TIP_SWEEP)})…')

    elif state == STATE_K_TIP_RAMP:
        done, k_now = _step_k_tip_ramp(now)
        _current_k_tip = k_now
        if done:
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = _k_ramp_after

    elif state == STATE_SWEEP_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged   = True
                _state_start = now
                state        = STATE_SWEEP_REC
                controller.get_logger().info(
                    f'k_tip = {_current_k_tip:.1f} N/m converged at {elapsed:.1f} s. '
                    f'Recording {RECORD_DURATION:.0f} s …')

    elif state == STATE_SWEEP_REC:
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'sweep', _current_k_tip, True))
        if elapsed >= RECORD_DURATION:
            if not COLLECTED_DATA:
                _csv_file.flush()
            _sweep_idx += 1
            if _sweep_idx >= len(K_TIP_SWEEP):
                controller.get_logger().info(
                    'Sweep complete. Releasing contact before returning home …')
                _set_task_stiffness(0.0)
                _use_task_vmc = False
                _state_start = now
                state        = STATE_UNLOAD
            else:
                k_prev = _current_k_tip
                k_next = K_TIP_SWEEP[_sweep_idx]
                _state_start = now
                _begin_k_tip_ramp(k_prev, k_next, STATE_SWEEP_CONV)
                state = STATE_K_TIP_RAMP
                controller.get_logger().info(
                    f'Ramping k_tip {k_prev:.1f} → {k_next} N/m over '
                    f'{RAMP_DURATION:.1f} s (sweep step '
                    f'{_sweep_idx + 1}/{len(K_TIP_SWEEP)})…')

    elif state == STATE_UNLOAD:
        if elapsed >= CONVERGE_HOLD:
            controller.get_logger().info(
                f'Contact released. Ramping PC1 → HOME over {RAMP_DURATION:.1f} s …')
            # Uniform K_RETURN so all joints follow the ramp back.
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
    f'Object stiffness hand — object: {OBJECT_NAME} | '
    f'k_rot = {K_ROT} N·m/rad | '
    f'k_tip gentle = {K_TIP_GENTLE} N/m | '
    f'sweep = {K_TIP_SWEEP} N/m | '
    f'k_return = {K_RETURN} N·m/rad')

try:
    while rclpy.ok() and state != STATE_DONE:
        rclpy.spin_once(controller, timeout_sec=0.01)
except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    arm.stopScript()

    # Zero stiffness before releasing — safe shutdown.
    vmc_joint.set_stiffness(0.0)
    vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0)
    vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(15))
    controller.get_logger().info('Stiffness zeroed (safe shutdown).')

    if _csv_file is not None and not _csv_file.closed:
        _csv_file.close()
        controller.get_logger().info(f'Data saved to: {_csv_path}')

    arm.disconnect()
    recv.disconnect()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

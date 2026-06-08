"""
Tunable compliance — in-hand object manipulation (ADAPT Hand).

The hand closes on an object to a PC1 grasp pose with uniform tip stiffness,
then the stiffness is shifted asymmetrically between hand sides to induce
controlled object rolling/sliding:

  UNIFORM  — all five fingertips at K_UNIFORM
  ASYM_A   — pinky + ring at K_LOW;  index + middle at K_HIGH  (thumb always K_UNIFORM)
  ASYM_B   — index + middle at K_LOW; pinky + ring at K_HIGH   (thumb always K_UNIFORM)

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
    FK_motor2thumb, FK_motor2finger, FK_motor2spread,
    FK_motor2thumbPos, FK_motor2fingerPos, FK_motor2palm,
    joint_to_motor,
)
from ModelIDHand.hand_params import FINGER_TIP_OFFSETS, eta
from StiffnessModelHand.stiffness2mixedspace import tip_stiffness_MixedSpace
from UR5_codes.UR5_config import UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCELERATION
from hand_config import (
    UR5_POSE_INHAND,
    INHAND_PC1_THUMB  as PC1_THUMB,
    INHAND_PC1_SPREAD as PC1_SPREAD,
    INHAND_PC1_INDEX  as PC1_INDEX,
    INHAND_PC1_MIDDLE as PC1_MIDDLE,
    INHAND_PC1_RING   as PC1_RING,
    INHAND_PC1_PINKY  as PC1_PINKY,
    HOME_THUMB, HOME_SPREAD, HOME_FINGER,
    FINGERTIPS, SIDE_A_SOFT, SIDE_B_SOFT,
    K_UNIFORM, K_HIGH, K_LOW,
    K_ROT, B_ROT, B_TIP, K_RETURN, K_BACKGROUND_IH,
    FRICTION_TAU_MAX,
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
Q_TARGET = joint_to_motor(
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
    'palm':   np.array(FK_motor2palm(np.zeros(3))[1]),
}

THETA_REF_DEG = np.degrees(np.concatenate([
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
    'thumb':         np.diag([K_ROT, K_ROT, K_BACKGROUND_IH, K_BACKGROUND_IH]),
    'spread_index':  K_ROT * np.eye(1),
    'spread_middle': K_ROT * np.eye(1),
    'spread_ring':   K_ROT * np.eye(1),
    'spread_pinky':  K_ROT * np.eye(1),
    'index':         K_BACKGROUND_IH * np.eye(3),
    'middle':        K_BACKGROUND_IH * np.eye(3),
    'ring':          K_BACKGROUND_IH * np.eye(3),
    'pinky':         K_BACKGROUND_IH * np.eye(3),
}

# Stiffness dicts for each experimental phase.
# Thumb is held at K_UNIFORM in all phases; only index/middle vs ring/pinky alternate.
K_DICT_UNIFORM = {f: K_UNIFORM for f in FINGERTIPS}
K_DICT_ASYM_A  = {f: (K_LOW  if f in SIDE_A_SOFT else K_HIGH) for f in FINGERTIPS}
K_DICT_ASYM_A['thumb'] = K_UNIFORM
K_DICT_ASYM_B  = {f: (K_LOW  if f in SIDE_B_SOFT else K_HIGH) for f in FINGERTIPS}
K_DICT_ASYM_B['thumb'] = K_UNIFORM

# =============================================================================
# ROS2 + controller initialisation
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
    vmc_task.dampers[_f].damping    = np.full(3, B_TIP)
    vmc_task.targets[_f]            = D_REF[_f].copy()
    vmc_task.attachment_points[_f]  = FINGER_TIP_OFFSETS[_f].copy()

vmc_task.springs['palm'].stiffness = np.full(3, K_UNIFORM)
vmc_task.dampers['palm'].damping   = np.full(3, B_TIP)
vmc_task.targets['palm']           = D_REF['palm'].copy()

grav_lim = GravFricLim()
grav_lim.friction_max = FRICTION_TAU_MAX
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
    folder = os.path.join(_HERE, 'outputs', 'inhand_manipulation')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, 'inhand_run.csv')


def _csv_header():
    cols = ['time_s', 'phase', 'converged']
    for _f in FINGERTIPS:
        cols.append(f'k_{_f}_Npm')
    for i in range(13):
        cols.append(f'q_motor_{i}_rad')
    for i in range(13):
        cols.append(f'q_dot_motor_{i}_rads')
    for i in range(13):
        cols.append(f'tau_joint_{i}_Nm')
    for i in range(13):
        cols.append(f'tau_task_{i}_Nm')
    for i in range(13):
        cols.append(f'tau_comp_{i}_Nm')
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


def _compute_row(q, q_dot, tau_joint, tau_task, tau_comp, phase, k_dict, converged):
    row = [f'{time.time() - _experiment_start:.4f}', phase, int(converged)]
    for _f in FINGERTIPS:
        row.append(f'{k_dict[_f]:.1f}')

    row += [f'{v:.6f}' for v in q]
    row += [f'{v:.6f}' for v in q_dot]
    row += [f'{v:.6f}' for v in tau_joint]
    row += [f'{v:.6f}' for v in tau_task]
    row += [f'{v:.6f}' for v in tau_comp]

    t = FK_motor2thumb(q)
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
STATE_INIT_ARM     = 0
STATE_SETTLE_ARM   = 1
STATE_RAMP_TO_PC1  = 2
STATE_UNIFORM_CONV = 3
STATE_UNIFORM_REC  = 4
STATE_CONFIRM_A    = 5
STATE_ASYM_A_RAMP  = 6
STATE_ASYM_A_CONV  = 7
STATE_ASYM_A_REC   = 8
STATE_CONFIRM_B    = 9
STATE_ASYM_B_RAMP  = 10
STATE_ASYM_B_CONV  = 11
STATE_ASYM_B_REC   = 12
STATE_UNLOAD       = 13
STATE_RAMP_TO_HOME = 14
STATE_RETURN       = 15
STATE_DONE         = 16

state             = STATE_INIT_ARM
_state_start      = time.time()
_experiment_start = time.time()
_arm_moving       = False
_converge_ticks   = 0
_CONVERGE_TICKS   = int(CONVERGE_HOLD * CONTROL_FREQUENCY)
_log_tick         = 0
_converged        = False
_use_task_vmc     = False
_confirm_ready    = False
_confirm_pending  = False


def _ask_confirm_async(prompt):
    global _confirm_ready, _confirm_pending
    _confirm_ready   = False
    _confirm_pending = True
    def _run():
        global _confirm_ready
        input(prompt)
        _confirm_ready = True
    threading.Thread(target=_run, daemon=True).start()

_ramp_t0            = None
_ramp_start_targets = None
_ramp_end_targets   = None

_k_ramp_t0    = None
_k_ramp_start = {}
_k_ramp_end   = {}
_k_ramp_after = None

_current_k_dict = {f: K_UNIFORM for f in FINGERTIPS}


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


def _set_task_stiffness(k_dict):
    # Palm spring is intentionally held at K_UNIFORM throughout — the
    # asymmetric study modulates only the fingertip springs.
    for _f in FINGERTIPS:
        vmc_task.springs[_f].stiffness = np.full(3, k_dict[_f])


def _set_joint_stiffness_experiment():
    vmc_joint.set_stiffness(K_ROT)
    vmc_joint.set_damping(B_ROT)
    vmc_joint.stiffness['thumb'] = np.array([K_ROT, K_ROT, K_BACKGROUND_IH, K_BACKGROUND_IH])
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.stiffness[_f] = np.full(3, K_BACKGROUND_IH)


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


def _begin_k_ramp(start_dict, end_dict, after_state):
    global _k_ramp_t0, _k_ramp_start, _k_ramp_end, _k_ramp_after
    _k_ramp_t0    = time.time()
    _k_ramp_start = {f: start_dict[f] for f in FINGERTIPS}
    _k_ramp_end   = {f: end_dict[f]   for f in FINGERTIPS}
    _k_ramp_after = after_state


def _step_k_ramp(now):
    alpha = min(1.0, (now - _k_ramp_t0) / RAMP_DURATION)
    for _f in FINGERTIPS:
        _current_k_dict[_f] = (1 - alpha) * _k_ramp_start[_f] + alpha * _k_ramp_end[_f]
    _set_task_stiffness(_current_k_dict)
    return alpha >= 1.0


def _tick_converge(q_dot, elapsed):
    global _converge_ticks
    if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
        _converge_ticks += 1
    else:
        _converge_ticks = 0
    return (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT)


def _write_record_row(q, q_dot, tau_joint, tau_task, tau_comp, phase):
    global _log_tick
    _log_tick += 1
    if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
        _csv_writer.writerow(
            _compute_row(q, q_dot, tau_joint, tau_task, tau_comp, phase, _current_k_dict, True))

# =============================================================================
# Control callback
# =============================================================================

def control_callback():
    global state, _state_start
    global _converge_ticks, _log_tick, _converged
    global _use_task_vmc
    global _confirm_ready, _confirm_pending

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

    if state == STATE_INIT_ARM:
        if not _arm_moving:
            controller.get_logger().info('Moving UR5 to squeezing pose …')
            _move_arm_async(UR5_POSE_INHAND, UR5_INIT_SPEED, STATE_SETTLE_ARM)

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
        if _tick_converge(q_dot, elapsed) and not _converged:
            _converged   = True
            _log_tick    = 0
            _state_start = now
            state        = STATE_UNIFORM_REC
            controller.get_logger().info(
                f'Uniform grasp converged at {elapsed:.1f} s. '
                f'Recording {RECORD_DURATION:.0f} s …')

    elif state == STATE_UNIFORM_REC:
        _write_record_row(q, q_dot, tau_joint, tau_task, tau_comp, 'uniform')
        if elapsed >= RECORD_DURATION:
            if not COLLECTED_DATA:
                _csv_file.flush()
            _state_start = now
            state = STATE_CONFIRM_A
            controller.get_logger().info('UNIFORM recorded. Waiting for confirmation …')

    elif state == STATE_CONFIRM_A:
        if not _confirm_pending:
            _ask_confirm_async(
                f'\n[Confirm] Press ENTER to start ASYM_A '
                f'(pinky+ring → {K_LOW} N/m, index+middle → {K_HIGH} N/m, thumb → {K_UNIFORM} N/m) …')
        elif _confirm_ready:
            _confirm_pending = False
            _begin_k_ramp(K_DICT_UNIFORM, K_DICT_ASYM_A, STATE_ASYM_A_CONV)
            state        = STATE_ASYM_A_RAMP
            _state_start = now
            controller.get_logger().info(
                f'Ramping to ASYM_A (pinky+ring → {K_LOW} N/m, '
                f'index+middle → {K_HIGH} N/m, thumb → {K_UNIFORM} N/m) over {RAMP_DURATION:.1f} s …')

    elif state == STATE_ASYM_A_RAMP:
        if _step_k_ramp(now):
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = _k_ramp_after

    elif state == STATE_ASYM_A_CONV:
        if _tick_converge(q_dot, elapsed) and not _converged:
            _converged   = True
            _log_tick    = 0
            _state_start = now
            state        = STATE_ASYM_A_REC
            controller.get_logger().info(
                f'ASYM_A converged at {elapsed:.1f} s. '
                f'Recording {RECORD_DURATION:.0f} s …')

    elif state == STATE_ASYM_A_REC:
        _write_record_row(q, q_dot, tau_joint, tau_task, tau_comp, 'asym_a')
        if elapsed >= RECORD_DURATION:
            if not COLLECTED_DATA:
                _csv_file.flush()
            _state_start = now
            state = STATE_CONFIRM_B
            controller.get_logger().info('ASYM_A recorded. Waiting for confirmation …')

    elif state == STATE_CONFIRM_B:
        if not _confirm_pending:
            _ask_confirm_async(
                f'\n[Confirm] Press ENTER to start ASYM_B '
                f'(index+middle → {K_LOW} N/m, pinky+ring → {K_HIGH} N/m, thumb → {K_UNIFORM} N/m) …')
        elif _confirm_ready:
            _confirm_pending = False
            _begin_k_ramp(K_DICT_ASYM_A, K_DICT_ASYM_B, STATE_ASYM_B_CONV)
            state        = STATE_ASYM_B_RAMP
            _state_start = now
            controller.get_logger().info(
                f'Ramping to ASYM_B (index+middle → {K_LOW} N/m, '
                f'pinky+ring → {K_HIGH} N/m, thumb → {K_UNIFORM} N/m) over {RAMP_DURATION:.1f} s …')

    elif state == STATE_ASYM_B_RAMP:
        if _step_k_ramp(now):
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = _k_ramp_after

    elif state == STATE_ASYM_B_CONV:
        if _tick_converge(q_dot, elapsed) and not _converged:
            _converged   = True
            _log_tick    = 0
            _state_start = now
            state        = STATE_ASYM_B_REC
            controller.get_logger().info(
                f'ASYM_B converged at {elapsed:.1f} s. '
                f'Recording {RECORD_DURATION:.0f} s …')

    elif state == STATE_ASYM_B_REC:
        _write_record_row(q, q_dot, tau_joint, tau_task, tau_comp, 'asym_b')
        if elapsed >= RECORD_DURATION:
            if not COLLECTED_DATA:
                _csv_file.flush()
            controller.get_logger().info(
                'All phases recorded. Releasing contact before returning home …')
            _set_task_stiffness({f: 0.0 for f in FINGERTIPS})
            _use_task_vmc = False
            vmc_joint.set_stiffness(K_RETURN)
            vmc_joint.set_damping(B_RETURN)
            _state_start  = now
            state         = STATE_UNLOAD

    elif state == STATE_UNLOAD:
        if elapsed >= CONVERGE_HOLD:
            controller.get_logger().info(
                f'Contact released. Ramping PC1 → HOME over {RAMP_DURATION:.1f} s …')
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
        if _tick_converge(q_dot, elapsed):
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
    arm.stopScript()

    vmc_joint.set_stiffness(0.0)
    vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0)
    vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(13))
    controller.get_logger().info('Stiffness zeroed (safe shutdown).')

    if _csv_file is not None and not _csv_file.closed:
        _csv_file.close()
        controller.get_logger().info(f'Data saved to: {_csv_path}')

    arm.disconnect()
    recv.disconnect()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

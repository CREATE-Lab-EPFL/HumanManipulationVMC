"""
Proprioceptive sensing — dynamic stiffness identification with the ADAPT Hand.

Identical grasp setup to object_stiffness_hand.py: 4 fingers clamped at K_TIP_HOLD,
thumb probes alone.  Instead of a static sweep, the thumb tip stiffness is ramped
from K_TIP_GENTLE → K_TIP_PROBE at different speeds (RAMP_DURATIONS), capturing the
full transient response.

One CSV per (object, ramp_duration[, run]):
  outputs/object_stiffness_dynamic/{object}_dur{N}ms[_run{R}].csv
Columns are identical to object_stiffness_hand.py plus 'ramp_duration_s'.
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
    FK_motor2thumb, FK_motor2finger, FK_motor2spread,
    FK_motor2thumbPos, FK_motor2fingerPos, FK_motor2palm,
    joint_to_motor,
)
from ModelIDHand.hand_params import FINGER_TIP_OFFSETS, eta, goal_limit_torque
from ModelIDHand.motor_config import SOFTWARE_MOTOR_ORDER
from StiffnessModelHand.stiffness2mixedspace import tip_stiffness_MixedSpace
from UR5_codes.UR5_config import UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCELERATION
from hand_config import (
    UR5_POSE_SQUEEZING,
    PC1_THUMB, PC1_SPREAD, PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
    HOME_THUMB, HOME_SPREAD, HOME_FINGER,
    FINGERTIPS, OBJECTS,
    K_TIP_GENTLE, K_TIP_PROBE, K_TIP_HOLD, N_RUNS,
    RAMP_DURATIONS, POST_RAMP_DURATION, BASELINE_DURATION,
    K_ROT, B_ROT, B_TIP, K_RETURN, B_FLEX_DAMP,
    FRICTION_TAU_MAX,
    SETTLE_TIME, RAMP_DURATION,
    CONVERGE_VEL_THR, CONVERGE_HOLD, CONVERGE_TIMEOUT,
)
from UR5_codes.UR5_readPose import UR5Receiver
import rtde_control

# =============================================================================
# Constants
# =============================================================================
COLLECTED_DATA = True

B_RETURN       = K_RETURN * (B_ROT / K_ROT if K_ROT else 0.0)
LOG_EVERY      = max(1, int(CONTROL_FREQUENCY / 30))   # baseline  ~30 Hz
LOG_EVERY_RAMP = max(1, int(CONTROL_FREQUENCY / 100))  # ramp / post-ramp ~100 Hz

_sat_warned: set = set()

# =============================================================================
# Derived quantities  (identical to object_stiffness_hand.py)
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

_d_ref_model_dict = {f: D_REF[f].copy() for f in FINGERTIPS}
_d_ref_model_dict['palm'] = D_REF['palm'].copy()

_finger_hold_pos: dict = {}
_fingers_frozen: bool  = False

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
# Object / ramp-condition sequencing
# =============================================================================
_obj_idx      = 0
_ramp_dur_idx = 0
_n_runs       = N_RUNS if not COLLECTED_DATA else 1

def _find_resume_run():
    if _n_runs <= 1:
        return 0
    folder = os.path.join(_HERE, 'outputs', 'object_stiffness_dynamic')
    if not os.path.isdir(folder):
        return 0
    for run in range(_n_runs - 1, -1, -1):
        suffix = f'_run{run + 1}'
        if all(
            os.path.isfile(os.path.join(folder,
                f'object_stiffness_dynamic_{obj}_dur{int(round(dur * 1000))}ms{suffix}.csv'))
            for obj in OBJECTS
            for dur in RAMP_DURATIONS
        ):
            return run + 1
    return 0

_run_idx          = _find_resume_run()
OBJECT_NAME       = OBJECTS[_obj_idx]
_current_ramp_dur = RAMP_DURATIONS[0]

# =============================================================================
# ROS2 + controllers
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

# =============================================================================
# CSV helpers
# =============================================================================
_experiment_start = time.time()
_csv_path   = None
_csv_file   = None
_csv_writer = None


def _dur_tag():
    return f'{int(round(_current_ramp_dur * 1000))}ms'


def _output_path():
    folder = os.path.join(_HERE, 'outputs', 'object_stiffness_dynamic')
    os.makedirs(folder, exist_ok=True)
    suffix = f'_run{_run_idx + 1}' if _n_runs > 1 else ''
    return os.path.join(folder,
        f'object_stiffness_dynamic_{OBJECT_NAME}_dur{_dur_tag()}{suffix}.csv')


def _csv_header():
    cols = ['time_s', 'phase', 'ramp_duration_s', 'k_tip_Npm']
    for i in range(13):
        cols.append(f'q_motor_{i}_rad')
    for i in range(13):
        cols.append(f'q_dot_motor_{i}_rads')
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


def _open_csv():
    global _csv_path, _csv_file, _csv_writer, _experiment_start
    if _csv_file is not None and not _csv_file.closed:
        _csv_file.close()
    _experiment_start = time.time()
    if not COLLECTED_DATA:
        _csv_path   = _output_path()
        _csv_file   = open(_csv_path, 'w', newline='')
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(_csv_header())
        controller.get_logger().info(f'Saving to: {_csv_path}')


def _close_csv():
    global _csv_file
    if _csv_file is not None and not _csv_file.closed:
        _csv_file.flush()
        _csv_file.close()

# =============================================================================
# Computation helpers  (identical to object_stiffness_hand.py)
# =============================================================================

def _tip_pos(finger, q):
    r = FINGER_TIP_OFFSETS[finger]
    if finger == 'thumb':
        return np.array(FK_motor2thumbPos(q, 'IP', r))
    return np.array(FK_motor2fingerPos(q, finger, 'DIP', r))


def _compute_row(q, q_dot, phase, k_tip):
    row = [f'{time.time() - _experiment_start:.4f}', phase,
           f'{_current_ramp_dur:.3f}', f'{k_tip:.4f}']

    row += [f'{v:.6f}' for v in q]
    row += [f'{v:.6f}' for v in q_dot]

    t = FK_motor2thumb(q)
    row += [f'{t[i]:.6f}' for i in range(4)]
    for _f in ['index', 'middle', 'ring', 'pinky']:
        row.append(f'{FK_motor2spread(q, _f):.6f}')
    for _f in ['index', 'middle', 'ring', 'pinky']:
        ang = FK_motor2finger(q, _f)
        row += [f'{ang[i]:.6f}' for i in range(3)]

    k_hold = K_TIP_HOLD if _finger_hold_pos else k_tip
    K_task_now = {
        f: (k_tip if f == 'thumb' else k_hold) * np.eye(3) for f in FINGERTIPS
    }
    K_task_now['palm'] = k_tip * np.eye(3)

    for _f in FINGERTIPS:
        pos  = _tip_pos(_f, q)
        ref  = _d_ref_model_dict[_f]
        disp = pos - ref

        f1   = stiff_model.tip_force(_f, q, THETA_REF_DEG, _d_ref_model_dict,
                                      K_JOINT_DICT_MODEL, K_task_now)
        K1   = stiff_model.tip_stiffness(_f, q, K_JOINT_DICT_MODEL, K_task_now)
        eig1 = np.linalg.eigvalsh(K1)
        f2   = f1
        K2   = stiff_model.tip_stiffness(_f, q, K_JOINT_DICT_MODEL, K_task_now,
                                          d_ref_dict=_d_ref_model_dict, f_ext=f1)
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
STATE_INIT_ARM         = 0
STATE_SETTLE_ARM       = 1
STATE_CLOSE_CONFIRM    = 2
STATE_RAMP_TO_PC1      = 3
STATE_GENTLE_CONV      = 4
STATE_GENTLE_REC       = 5   # baseline at K_TIP_GENTLE (BASELINE_DURATION), logged to CSV
STATE_K_DYN_RAMP       = 6   # ramp K_TIP_GENTLE → K_TIP_PROBE over _current_ramp_dur
STATE_POST_RAMP_REC    = 7   # record POST_RAMP_DURATION at K_TIP_PROBE
STATE_K_RESET_RAMP     = 8   # ramp K back to K_TIP_GENTLE (not logged) before next condition
STATE_UNLOAD           = 9
STATE_RAMP_TO_HOME     = 10
STATE_RETURN           = 11
STATE_CONFIRM_NEXT     = 12
STATE_CONFIRM_NEXT_RUN = 13
STATE_DONE             = 14

state             = STATE_INIT_ARM
_state_start      = time.time()
_arm_moving       = False
_converge_ticks   = 0
_CONVERGE_TICKS   = int(CONVERGE_HOLD * CONTROL_FREQUENCY)
_log_tick         = 0
_current_k_tip    = K_TIP_GENTLE
_use_task_vmc     = False
_confirm_ready    = False
_confirm_pending  = False

_ramp_t0            = None
_ramp_start_targets = None
_ramp_end_targets   = None

_k_ramp_t0      = None
_k_ramp_start_k = None
_k_ramp_end_k   = None
_k_ramp_dur_s   = None   # duration used for the current K ramp


def _ask_confirm_async(prompt):
    global _confirm_ready, _confirm_pending
    _confirm_ready   = False
    _confirm_pending = True
    def _run():
        global _confirm_ready
        input(prompt)
        _confirm_ready = True
    threading.Thread(target=_run, daemon=True).start()


def _reset_for_object():
    """Reset per-object state when advancing to a new object or run."""
    global _current_k_tip, _use_task_vmc, _log_tick
    global _converge_ticks, _confirm_ready, _confirm_pending
    global _ramp_dur_idx, _current_ramp_dur, _fingers_frozen
    _current_k_tip    = K_TIP_GENTLE
    _use_task_vmc     = False
    _log_tick         = 0
    _converge_ticks   = 0
    _confirm_ready    = False
    _confirm_pending  = False
    _ramp_dur_idx     = 0
    _current_ramp_dur = RAMP_DURATIONS[0]
    _fingers_frozen   = False
    _finger_hold_pos.clear()
    for _f in FINGERTIPS:
        _d_ref_model_dict[_f] = D_REF[_f].copy()


def _reset_trial():
    """Advance to the next object within the current run."""
    global _obj_idx, OBJECT_NAME
    _close_csv()
    _obj_idx    += 1
    OBJECT_NAME  = OBJECTS[_obj_idx]
    _reset_for_object()
    controller.get_logger().info(
        f'Next object: {OBJECT_NAME}  ({_obj_idx + 1}/{len(OBJECTS)})')


def _reset_run():
    """Start a new run: reset object index, advance run index."""
    global _run_idx, _obj_idx, OBJECT_NAME
    _close_csv()
    _run_idx   += 1
    _obj_idx    = 0
    OBJECT_NAME = OBJECTS[0]
    _reset_for_object()
    controller.get_logger().info(
        f'Run {_run_idx + 1}/{_n_runs} — first object: {OBJECT_NAME}')


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


def _set_task_stiffness(k_tip):
    for _f in FINGERTIPS:
        vmc_task.springs[_f].stiffness = np.full(3, k_tip)
    vmc_task.springs['palm'].stiffness = np.full(3, k_tip)


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


def _set_thumb_stiffness(k_tip):
    vmc_task.springs['thumb'].stiffness = np.full(3, k_tip)


def _freeze_and_hold_fingers(q):
    for _f in ['index', 'middle', 'ring', 'pinky']:
        pos = _tip_pos(_f, q)
        _finger_hold_pos[_f]           = pos.copy()
        vmc_task.targets[_f]           = pos.copy()
        vmc_task.springs[_f].stiffness = np.full(3, K_TIP_HOLD)
        _d_ref_model_dict[_f]          = pos.copy()


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


def _begin_k_ramp(k_start, k_end, dur):
    global _k_ramp_t0, _k_ramp_start_k, _k_ramp_end_k, _k_ramp_dur_s
    _k_ramp_t0      = time.time()
    _k_ramp_start_k = k_start
    _k_ramp_end_k   = k_end
    _k_ramp_dur_s   = dur


def _step_k_ramp(now):
    alpha = min(1.0, (now - _k_ramp_t0) / _k_ramp_dur_s)
    k_now = (1 - alpha) * _k_ramp_start_k + alpha * _k_ramp_end_k
    _set_thumb_stiffness(k_now)
    return alpha >= 1.0, k_now

# =============================================================================
# Control callback
# =============================================================================

def control_callback():
    global state, _state_start
    global _converge_ticks, _log_tick
    global _current_k_tip, _use_task_vmc
    global _ramp_dur_idx, _current_ramp_dur
    global _fingers_frozen, _obj_idx
    global _confirm_ready, _confirm_pending

    q     = controller.get_joint_positions()
    q_dot = controller.get_joint_velocities()

    tau_joint = vmc_joint.hand_torques(q, q_dot)
    tau_task  = vmc_task.hand_torques(q, q_dot) if _use_task_vmc else np.zeros(13)
    tau_vmc   = tau_joint + tau_task
    tau_comp  = grav_lim.compute_compensation_torques(
        q, q_dot, tau_vmc, recv.get_tcp_rotation_matrix())
    tau_total = tau_vmc + tau_comp
    controller.publish_torques(tau_total)

    for i in range(13):
        if i not in _sat_warned and abs(tau_total[i]) >= goal_limit_torque * 0.95:
            _sat_warned.add(i)
            controller.get_logger().warn(
                f'[SAT] {SOFTWARE_MOTOR_ORDER[i]} (motor {i}): '
                f'{tau_total[i]:.3f} N·m  (limit {goal_limit_torque} N·m)')

    now     = time.time()
    elapsed = now - _state_start

    # ------------------------------------------------------------------
    if state == STATE_INIT_ARM:
        if not _arm_moving:
            controller.get_logger().info('Moving UR5 to squeezing pose …')
            _move_arm_async(UR5_POSE_SQUEEZING, UR5_INIT_SPEED, STATE_SETTLE_ARM)

    elif state == STATE_SETTLE_ARM:
        if elapsed >= SETTLE_TIME:
            controller.get_logger().info('UR5 settled. Waiting for close confirmation …')
            _state_start = now
            state        = STATE_CLOSE_CONFIRM

    elif state == STATE_CLOSE_CONFIRM:
        if not _confirm_pending:
            _ask_confirm_async(
                f'\n[Confirm] Close for {OBJECT_NAME} object? Press ENTER …\n')
        elif _confirm_ready:
            _confirm_ready   = False
            _confirm_pending = False
            controller.get_logger().info(
                f'Closing — ramping HOME → PC1 over {RAMP_DURATION:.1f} s …')
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
            _log_tick    = 0
            _state_start = now
            _open_csv()
            state        = STATE_GENTLE_REC
            controller.get_logger().info(
                f'Converged. Baseline recording {BASELINE_DURATION:.1f} s '
                f'(dur {_dur_tag()})…')

    elif state == STATE_GENTLE_REC:
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(_compute_row(q, q_dot, 'baseline', K_TIP_GENTLE))
        if elapsed >= BASELINE_DURATION:
            if not COLLECTED_DATA:
                _csv_file.flush()
            if not _fingers_frozen:
                _freeze_and_hold_fingers(q)
                _fingers_frozen = True
                controller.get_logger().info(
                    f'Fingers frozen at K_TIP_HOLD = {K_TIP_HOLD:.0f} N/m.')
            _current_k_tip = K_TIP_GENTLE
            _log_tick      = 0
            _state_start   = now
            _begin_k_ramp(K_TIP_GENTLE, K_TIP_PROBE, _current_ramp_dur)
            state = STATE_K_DYN_RAMP
            controller.get_logger().info(
                f'Ramping thumb k_tip {K_TIP_GENTLE:.1f} → {K_TIP_PROBE:.1f} N/m '
                f'over {_current_ramp_dur:.2f} s …')

    elif state == STATE_K_DYN_RAMP:
        done, k_now = _step_k_ramp(now)
        _current_k_tip = k_now
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY_RAMP == 0:
            _csv_writer.writerow(_compute_row(q, q_dot, 'ramp', k_now))
        if done:
            _log_tick    = 0
            _state_start = now
            state        = STATE_POST_RAMP_REC
            controller.get_logger().info(
                f'Ramp done. Recording {POST_RAMP_DURATION:.1f} s at '
                f'k_tip = {K_TIP_PROBE:.1f} N/m …')

    elif state == STATE_POST_RAMP_REC:
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY_RAMP == 0:
            _csv_writer.writerow(_compute_row(q, q_dot, 'post_ramp', K_TIP_PROBE))
        if elapsed >= POST_RAMP_DURATION:
            _close_csv()
            _ramp_dur_idx += 1
            if _ramp_dur_idx < len(RAMP_DURATIONS):
                _current_ramp_dur = RAMP_DURATIONS[_ramp_dur_idx]
                controller.get_logger().info(
                    f'Condition done. Resetting thumb K → {K_TIP_GENTLE:.1f} N/m '
                    f'(over {RAMP_DURATION:.1f} s) before next ramp '
                    f'({_dur_tag()}, {_ramp_dur_idx + 1}/{len(RAMP_DURATIONS)})…')
                _log_tick    = 0
                _state_start = now
                _begin_k_ramp(K_TIP_PROBE, K_TIP_GENTLE, RAMP_DURATION)
                state = STATE_K_RESET_RAMP
            else:
                controller.get_logger().info(
                    f'All {len(RAMP_DURATIONS)} ramp conditions done. '
                    'Releasing contact …')
                _set_task_stiffness(0.0)
                _use_task_vmc = False
                _state_start  = now
                state         = STATE_UNLOAD

    elif state == STATE_K_RESET_RAMP:
        done, k_now = _step_k_ramp(now)
        _current_k_tip = k_now
        if done:
            _current_k_tip = K_TIP_GENTLE
            _log_tick      = 0
            _state_start   = now
            _open_csv()
            state = STATE_GENTLE_REC
            controller.get_logger().info(
                f'Reset done. Baseline recording for dur {_dur_tag()} …')

    elif state == STATE_UNLOAD:
        if elapsed >= CONVERGE_HOLD:
            controller.get_logger().info(
                f'Contact released. Ramping PC1 → HOME over {RAMP_DURATION:.1f} s …')
            vmc_joint.thumb = FK_motor2thumb(q)
            for _f in ['index', 'middle', 'ring', 'pinky']:
                vmc_joint.spread[_f]    = np.array([FK_motor2spread(q, _f)])
            vmc_joint.index_target      = FK_motor2finger(q, 'index')
            vmc_joint.middle_target     = FK_motor2finger(q, 'middle')
            vmc_joint.ring_pinky_target = FK_motor2finger(q, 'ring')
            _set_joint_stiffness_uniform(K_RETURN, B_RETURN)
            _begin_ramp(HOME_POSE_TARGETS)
            _state_start = now
            state        = STATE_RAMP_TO_HOME

    elif state == STATE_RAMP_TO_HOME:
        if _step_ramp(now):
            _converge_ticks = 0
            _state_start    = now
            state           = STATE_RETURN
            controller.get_logger().info('Ramp to HOME done. Waiting for convergence …')

    elif state == STATE_RETURN:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if _obj_idx + 1 < len(OBJECTS):
                state        = STATE_CONFIRM_NEXT
                _state_start = now
                controller.get_logger().info(
                    f'Object {_obj_idx + 1}/{len(OBJECTS)} done '
                    f'(run {_run_idx + 1}/{_n_runs}). '
                    f'Place next object and press ENTER …')
            elif _run_idx + 1 < _n_runs:
                state        = STATE_CONFIRM_NEXT_RUN
                _state_start = now
                controller.get_logger().info(
                    f'Run {_run_idx + 1}/{_n_runs} complete. '
                    f'Reposition objects for run {_run_idx + 2} and press ENTER …')
            else:
                state = STATE_DONE
                controller.get_logger().info(f'All {_n_runs} run(s) complete.')

    elif state == STATE_CONFIRM_NEXT:
        if not _confirm_pending:
            _ask_confirm_async(
                f'\n[Confirm] Place {OBJECTS[_obj_idx + 1]} and press ENTER …\n')
        elif _confirm_ready:
            _reset_trial()
            _state_start = now
            state        = STATE_CLOSE_CONFIRM

    elif state == STATE_CONFIRM_NEXT_RUN:
        if not _confirm_pending:
            _ask_confirm_async(
                f'\n[Confirm] Run {_run_idx + 1}/{_n_runs} done. '
                f'Reposition for run {_run_idx + 2} — place {OBJECTS[0]} and press ENTER …\n')
        elif _confirm_ready:
            _reset_run()
            _state_start = now
            state        = STATE_CLOSE_CONFIRM

    elif state == STATE_DONE:
        pass

# =============================================================================
# Run
# =============================================================================
timer_period = 1.0 / CONTROL_FREQUENCY
controller.create_timer(timer_period, control_callback)
controller.get_logger().info(
    f'Object stiffness dynamic — objects: {OBJECTS} | runs: {_n_runs} | '
    f'ramp durations: {RAMP_DURATIONS} s | '
    f'k_tip gentle = {K_TIP_GENTLE} N/m → probe = {K_TIP_PROBE} N/m | '
    f'fingers hold = {K_TIP_HOLD} N/m | '
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

    _close_csv()
    if _csv_path is not None:
        controller.get_logger().info(f'Last data saved to: {_csv_path}')

    arm.disconnect()
    recv.disconnect()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

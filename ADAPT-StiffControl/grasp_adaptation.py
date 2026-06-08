"""
ADAPT Hand — grasp adaptation via compliance sensing + stiffness-descent GD.

stiffness_descent: K_joint + K_task for the thumb updated each tick to drive
analytic tip force → f_des = F_GAIN / C_O.

Protocol (arm already at GRASP_POSE before running):
  1. Settle SETTLE_TIME s; hand ramps HOME → PC1 at K_TIP_GENTLE
  2. Repeat N_PROBE_ROUNDS times:
       a. Ramp thumb K → K_TIP_PROBE; wait convergence
       b. Record pos_probe, F_probe for SENSE_DURATION s
       c. (if more rounds) ramp K back to K_TIP_GENTLE; wait
  3. Compute C_O = avg(||Δpos||/||ΔF||); f_des = F_GAIN / C_O
  4. Gradient descent until |f_meas − f_des| < F_CONVERGE_THR
  5. UR5 presses −PRESS_HEIGHT (shows force); UR5 returns to GRASP_POSE
  6. Release hand (k=0); UR5 retracts to START_POSE
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
    UR5_POSE_GRASP_OBJ,
    PC1_THUMB, PC1_SPREAD, PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
    HOME_THUMB, HOME_SPREAD, HOME_FINGER,
    FINGERTIPS, OBJECTS,
    K_TIP_GENTLE, K_TIP_PROBE, K_TIP_HOLD,
    F_GAIN, GD_LR, F_CONVERGE_THR,
    K_ROT, B_ROT, B_TIP, K_RETURN, B_FLEX_DAMP,
    FRICTION_TAU_MAX,
    APPROACH_HEIGHT, PRESS_HEIGHT,
    SETTLE_TIME, RAMP_DURATION, CONVERGE_VEL_THR, CONVERGE_HOLD,
    CONVERGE_TIMEOUT, SENSE_DURATION, HOLD_TIME, N_PROBE_ROUNDS,
    PROBE_RAMP_DURATION, PROBE_CONVERGE_HOLD,
)
import rtde_control

# =============================================================================
# Collected data flag
# =============================================================================
COLLECTED_DATA = False

B_RETURN  = K_RETURN * (B_ROT / K_ROT if K_ROT else 0.0)
LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 30))

# =============================================================================
# UR5 poses
# =============================================================================
_z_below = np.array([0.0, 0.0, -APPROACH_HEIGHT, 0.0, 0.0, 0.0])
_z_press = np.array([0.0, 0.0, -PRESS_HEIGHT,    0.0, 0.0, 0.0])
UR5_START_POSE = {k: v + _z_below for k, v in UR5_POSE_GRASP_OBJ.items()}
UR5_PRESS_POSE = {k: v + _z_press for k, v in UR5_POSE_GRASP_OBJ.items()}

# =============================================================================
# Object selection
# =============================================================================
print('\nSelect object:')
for i, obj in enumerate(OBJECTS):
    print(f'  {i + 1}) {obj}')
_sel = int(input('Enter number: ')) - 1
assert 0 <= _sel < len(OBJECTS), 'Invalid selection'
OBJECT_NAME = OBJECTS[_sel]
print(f'Selected: {OBJECT_NAME}\n')

GRASP_POSE = UR5_POSE_GRASP_OBJ[OBJECT_NAME]
PRESS_POSE = UR5_PRESS_POSE[OBJECT_NAME]   # 15 cm below grasp — presses into object
START_POSE = UR5_START_POSE[OBJECT_NAME]   # 10 cm below grasp — approach/retract

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

_finger_hold_pos: dict = {}

THETA_REF_DEG = np.degrees(np.concatenate([
    PC1_THUMB,
    [PC1_SPREAD['index']], [PC1_SPREAD['middle']],
    [PC1_SPREAD['ring']],  [PC1_SPREAD['pinky']],
    PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
]))

# K_joint_dict used by the stiffness model for force/stiffness logging.
# Thumb CMC1+CMC2 active, flex joints passive (task VMC controls them).
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

# Thumb-only K dicts used by the gradient-descent (passed to stiffness_descent
# / ref_descent, which only need to see the thumb's virtual elements).
_GD_K_JOINT_INIT = {'thumb': K_ROT * np.diag([1.0, 1.0, 0.0, 0.0])}
_GD_THETA_THUMB_DEG = np.degrees(PC1_THUMB)  # 4-element, thumb only

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

# =============================================================================
# CSV
# =============================================================================

def _output_path():
    folder = os.path.join(_HERE, 'outputs', 'grasp_adaptation')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'grasp_{OBJECT_NAME}.csv')


def _csv_header():
    cols = ['time_s', 'phase', 'C_O_m_per_N',
            'f_des_x_N', 'f_des_y_N', 'f_des_z_N', 'f_des_mag_N',
            'f_meas_gd_x_N', 'f_meas_gd_y_N', 'f_meas_gd_z_N', 'f_meas_gd_mag_N',
            'converged']
    # GD thumb task stiffness (diagonal, 3 values)
    for ax in range(3):
        cols.append(f'K_thumb_task_{ax}{ax}_Npm')
    # GD thumb joint stiffness (diagonal, 4 values)
    for i in range(4):
        cols.append(f'K_thumb_joint_{i}_Nmrad')
    # GD thumb ref position
    for ax in 'xyz':
        cols.append(f'd_ref_thumb_{ax}_m')
    # GD thumb joint ref (degrees)
    for i in range(4):
        cols.append(f'theta_ref_thumb_{i}_deg')
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
    # Per-finger tip state and forces
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
    controller.get_logger().info('Data collection disabled.')

# =============================================================================
# Computation helpers
# =============================================================================

def _tip_pos(finger, q):
    r = FINGER_TIP_OFFSETS[finger]
    if finger == 'thumb':
        return np.array(FK_motor2thumbPos(q, 'IP', r))
    return np.array(FK_motor2fingerPos(q, finger, 'DIP', r))


def _thumb_K_task(k_thumb):
    """K_task dict: thumb at k_thumb, clamping fingers at K_TIP_HOLD (or k_thumb before freeze)."""
    k_hold = K_TIP_HOLD if _finger_hold_pos else k_thumb
    K = {f: (k_thumb if f == 'thumb' else k_hold) * np.eye(3) for f in FINGERTIPS}
    K['palm'] = k_thumb * np.eye(3)
    return K


def _compute_row(q, q_dot, phase, C_O, f_des, f_meas_gd,
                 gd_K_task_thumb, gd_K_joint_thumb_diag,
                 gd_d_ref_thumb, gd_theta_ref_thumb_deg,
                 converged):
    K_task_now = _thumb_K_task(_current_k)

    row = [f'{time.time() - _experiment_start:.4f}', phase,
           f'{C_O:.6e}']
    row += [f'{v:.6f}' for v in f_des]
    row.append(f'{float(np.linalg.norm(f_des)):.6f}')
    row += [f'{v:.6f}' for v in f_meas_gd]
    row.append(f'{float(np.linalg.norm(f_meas_gd)):.6f}')
    row.append(int(converged))

    # GD thumb task stiffness diagonal
    for i in range(3):
        row.append(f'{gd_K_task_thumb[i]:.6f}')
    # GD thumb joint stiffness diagonal
    for v in gd_K_joint_thumb_diag:
        row.append(f'{v:.6f}')
    # GD thumb d_ref
    row += [f'{v:.6f}' for v in gd_d_ref_thumb]
    # GD thumb theta_ref
    row += [f'{v:.6f}' for v in gd_theta_ref_thumb_deg]

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
                                      K_JOINT_DICT_MODEL, K_task_now)
        K1   = stiff_model.tip_stiffness(_f, q, K_JOINT_DICT_MODEL, K_task_now)
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
STATE_APPROACH        = 0   # trigger UR5 → GRASP_POSE
STATE_SETTLE          = 1   # wait SETTLE_TIME
STATE_RAMP_CLOSE      = 2   # hand ramps HOME → PC1 at K_TIP_GENTLE
STATE_SENSE_CONV      = 2   # wait convergence at K_TIP_GENTLE
STATE_SENSE_REC       = 3   # average pos_gentle, F_gentle
STATE_PROBE_RAMP      = 4   # ramp thumb K → K_TIP_PROBE
STATE_PROBE_CONV      = 5   # wait convergence at K_TIP_PROBE
STATE_PROBE_REC       = 6   # record probe; if rounds left → back ramp, else compute C_O
STATE_PROBE_BACK_RAMP = 7   # ramp thumb K back to K_TIP_GENTLE between rounds
STATE_ADAPT_GD        = 8   # gradient descent until force converges
STATE_PRESS           = 9   # UR5 pressing down −15 cm (shows exerted force)
STATE_RAISE           = 10  # trigger UR5 → GRASP_POSE
STATE_RELEASE         = 11  # k=0, wait CONVERGE_HOLD
STATE_RAMP_HOME       = 12  # snap joint targets, ramp to HOME + retract arm
STATE_RETRACT         = 13  # arm moving to START_POSE; hand ramping
STATE_DONE            = 14

state             = STATE_SETTLE
_state_start      = time.time()
_experiment_start = time.time()
_arm_moving       = False
_converge_ticks        = 0
_CONVERGE_TICKS        = int(CONVERGE_HOLD       * CONTROL_FREQUENCY)
_PROBE_CONVERGE_TICKS  = int(PROBE_CONVERGE_HOLD * CONTROL_FREQUENCY)
_log_tick         = 0
_converged        = False
_use_task_vmc     = False

_ramp_t0            = None
_ramp_start_targets = None
_ramp_end_targets   = None

_k_ramp_t0    = None
_k_ramp_start = None
_k_ramp_end   = None

_current_k = K_TIP_GENTLE
_C_O_mean  = 0.0

# Per-finger accumulators for the two-point compliance estimate
_pos_gentle_sum   = {f: np.zeros(3) for f in FINGERTIPS}
_force_gentle_sum = {f: np.zeros(3) for f in FINGERTIPS}
_pos_probe_sum    = {f: np.zeros(3) for f in FINGERTIPS}
_force_probe_sum  = {f: np.zeros(3) for f in FINGERTIPS}
_sense_count = 0
_probe_count = 0

# Gradient-descent state (initialised in STATE_PROBE_REC)
_gd_K_joint        = {k: v.copy() for k, v in _GD_K_JOINT_INIT.items()}
_gd_K_task         = {'thumb': K_TIP_GENTLE * np.eye(3)}
_gd_theta_ref_deg  = _GD_THETA_THUMB_DEG.copy()
_gd_d_ref          = {'thumb': D_REF['thumb'].copy()}
_gd_f_des           = np.zeros(3)
_gd_f_meas          = np.zeros(3)
_gd_converge_ticks  = 0

_probe_round  = 0   # which probe round we are on (0-indexed)


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


def _begin_k_ramp(k_start, k_end):
    global _k_ramp_t0, _k_ramp_start, _k_ramp_end
    _k_ramp_t0    = time.time()
    _k_ramp_start = k_start
    _k_ramp_end   = k_end


def _step_k_ramp(now):
    global _current_k
    alpha      = min(1.0, (now - _k_ramp_t0) / PROBE_RAMP_DURATION)
    _current_k = (1 - alpha) * _k_ramp_start + alpha * _k_ramp_end
    vmc_task.springs['thumb'].stiffness = np.full(3, _current_k)
    return alpha >= 1.0


def _gd_log_extras():
    """Return the GD tracking values for _compute_row."""
    K_task_diag  = np.diag(_gd_K_task['thumb'])
    K_joint_diag = np.diag(_gd_K_joint['thumb'])
    return (K_task_diag, K_joint_diag,
            _gd_d_ref['thumb'], _gd_theta_ref_deg)

# =============================================================================
# Control callback
# =============================================================================

def control_callback():
    global state, _state_start
    global _converge_ticks, _log_tick, _converged
    global _use_task_vmc, _C_O_mean
    global _sense_count, _probe_count, _probe_round
    global _current_k
    global _gd_K_joint, _gd_K_task, _gd_theta_ref_deg, _gd_d_ref
    global _gd_f_des, _gd_f_meas, _gd_converge_ticks

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

    # ------------------------------------------------------------------
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
            _converged      = False
            _state_start    = now
            state           = STATE_SENSE_CONV
            controller.get_logger().info(
                f'Closing at K_TIP_GENTLE = {K_TIP_GENTLE} N/m …')

    elif state == STATE_SENSE_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged   = True
                _log_tick    = 0
                _sense_count = 0
                for _f in FINGERTIPS:
                    _pos_gentle_sum[_f]   = np.zeros(3)
                    _force_gentle_sum[_f] = np.zeros(3)
                _state_start = now
                state        = STATE_SENSE_REC
                controller.get_logger().info(
                    f'Converged at {elapsed:.1f} s. Recording gentle point '
                    f'({K_TIP_GENTLE} N/m) for {SENSE_DURATION:.1f} s …')

    elif state == STATE_SENSE_REC:
        _log_tick += 1
        if _log_tick % LOG_EVERY == 0:
            for _f in FINGERTIPS:
                K_task_now = _thumb_K_task(K_TIP_GENTLE)
                _pos_gentle_sum[_f]   += _tip_pos(_f, q)
                _force_gentle_sum[_f] += np.asarray(stiff_model.tip_force(
                    _f, q, THETA_REF_DEG, _d_ref_model_dict,
                    K_JOINT_DICT_MODEL, K_task_now))
            _sense_count += 1
            if not COLLECTED_DATA:
                extras = _gd_log_extras()
                _csv_writer.writerow(_compute_row(
                    q, q_dot, 'sense', 0.0,
                    _gd_f_des, _gd_f_meas,
                    *extras, converged=True))
        if elapsed >= SENSE_DURATION:
            if not COLLECTED_DATA:
                _csv_file.flush()
            _freeze_and_hold_fingers(q)
            controller.get_logger().info(
                f'Gentle point recorded ({_sense_count} samples). '
                f'Fingers frozen. Ramping thumb K {K_TIP_GENTLE} → {K_TIP_PROBE} N/m …')
            _begin_k_ramp(K_TIP_GENTLE, K_TIP_PROBE)
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = STATE_PROBE_RAMP

    elif state == STATE_PROBE_RAMP:
        if _step_k_ramp(now):
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = STATE_PROBE_CONV

    elif state == STATE_PROBE_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _PROBE_CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged   = True
                _log_tick    = 0
                _state_start = now
                state        = STATE_PROBE_REC
                controller.get_logger().info(
                    f'Probe round {_probe_round + 1}/{N_PROBE_ROUNDS} converged at {elapsed:.1f} s. '
                    f'Recording ({K_TIP_PROBE} N/m) for {SENSE_DURATION:.1f} s …')

    elif state == STATE_PROBE_REC:
        _log_tick += 1
        if _log_tick % LOG_EVERY == 0:
            for _f in FINGERTIPS:
                K_task_now = _thumb_K_task(K_TIP_PROBE)
                _pos_probe_sum[_f]   += _tip_pos(_f, q)
                _force_probe_sum[_f] += np.asarray(stiff_model.tip_force(
                    _f, q, THETA_REF_DEG, _d_ref_model_dict,
                    K_JOINT_DICT_MODEL, K_task_now))
            _probe_count += 1
            if not COLLECTED_DATA:
                extras = _gd_log_extras()
                _csv_writer.writerow(_compute_row(
                    q, q_dot, 'probe', 0.0,
                    _gd_f_des, _gd_f_meas,
                    *extras, converged=True))
        if elapsed >= SENSE_DURATION:
            if not COLLECTED_DATA:
                _csv_file.flush()

            if _probe_round < N_PROBE_ROUNDS - 1:
                # More rounds: ramp K back to K_TIP_GENTLE then repeat probe
                controller.get_logger().info(
                    f'Probe round {_probe_round + 1}/{N_PROBE_ROUNDS} done. '
                    f'Ramping back to K_TIP_GENTLE …')
                _probe_round += 1
                _begin_k_ramp(K_TIP_PROBE, K_TIP_GENTLE)
                _converge_ticks = 0
                _converged      = False
                _log_tick       = 0
                _state_start    = now
                state           = STATE_PROBE_BACK_RAMP
            else:
                # All rounds done: compute C_O and start GD
                n_g   = max(1, _sense_count)
                n_p   = max(1, _probe_count)
                d_pos = _pos_probe_sum['thumb'] / n_p - _pos_gentle_sum['thumb'] / n_g
                d_F   = _force_probe_sum['thumb'] / n_p - _force_gentle_sum['thumb'] / n_g
                nF    = float(np.linalg.norm(d_F))
                if nF > 1e-9:
                    _C_O_mean = float(np.linalg.norm(d_pos) / nF)
                else:
                    _C_O_mean = 0.0
                f_des_mag = F_GAIN / _C_O_mean if _C_O_mean > 1e-12 else 0.0

                # Set f_des direction from initial tip force at probe stiffness
                f_init = np.asarray(stiff_model.tip_force(
                    'thumb', q, _GD_THETA_THUMB_DEG, {'thumb': _d_ref_model_dict['thumb']},
                    _GD_K_JOINT_INIT, {'thumb': K_TIP_PROBE * np.eye(3)}))
                f_init_mag = float(np.linalg.norm(f_init))
                if f_init_mag > 1e-9:
                    _gd_f_des = f_des_mag * f_init / f_init_mag
                else:
                    _gd_f_des = np.array([0.0, 0.0, -f_des_mag])

                # Initialise GD state (both modes use same K; ref mode keeps K fixed)
                _gd_K_joint       = {'thumb': K_ROT * np.diag([1.0, 1.0, 0.0, 0.0])}
                _gd_K_task        = {'thumb': K_TIP_PROBE * np.eye(3)}
                _gd_theta_ref_deg = _GD_THETA_THUMB_DEG.copy()
                _gd_d_ref         = {'thumb': _d_ref_model_dict['thumb'].copy()}

                controller.get_logger().info(
                    f'All {N_PROBE_ROUNDS} probe rounds done. '
                    f'C_O = {_C_O_mean * 1e3:.2f} mm/N  '
                    f'→ f_des = {f_des_mag:.3f} N')
                _gd_converge_ticks = 0
                _log_tick          = 0
                _state_start       = now
                state              = STATE_ADAPT_GD

    elif state == STATE_PROBE_BACK_RAMP:
        if _step_k_ramp(now):
            # K is back at K_TIP_GENTLE — start next probe ramp
            controller.get_logger().info(
                f'Starting probe round {_probe_round + 1}/{N_PROBE_ROUNDS} …')
            _begin_k_ramp(K_TIP_GENTLE, K_TIP_PROBE)
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = STATE_PROBE_RAMP

    elif state == STATE_ADAPT_GD:
        # Compute analytic tip force with current GD parameters
        _gd_f_meas = np.asarray(stiff_model.tip_force(
            'thumb', q, _gd_theta_ref_deg, _gd_d_ref,
            _gd_K_joint, _gd_K_task))

        _gd_K_joint, _gd_K_task = stiff_model.stiffness_descent(
            'thumb', q, _gd_theta_ref_deg, _gd_d_ref,
            _gd_K_joint, _gd_K_task, _gd_f_meas, _gd_f_des,
            lr_joint=GD_LR, lr_task=GD_LR)
        k_task_diag = np.maximum(np.diag(_gd_K_task['thumb']), 0.0)
        vmc_task.springs['thumb'].stiffness = k_task_diag
        _current_k = float(np.mean(k_task_diag))
        k_jt = np.diag(_gd_K_joint['thumb'])
        vmc_joint.stiffness['thumb'] = np.maximum(
            np.array([k_jt[0], k_jt[1], 0.0, 0.0]), 0.0)

        f_err = float(np.linalg.norm(_gd_f_meas - _gd_f_des))
        if f_err < F_CONVERGE_THR:
            _gd_converge_ticks += 1
        else:
            _gd_converge_ticks = 0
        gd_converged = _gd_converge_ticks >= _CONVERGE_TICKS

        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            extras = _gd_log_extras()
            _csv_writer.writerow(_compute_row(
                q, q_dot, 'adapt_gd', _C_O_mean,
                _gd_f_des, _gd_f_meas,
                *extras, converged=gd_converged))

        if gd_converged or elapsed >= CONVERGE_TIMEOUT:
            if not COLLECTED_DATA:
                _csv_file.flush()
            controller.get_logger().info(
                f'GD {"converged" if gd_converged else "timed out"} at {elapsed:.1f} s. '
                f'|f_meas - f_des| = {f_err:.4f} N. Pressing down …')
            _log_tick    = 0
            _state_start = now
            _move_arm_async(PRESS_POSE, UR5_INIT_SPEED, STATE_RAISE)
            state        = STATE_PRESS

    elif state == STATE_PRESS:
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            extras = _gd_log_extras()
            _csv_writer.writerow(_compute_row(
                q, q_dot, 'press', _C_O_mean,
                _gd_f_des, _gd_f_meas,
                *extras, converged=True))

    elif state == STATE_RAISE:
        if not _arm_moving:
            controller.get_logger().info('Raising back to grasp pose …')
            _log_tick = 0
            _move_arm_async(GRASP_POSE, UR5_INIT_SPEED, STATE_RELEASE)

    elif state == STATE_RELEASE:
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            extras = _gd_log_extras()
            _csv_writer.writerow(_compute_row(
                q, q_dot, 'raise', _C_O_mean,
                _gd_f_des, _gd_f_meas,
                *extras, converged=True))
        if not _arm_moving:
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
            _move_arm_async(START_POSE, UR5_INIT_SPEED, STATE_DONE)
            _state_start = now
            state        = STATE_RETRACT

    elif state == STATE_RETRACT:
        if _step_ramp(now):
            controller.get_logger().info('Hand home.')

    elif state == STATE_DONE:
        pass

# =============================================================================
# Run
# =============================================================================
timer_period = 1.0 / CONTROL_FREQUENCY
controller.create_timer(timer_period, control_callback)
controller.get_logger().info(
    f'Grasp adaptation | object: {OBJECT_NAME} | '
    f'K_TIP_GENTLE={K_TIP_GENTLE} N/m | K_TIP_PROBE={K_TIP_PROBE} N/m | '
    f'F_GAIN={F_GAIN} | GD_LR={GD_LR} | F_CONVERGE_THR={F_CONVERGE_THR} N')

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

"""
ADAPT Hand — grasp adaptation via online compliance sensing.

The hand descends to grasp one of two known objects, closes at a gentle
stiffness to sense the object's compliance from fingertip displacement,
then adapts the grasp stiffness according to:

    k_applied = clip( K_SCALE × δ_mean, K_MIN, K_MAX )

where δ_mean is the mean fingertip displacement magnitude (all five fingers,
steady-state average). Stiff objects block the fingers further from the
PC1 target → larger δ → higher adapted stiffness. After adaptation the
hand lifts the object, holds, then places it back and returns home.

Protocol per object:
  1. UR5 moves to ABOVE_POSE (approach from above).
  2. UR5 descends to GRASP_POSE.
  3. Hand closes to PC1 at K_TIP_GENTLE (sensing phase).
  4. Wait SENSE_DURATION; accumulate δ_mean.
  5. Compute k_applied; ramp stiffness from K_TIP_GENTLE → k_applied.
  6. Wait for convergence at k_applied.
  7. UR5 lifts to LIFT_POSE.
  8. Hold HOLD_TIME.
  9. UR5 returns to GRASP_POSE (place back).
  10. Open hand; return home.

Outputs: ADAPT-StiffControl/outputs/grasp_adaptation/grasp_<object>_N.csv
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
    FK_motor2wrist, FK_motor2thumb, FK_motor2finger, FK_motor2spread,
    FK_motor2thumbPos, FK_motor2fingerPos, FK_motor2palm,
    joint_to_motor,
)
from ModelIDHand.hand_params import FINGER_TIP_OFFSETS
from StiffnessModelHand.stiffness2mixedspace import tip_stiffness_MixedSpace
from UR5_codes.UR5_config import UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCELERATION
from UR5_codes.UR5_readPose import UR5Receiver
from hand_config import (
    UR5_POSE_GRASP_OBJ,
    PC1_WRIST, PC1_THUMB, PC1_SPREAD, PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
    HOME_WRIST, HOME_THUMB, HOME_SPREAD, HOME_FINGER,
    FINGERTIPS, OBJECTS,
    K_TIP_GENTLE, K_SCALE, K_MIN, K_MAX,
    K_ROT, B_ROT, B_TIP, K_RETURN, B_FLEX_DAMP,
    APPROACH_HEIGHT, LIFT_HEIGHT,
    SETTLE_TIME, RAMP_DURATION, CONVERGE_VEL_THR, CONVERGE_HOLD,
    CONVERGE_TIMEOUT, SENSE_DURATION, HOLD_TIME,
)
import rtde_control

COLLECTED_DATA = False

B_RETURN  = K_RETURN * (B_ROT / K_ROT if K_ROT else 0.0)
LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 30))

# =============================================================================
# UR5 object poses (from hand_config.py)
# =============================================================================
# Pre-compute derived poses.
_lift_offset = np.array([0.0, 0.0, LIFT_HEIGHT,    0.0, 0.0, 0.0])
_appr_offset = np.array([0.0, 0.0, APPROACH_HEIGHT, 0.0, 0.0, 0.0])
UR5_LIFT_POSE  = {k: v + _lift_offset for k, v in UR5_POSE_GRASP_OBJ.items()}
UR5_ABOVE_POSE = {k: v + _appr_offset for k, v in UR5_POSE_GRASP_OBJ.items()}


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
LIFT_POSE  = UR5_LIFT_POSE[OBJECT_NAME]
ABOVE_POSE = UR5_ABOVE_POSE[OBJECT_NAME]

# =============================================================================
# Derived quantities
# =============================================================================
Q_TARGET = joint_to_motor(
    PC1_WRIST, PC1_THUMB,
    PC1_SPREAD['index'],
    PC1_INDEX[:2], PC1_MIDDLE[:2], PC1_RING[:2], PC1_PINKY[:2],
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
    PC1_WRIST, PC1_THUMB,
    [PC1_SPREAD['index']], [PC1_SPREAD['middle']],
    [PC1_SPREAD['ring']],  [PC1_SPREAD['pinky']],
    PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
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

vmc_task.springs['palm'].stiffness = np.full(3, K_TIP_GENTLE)
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
    folder = os.path.join(_HERE, 'outputs', 'grasp_adaptation')
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder)
                if f.startswith(f'grasp_{OBJECT_NAME}_') and f.endswith('.csv')]
    n = max((int(f.replace(f'grasp_{OBJECT_NAME}_', '').replace('.csv', ''))
             for f in existing), default=0) + 1
    return os.path.join(folder, f'grasp_{OBJECT_NAME}_{n}.csv')


def _csv_header():
    cols = ['time_s', 'phase', 'k_tip_Npm', 'delta_mean_m', 'k_applied_Npm', 'converged']
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
        cols.append(f'disp_{_f}_mag_m')
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


def _compute_row(q, q_dot, phase, k_tip, delta_mean, k_applied, converged):
    row = [f'{time.time() - _experiment_start:.4f}', phase,
           f'{k_tip:.2f}', f'{delta_mean:.6f}', f'{k_applied:.2f}', int(converged)]

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
        mag  = float(np.linalg.norm(disp))

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
        row.append(f'{mag:.6f}')
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
STATE_APPROACH    = 0
STATE_DESCEND     = 1
STATE_SETTLE_ARM  = 2
STATE_RAMP_CLOSE  = 3
STATE_SENSE_CONV  = 4
STATE_SENSE_REC   = 5
STATE_ADAPT_RAMP  = 6
STATE_ADAPT_CONV  = 7
STATE_LIFT        = 8
STATE_HOLD        = 9
STATE_PLACE       = 10
STATE_UNLOAD      = 11
STATE_RAMP_HOME   = 12
STATE_RETURN      = 13
STATE_DONE        = 14

state             = STATE_APPROACH
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
_k_ramp_start = None
_k_ramp_end   = None

_current_k    = K_TIP_GENTLE
_k_applied    = K_TIP_GENTLE
_delta_mean   = 0.0
_disp_accum   = 0.0
_disp_count   = 0


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


def _begin_k_ramp(k_start, k_end):
    global _k_ramp_t0, _k_ramp_start, _k_ramp_end
    _k_ramp_t0    = time.time()
    _k_ramp_start = k_start
    _k_ramp_end   = k_end


def _step_k_ramp(now):
    global _current_k
    alpha      = min(1.0, (now - _k_ramp_t0) / RAMP_DURATION)
    _current_k = (1 - alpha) * _k_ramp_start + alpha * _k_ramp_end
    _set_task_stiffness(_current_k)
    return alpha >= 1.0

# =============================================================================
# Control callback
# =============================================================================

def control_callback():
    global state, _state_start
    global _converge_ticks, _log_tick, _converged
    global _use_task_vmc, _k_applied, _delta_mean
    global _disp_accum, _disp_count

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

    if state == STATE_APPROACH:
        if not _arm_moving:
            controller.get_logger().info(
                f'Approaching above {OBJECT_NAME} …')
            _move_arm_async(ABOVE_POSE, UR5_INIT_SPEED, STATE_DESCEND)

    elif state == STATE_DESCEND:
        if not _arm_moving:
            controller.get_logger().info('Descending to grasp pose …')
            _move_arm_async(GRASP_POSE, UR5_INIT_SPEED, STATE_SETTLE_ARM)

    elif state == STATE_SETTLE_ARM:
        if elapsed >= SETTLE_TIME:
            controller.get_logger().info(
                f'Arm settled. Ramping HOME → PC1 over {RAMP_DURATION:.1f} s …')
            _begin_ramp(PC1_POSE_TARGETS)
            _state_start = now
            state        = STATE_RAMP_CLOSE

    elif state == STATE_RAMP_CLOSE:
        if _step_ramp(now):
            _set_joint_stiffness_experiment()
            _set_task_stiffness(K_TIP_GENTLE)
            for _f in FINGERTIPS:
                vmc_task.targets[_f] = D_REF[_f].copy()
            _use_task_vmc   = True
            _converge_ticks = 0
            _converged      = False
            _state_start    = now
            state           = STATE_SENSE_CONV
            controller.get_logger().info(
                f'Closing at K_TIP_GENTLE = {K_TIP_GENTLE} N/m for sensing …')

    elif state == STATE_SENSE_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged    = True
                _log_tick     = 0
                _disp_accum   = 0.0
                _disp_count   = 0
                _state_start  = now
                state         = STATE_SENSE_REC
                controller.get_logger().info(
                    f'Sensing converged at {elapsed:.1f} s. '
                    f'Recording {SENSE_DURATION:.1f} s …')

    elif state == STATE_SENSE_REC:
        _log_tick += 1
        # Accumulate mean displacement across all five fingertips.
        if _log_tick % LOG_EVERY == 0:
            sample = float(np.mean([
                np.linalg.norm(_tip_pos(_f, q) - D_REF[_f]) for _f in FINGERTIPS
            ]))
            _disp_accum += sample
            _disp_count += 1
            if not COLLECTED_DATA:
                _csv_writer.writerow(
                    _compute_row(q, q_dot, 'sense', K_TIP_GENTLE,
                                 sample, K_TIP_GENTLE, True))
        if elapsed >= SENSE_DURATION:
            if not COLLECTED_DATA:
                _csv_file.flush()
            _delta_mean = _disp_accum / max(1, _disp_count)
            _k_applied  = float(np.clip(K_SCALE * _delta_mean, K_MIN, K_MAX))
            controller.get_logger().info(
                f'δ_mean = {_delta_mean * 1e3:.2f} mm → '
                f'k_applied = {_k_applied:.1f} N/m '
                f'(K_SCALE × δ / clip [{K_MIN}, {K_MAX}])')
            _begin_k_ramp(K_TIP_GENTLE, _k_applied)
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = STATE_ADAPT_RAMP

    elif state == STATE_ADAPT_RAMP:
        if _step_k_ramp(now):
            _converge_ticks = 0
            _converged      = False
            _log_tick       = 0
            _state_start    = now
            state           = STATE_ADAPT_CONV

    elif state == STATE_ADAPT_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'adapt', _current_k,
                             _delta_mean, _k_applied, False))
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged   = True
                _log_tick    = 0
                _state_start = now
                controller.get_logger().info(
                    f'Adapted grasp converged at {elapsed:.1f} s. Lifting …')
                _move_arm_async(LIFT_POSE, UR5_INIT_SPEED, STATE_HOLD)
                state = STATE_LIFT

    elif state == STATE_LIFT:
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'lift', _current_k,
                             _delta_mean, _k_applied, True))

    elif state == STATE_HOLD:
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'hold', _current_k,
                             _delta_mean, _k_applied, True))
        if elapsed >= HOLD_TIME:
            if not COLLECTED_DATA:
                _csv_file.flush()
            controller.get_logger().info('Placing back …')
            _log_tick = 0
            _move_arm_async(GRASP_POSE, UR5_INIT_SPEED, STATE_UNLOAD)
            state = STATE_PLACE

    elif state == STATE_PLACE:
        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'place', _current_k,
                             _delta_mean, _k_applied, True))

    elif state == STATE_UNLOAD:
        controller.get_logger().info('Releasing grasp …')
        _set_task_stiffness(0.0)
        _use_task_vmc = False
        _state_start  = now
        state         = STATE_RAMP_HOME

    elif state == STATE_RAMP_HOME:
        if elapsed >= CONVERGE_HOLD:
            controller.get_logger().info(
                f'Contact released. Ramping PC1 → HOME over {RAMP_DURATION:.1f} s …')
            _set_joint_stiffness_uniform(K_RETURN, B_RETURN)
            _begin_ramp(HOME_POSE_TARGETS)
            _state_start = now
            state        = STATE_RETURN

    elif state == STATE_RETURN:
        if _step_ramp(now):
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
    f'Grasp adaptation | object: {OBJECT_NAME} | '
    f'K_TIP_GENTLE = {K_TIP_GENTLE} N/m | '
    f'K_SCALE = {K_SCALE} | '
    f'k_applied ∈ [{K_MIN}, {K_MAX}] N/m')

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

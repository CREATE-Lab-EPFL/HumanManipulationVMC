"""
Tunable compliance — dynamic grasping of a cylinder (ADAPT Hand).

The hand grasps a cylinder on the table, then the UR5 transports it
horizontally for TRANSPORT_DISTANCE at TRANSPORT_SPEED. Three conditions:

  'stiff'     — K_STIFF throughout; object tends to fall (no compliance)
  'compliant' — K_COMPLIANT throughout; grip force too low under dynamics
  'adaptive'  — K_SOFT_INIT at grasp so the hand conforms; once the UR5
                has moved STIFFENING_DIST the stiffness ramps to K_RIGID

Protocol:
  1. UR5 moves to UR5_POSE_GRASP; hand settles.
  2. Hand closes to PC1 pose with condition-dependent stiffness.
  3. Wait for convergence (GRASP_SETTLE phase).
  4. UR5 moves forward TRANSPORT_DISTANCE; adaptive case stiffens on trigger.
  5. UR5 returns to grasp start.
  6. Hand unloads and returns home.

Outputs: TunableCompliance/Hand/outputs/dynamic_grasp/dynamic_grasp_<cond>_N.csv
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
    UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCELERATION,
)
from UR5_codes.UR5_readPose import UR5Receiver
import rtde_control
import rtde_receive

# =============================================================================
# Custom parameters
# =============================================================================
K_STIFF        = 150.0   # [N/m]  always-stiff condition
K_COMPLIANT    = 5.0     # [N/m]  always-compliant condition
K_SOFT_INIT    = 5.0     # [N/m]  adaptive — initial soft stiffness
K_RIGID        = 150.0   # [N/m]  adaptive — post-trigger rigid stiffness
STIFFENING_DIST = 0.05   # [m]    UR5 displacement that triggers stiffening
COLLECT_DATA   = False

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
GRASP_SETTLE     = 3.0   # [s]  hold before transport starts
RECORD_DURATION  = 5.0   # [s]  post-transport recording

LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 30))

# =============================================================================
# UR5 poses — PLACEHOLDER: tune to actual cylinder position
# =============================================================================
# Hand positioned around the cylinder, palm facing down/forward.
UR5_POSE_GRASP = np.array([0.0, 0.50, 0.15, 0.04, -2.18, -2.18])

# Pure horizontal transport along the X axis of the base frame.
TRANSPORT_DIRECTION = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
TRANSPORT_DISTANCE  = 0.30   # [m]
TRANSPORT_SPEED     = 0.10   # [m/s]

_dir_unit      = TRANSPORT_DIRECTION[:3] / np.linalg.norm(TRANSPORT_DIRECTION[:3])
_dir6          = np.concatenate([_dir_unit, [0.0, 0.0, 0.0]])
UR5_POSE_TRANSPORT_END = UR5_POSE_GRASP + _dir6 * TRANSPORT_DISTANCE

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

FINGERTIPS = ['thumb', 'index', 'middle', 'ring', 'pinky']
CONDITIONS = ['stiff', 'compliant', 'adaptive']

# =============================================================================
# Condition selection
# =============================================================================
print('\nSelect condition:')
for i, c in enumerate(CONDITIONS):
    print(f'  {i + 1}) {c}')
_sel = int(input('Enter number: ')) - 1
assert 0 <= _sel < len(CONDITIONS), 'Invalid selection'
CONDITION = CONDITIONS[_sel]
print(f'Selected: {CONDITION}\n')

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

# Initial stiffness dict per condition.
_K_INIT = {
    'stiff':     K_STIFF,
    'compliant': K_COMPLIANT,
    'adaptive':  K_SOFT_INIT,
}
K_DICT_INIT  = {f: _K_INIT[CONDITION]  for f in FINGERTIPS}
K_DICT_RIGID = {f: K_RIGID             for f in FINGERTIPS}

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

vmc_task.springs['palm'].stiffness = np.full(3, _K_INIT[CONDITION])
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

arm      = rtde_control.RTDEControlInterface(UR5_IP)
arm_recv = rtde_receive.RTDEReceiveInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info('UR5 connected')

# =============================================================================
# CSV
# =============================================================================

def _output_path():
    folder = os.path.join(_HERE, 'outputs', 'dynamic_grasp')
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder)
                if f.startswith(f'dynamic_grasp_{CONDITION}_') and f.endswith('.csv')]
    n = max((int(f.replace(f'dynamic_grasp_{CONDITION}_', '').replace('.csv', ''))
             for f in existing), default=0) + 1
    return os.path.join(folder, f'dynamic_grasp_{CONDITION}_{n}.csv')


def _csv_header():
    cols = ['time_s', 'phase', 'converged', 'ur5_disp_m']
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


def _ur5_displacement():
    pose = np.array(arm_recv.getActualTCPPose())
    return float(np.linalg.norm(pose[:3] - _grasp_tcp_pos))


def _compute_row(q, q_dot, phase, k_dict, converged, ur5_disp):
    row = [f'{time.time() - _experiment_start:.4f}', phase, int(converged),
           f'{ur5_disp:.4f}']
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
    K_task_now['palm'] = k_dict['thumb'] * np.eye(3)

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
STATE_INIT_ARM      = 0
STATE_SETTLE_ARM    = 1
STATE_RAMP_TO_GRASP = 2
STATE_GRASP_CONV    = 3
STATE_GRASP_SETTLE  = 4
STATE_TRANSPORT     = 5
STATE_RETURN_ARM    = 6
STATE_UNLOAD        = 7
STATE_RAMP_TO_HOME  = 8
STATE_RETURN        = 9
STATE_DONE          = 10

state             = STATE_INIT_ARM
_state_start      = time.time()
_experiment_start = time.time()
_arm_moving       = False
_converge_ticks   = 0
_CONVERGE_TICKS   = int(CONVERGE_HOLD * CONTROL_FREQUENCY)
_log_tick         = 0
_converged        = False
_use_task_vmc     = False
_triggered        = False   # stiffening trigger fired (adaptive only)
_k_ramping        = False

_ramp_t0            = None
_ramp_start_targets = None
_ramp_end_targets   = None

_k_ramp_t0    = None
_k_ramp_start = {}
_k_ramp_end   = {}

_current_k_dict = {f: _K_INIT[CONDITION] for f in FINGERTIPS}
_grasp_tcp_pos  = None   # TCP position at transport start (for displacement)


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
    vmc_task.springs['palm'].stiffness = np.full(3, k_dict['thumb'])


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


def _begin_k_ramp(start_dict, end_dict):
    global _k_ramp_t0, _k_ramp_start, _k_ramp_end, _k_ramping
    _k_ramp_t0    = time.time()
    _k_ramp_start = {f: start_dict[f] for f in FINGERTIPS}
    _k_ramp_end   = {f: end_dict[f]   for f in FINGERTIPS}
    _k_ramping    = True


def _step_k_ramp(now):
    global _current_k_dict, _k_ramping
    alpha = min(1.0, (now - _k_ramp_t0) / RAMP_DURATION)
    for _f in FINGERTIPS:
        _current_k_dict[_f] = (1 - alpha) * _k_ramp_start[_f] + alpha * _k_ramp_end[_f]
    _set_task_stiffness(_current_k_dict)
    if alpha >= 1.0:
        _k_ramping = False

# =============================================================================
# Control callback
# =============================================================================

def control_callback():
    global state, _state_start, _arm_moving
    global _converge_ticks, _log_tick, _converged
    global _use_task_vmc, _triggered, _grasp_tcp_pos

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
            controller.get_logger().info('Moving UR5 to grasp pose …')
            _move_arm_async(UR5_POSE_GRASP, UR5_INIT_SPEED, STATE_SETTLE_ARM)

    elif state == STATE_SETTLE_ARM:
        if elapsed >= SETTLE_TIME:
            controller.get_logger().info(
                f'UR5 settled. Ramping HOME → PC1 over {RAMP_DURATION:.1f} s …')
            _begin_ramp(PC1_POSE_TARGETS)
            _state_start = now
            state        = STATE_RAMP_TO_GRASP

    elif state == STATE_RAMP_TO_GRASP:
        if _step_ramp(now):
            _set_joint_stiffness_experiment()
            _set_task_stiffness(K_DICT_INIT)
            for _f in FINGERTIPS:
                vmc_task.targets[_f] = D_REF[_f].copy()
            _use_task_vmc   = True
            _converge_ticks = 0
            _converged      = False
            _state_start    = now
            state           = STATE_GRASP_CONV
            controller.get_logger().info(
                f'Ramp done. Closing on cylinder at k = {_K_INIT[CONDITION]} N/m …')

    elif state == STATE_GRASP_CONV:
        if np.max(np.abs(q_dot)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0
        if (_converge_ticks >= _CONVERGE_TICKS) or (elapsed >= CONVERGE_TIMEOUT):
            if not _converged:
                _converged   = True
                _log_tick    = 0
                _state_start = now
                state        = STATE_GRASP_SETTLE
                controller.get_logger().info(
                    f'Grasp converged at {elapsed:.1f} s. '
                    f'Settling {GRASP_SETTLE:.1f} s before transport …')

    elif state == STATE_GRASP_SETTLE:
        _log_tick += 1
        if COLLECT_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'grasp_settle', _current_k_dict, True, 0.0))
        if elapsed >= GRASP_SETTLE:
            _grasp_tcp_pos = np.array(arm_recv.getActualTCPPose())[:3]
            _log_tick      = 0
            _state_start   = now
            _move_arm_async(UR5_POSE_TRANSPORT_END, TRANSPORT_SPEED, STATE_RETURN_ARM)
            state = STATE_TRANSPORT
            controller.get_logger().info(
                f'Transport started — moving {TRANSPORT_DISTANCE:.2f} m at '
                f'{TRANSPORT_SPEED:.2f} m/s …')

    elif state == STATE_TRANSPORT:
        if _k_ramping:
            _step_k_ramp(now)

        ur5_disp = _ur5_displacement()

        if CONDITION == 'adaptive' and not _triggered:
            if ur5_disp >= STIFFENING_DIST:
                _triggered = True
                controller.get_logger().info(
                    f'Stiffening trigger at {ur5_disp:.3f} m displacement — '
                    f'ramping {K_SOFT_INIT} → {K_RIGID} N/m over {RAMP_DURATION:.1f} s …')
                _begin_k_ramp(_current_k_dict, K_DICT_RIGID)

        _log_tick += 1
        if COLLECT_DATA and _log_tick % LOG_EVERY == 0:
            _csv_writer.writerow(
                _compute_row(q, q_dot, 'transport', _current_k_dict, True, ur5_disp))

        # STATE_RETURN_ARM is set by the UR5 thread when moveL completes.

    elif state == STATE_RETURN_ARM:
        if not _arm_moving:
            controller.get_logger().info('UR5 returning to grasp start …')
            _move_arm_async(UR5_POSE_GRASP, TRANSPORT_SPEED, STATE_UNLOAD)

    elif state == STATE_UNLOAD:
        if COLLECT_DATA:
            _csv_file.flush()
        controller.get_logger().info('Releasing contact …')
        _set_task_stiffness({f: 0.0 for f in FINGERTIPS})
        _use_task_vmc = False
        _state_start  = now
        state         = STATE_RAMP_TO_HOME

    elif state == STATE_RAMP_TO_HOME:
        if elapsed >= CONVERGE_HOLD:
            controller.get_logger().info(
                f'Contact released. Ramping PC1 → HOME over {RAMP_DURATION:.1f} s …')
            _set_joint_stiffness_uniform(K_RETURN, B_RETURN)
            _begin_ramp(HOME_POSE_TARGETS)
            _state_start = now
            state        = STATE_RETURN

    elif state == STATE_RETURN:
        if _step_ramp(now):
            _converge_ticks = 0
            state           = STATE_DONE
            controller.get_logger().info('Home reached. Experiment complete.')

    elif state == STATE_DONE:
        pass

# =============================================================================
# Run
# =============================================================================
timer_period = 1.0 / CONTROL_FREQUENCY
controller.create_timer(timer_period, control_callback)
controller.get_logger().info(
    f'Dynamic grasp | condition: {CONDITION} | '
    f'k_init = {_K_INIT[CONDITION]} N/m | '
    f'transport = {TRANSPORT_DISTANCE:.2f} m @ {TRANSPORT_SPEED:.2f} m/s')

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

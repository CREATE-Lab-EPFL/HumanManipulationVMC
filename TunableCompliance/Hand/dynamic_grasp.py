"""
Tunable compliance — dynamic grasping of a bottle (ADAPT Hand).

Runs all three conditions back-to-back in a single session:
  1) stiff    — K_TIP = K_STIFF throughout
  2) soft     — K_TIP = K_SOFT throughout
  3) adaptive — K_TIP = K_SOFT at close; ramps to K_STIFF after SOFT_DURATION

After each trial the hand returns home (at K_HOME stiffness) and the UR5 arm
resets to UR5_POSE_BOTTLE_START.  The operator confirms before the next trial.

Outputs: TunableCompliance/Hand/outputs/dynamic_grasp/dynamic_grasp_<cond>.csv
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
from KinematicsHand.FK_Hand     import (
    FK_motor2thumbPos, FK_motor2fingerPos, FK_motor2palm,
    joint_to_motor,
)
from ModelIDHand.hand_params    import FINGER_TIP_OFFSETS, eta
from StiffnessModelHand.stiffness2mixedspace import tip_stiffness_MixedSpace
from UR5_codes.UR5_config       import UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCELERATION
from UR5_codes.UR5_readPose     import UR5Receiver
from hand_config import (
    UR5_POSE_BOTTLE_START,
    GRASP_PC1_THUMB  as PC1_THUMB,
    GRASP_PC1_SPREAD as PC1_SPREAD,
    GRASP_PC1_INDEX  as PC1_INDEX,
    GRASP_PC1_MIDDLE as PC1_MIDDLE,
    GRASP_PC1_RING   as PC1_RING,
    GRASP_PC1_PINKY  as PC1_PINKY,
    HOME_THUMB, HOME_SPREAD, HOME_FINGER,
    FINGERTIPS,
    K_SOFT, K_STIFF, SOFT_DURATION, K_RAMP_DURATION,
    K_ROT, B_ROT, B_TIP, K_HOME,
    FRICTION_TAU_MAX,
    APPROACH_DIRECTION, APPROACH_SPEED, TOTAL_DISTANCE, CLOSE_DISTANCE, HOME_DURATION,
)
import rtde_control

# =============================================================================
# Collected data
# =============================================================================
COLLECTED_DATA = False

LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 30))

# =============================================================================
# Conditions — run in this order
# =============================================================================
CONDITIONS_ORDER = ['stiff', 'soft', 'adaptive']
_K_INIT_MAP = {'soft': K_SOFT, 'stiff': K_STIFF, 'adaptive': K_SOFT}

# =============================================================================
# UR5 trajectory (fixed geometry, same for every trial)
# =============================================================================
CLOSE_POSE = UR5_POSE_BOTTLE_START.copy()
CLOSE_POSE[:3] += CLOSE_DISTANCE * APPROACH_DIRECTION

_POST_D     = TOTAL_DISTANCE - CLOSE_DISTANCE
_POST_Z     = _POST_D / 2.0
_POST_SPEED = np.sqrt(APPROACH_SPEED**2 + (APPROACH_SPEED / 2)**2)

FINAL_POSE = CLOSE_POSE.copy()
FINAL_POSE[:3] += _POST_D * APPROACH_DIRECTION
FINAL_POSE[2]  += _POST_Z

_CLOSE_TIME_S = CLOSE_DISTANCE / APPROACH_SPEED

# =============================================================================
# FK targets for task spring
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

THETA_REF_DEG = np.degrees(np.concatenate([
    PC1_THUMB,
    [PC1_SPREAD['index']], [PC1_SPREAD['middle']],
    [PC1_SPREAD['ring']],  [PC1_SPREAD['pinky']],
    PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
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
    vmc_task.dampers[_f].damping    = np.full(3, B_TIP)
    vmc_task.targets[_f]            = D_REF[_f].copy()
    vmc_task.attachment_points[_f]  = FINGER_TIP_OFFSETS[_f].copy()

vmc_task.dampers['palm'].damping   = np.full(3, B_TIP)
vmc_task.targets['palm']           = D_REF['palm'].copy()

grav_lim = GravFricLim()
grav_lim.friction_max = FRICTION_TAU_MAX
recv     = UR5Receiver()

print('Initialising stiffness model …')
_t0 = time.time()
stiff_model = tip_stiffness_MixedSpace(eta=eta, mode='normal')
print(f'  done in {time.time() - _t0:.1f} s')

arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info('UR5 connected')

# =============================================================================
# CSV helpers
# =============================================================================
_FIELDNAMES = (
    ['time_s', 'phase', 'k_tip_Npm'] +
    [f'q_{i}'     for i in range(13)] +
    [f'q_dot_{i}' for i in range(13)] +
    [f'tau_{i}'   for i in range(13)] +
    [f'tip_{f}_{ax}_m'   for f in FINGERTIPS for ax in 'xyz'] +
    [f'disp_{f}_{ax}_m'  for f in FINGERTIPS for ax in 'xyz'] +
    [f'force_{f}_{ax}_N' for f in FINGERTIPS for ax in 'xyz'] +
    [f'force_{f}_mag_N'  for f in FINGERTIPS]
)


def _open_csv(condition):
    if COLLECTED_DATA:
        return None, None, None
    folder = os.path.join(_HERE, 'outputs', 'dynamic_grasp')
    os.makedirs(folder, exist_ok=True)
    path   = os.path.join(folder, f'dynamic_grasp_{condition}.csv')
    f      = open(path, 'w', newline='')
    w      = csv.writer(f)
    w.writerow(_FIELDNAMES)
    controller.get_logger().info(f'Saving to: {path}')
    return path, f, w


def _close_csv(csv_file, csv_path):
    if csv_file is not None and not csv_file.closed:
        csv_file.close()
        controller.get_logger().info(f'Data saved to: {csv_path}')

# =============================================================================
# State machine
# =============================================================================
STATE_MOVING       = 0
STATE_RETURN_HOME  = 1   # hand → HOME, arm → BOTTLE_START
STATE_CONFIRM_NEXT = 2   # wait for operator confirmation
STATE_DONE         = 3

state = STATE_MOVING

# --- condition sequencing ---
_cond_idx       = 0
CONDITION       = CONDITIONS_ORDER[_cond_idx]
_experiment_start = time.time()

# --- per-trial state ---
_arm_moving     = False
_arm_resetting  = False
_hand_closed    = False
_stiffened      = False
_close_time     = None
_k_ramp_t0      = None
_move_start     = None
_home_start     = None
_log_tick       = 0
_current_k_tip  = 0.0
_countdown_said = set()
_confirm_ready  = False
_confirm_pending = False

_csv_path, _csv_file, _csv_writer = _open_csv(CONDITION)


def _set_task_stiffness(k):
    global _current_k_tip
    _current_k_tip = k
    for _f in FINGERTIPS:
        vmc_task.springs[_f].stiffness = np.full(3, k)
    vmc_task.springs['palm'].stiffness = np.full(3, k)


def _tip_pos(finger, q):
    r = FINGER_TIP_OFFSETS[finger]
    if finger == 'thumb':
        return np.array(FK_motor2thumbPos(q, 'IP', r))
    return np.array(FK_motor2fingerPos(q, finger, 'DIP', r))


def _close_hand():
    vmc_joint.thumb             = PC1_THUMB.copy()
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.spread[_f]    = np.array([PC1_SPREAD[_f]])
    vmc_joint.index_target      = PC1_INDEX.copy()
    vmc_joint.middle_target     = PC1_MIDDLE.copy()
    vmc_joint.ring_pinky_target = PC1_RING.copy()
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.stiffness[_f] = np.zeros(3)
        vmc_joint.damping[_f]   = np.full(3, B_ROT)
    _set_task_stiffness(_K_INIT_MAP[CONDITION])


def _hand_to_home_stiff():
    """Open hand and set high return stiffness."""
    _set_task_stiffness(0.0)
    vmc_joint.thumb             = HOME_THUMB.copy()
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.spread[_f]    = HOME_SPREAD[_f].copy()
    vmc_joint.index_target      = HOME_FINGER.copy()
    vmc_joint.middle_target     = HOME_FINGER.copy()
    vmc_joint.ring_pinky_target = HOME_FINGER.copy()
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.stiffness[_f] = np.full(3, K_HOME)
        vmc_joint.damping[_f]   = np.full(3, B_ROT)
    vmc_joint.stiffness['thumb'] = np.full(4, K_HOME)


def _start_transport():
    global _arm_moving, _move_start
    def _run():
        global state, _arm_moving
        arm.moveL(CLOSE_POSE.tolist(), APPROACH_SPEED, UR5_INIT_ACCELERATION)
        arm.moveL(FINAL_POSE.tolist(), _POST_SPEED,    UR5_INIT_ACCELERATION)
        state       = STATE_RETURN_HOME
        _arm_moving = False
    _arm_moving = True
    _move_start = time.time()
    threading.Thread(target=_run, daemon=True).start()


def _start_arm_reset():
    global _arm_resetting
    def _run():
        global _arm_resetting
        arm.moveL(UR5_POSE_BOTTLE_START.tolist(), UR5_INIT_SPEED, UR5_INIT_ACCELERATION)
        _arm_resetting = False
    _arm_resetting = True
    threading.Thread(target=_run, daemon=True).start()


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
    """Reset all per-trial state for the next condition."""
    global _arm_moving, _hand_closed, _stiffened, _close_time
    global _k_ramp_t0, _move_start, _home_start, _log_tick
    global _current_k_tip, _countdown_said, _confirm_ready, _confirm_pending
    global _csv_path, _csv_file, _csv_writer, CONDITION, _experiment_start
    _arm_moving     = False
    _hand_closed    = False
    _stiffened      = False
    _close_time     = None
    _k_ramp_t0      = None
    _move_start     = None
    _home_start     = None
    _log_tick       = 0
    _current_k_tip  = 0.0
    _countdown_said = set()
    _confirm_ready  = False
    _confirm_pending = False
    CONDITION = CONDITIONS_ORDER[_cond_idx]
    _experiment_start = time.time()
    _csv_path, _csv_file, _csv_writer = _open_csv(CONDITION)


# =============================================================================
# Control callback
# =============================================================================

def control_callback():
    global _hand_closed, _stiffened, _close_time, _k_ramp_t0, _log_tick
    global _countdown_said, state, _home_start, _arm_resetting
    global _cond_idx, _confirm_ready, _confirm_pending

    q     = controller.get_joint_positions()
    q_dot = controller.get_joint_velocities()

    tau_joint = vmc_joint.hand_torques(q, q_dot)
    tau_task  = vmc_task.hand_torques(q, q_dot)
    tau_vmc   = tau_joint + tau_task
    tau_comp  = grav_lim.compute_compensation_torques(
        q, q_dot, tau_vmc, recv.get_tcp_rotation_matrix())
    controller.publish_torques(tau_vmc + tau_comp)

    now = time.time()

    if state == STATE_MOVING:
        if not _arm_moving:
            controller.get_logger().info(
                f'[{CONDITION}] Starting transport …')
            _start_transport()
            return

        elapsed_move = now - _move_start

        # 5-second countdown.
        if not _hand_closed:
            secs_left = int(np.ceil(_CLOSE_TIME_S - elapsed_move))
            if 1 <= secs_left <= 5 and secs_left not in _countdown_said:
                controller.get_logger().info(f'CLOSING IN {secs_left} s …')
                _countdown_said.add(secs_left)

        if not _hand_closed and elapsed_move >= _CLOSE_TIME_S:
            _close_hand()
            _hand_closed = True
            _close_time  = now
            controller.get_logger().info(
                f'Hand closed — k_tip = {_K_INIT_MAP[CONDITION]:.0f} N/m')

        if CONDITION == 'adaptive' and _hand_closed and not _stiffened:
            if _k_ramp_t0 is None:
                if now - _close_time >= SOFT_DURATION:
                    _k_ramp_t0 = now
                    controller.get_logger().info('Adaptive: ramping stiffness …')
            else:
                alpha = min(1.0, (now - _k_ramp_t0) / K_RAMP_DURATION)
                _set_task_stiffness(K_SOFT + alpha * (K_STIFF - K_SOFT))
                if alpha >= 1.0:
                    _stiffened = True
                    controller.get_logger().info(
                        f'Stiffened — k_tip = {K_STIFF:.0f} N/m')

        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            if not _hand_closed:
                phase = 'open'
            elif CONDITION == 'adaptive' and not _stiffened:
                phase = 'soft' if _k_ramp_t0 is None else 'ramping'
            else:
                phase = 'closed'
            K_task_now = {f: _current_k_tip * np.eye(3) for f in FINGERTIPS}
            K_task_now['palm'] = _current_k_tip * np.eye(3)
            tip_vals, disp_vals, force_vals, mag_vals = [], [], [], []
            for _f in FINGERTIPS:
                pos  = _tip_pos(_f, q)
                disp = pos - D_REF[_f]
                frc  = stiff_model.tip_force(
                    _f, q, THETA_REF_DEG, D_REF, K_JOINT_DICT_MODEL, K_task_now)
                tip_vals   += [f'{v:.6f}' for v in pos]
                disp_vals  += [f'{v:.6f}' for v in disp]
                force_vals += [f'{v:.6f}' for v in frc]
                mag_vals.append(f'{float(np.linalg.norm(frc)):.6f}')
            row = ([f'{now - _experiment_start:.4f}', phase, f'{_current_k_tip:.1f}'] +
                   [f'{v:.6f}' for v in q] +
                   [f'{v:.6f}' for v in q_dot] +
                   [f'{v:.6f}' for v in tau_vmc] +
                   tip_vals + disp_vals + force_vals + mag_vals)
            _csv_writer.writerow(row)

    elif state == STATE_RETURN_HOME:
        if _home_start is None:
            _hand_to_home_stiff()
            _start_arm_reset()
            _home_start = now
            controller.get_logger().info(
                f'[{CONDITION}] Lift done — hand returning to home, arm resetting …')
        elif now - _home_start >= HOME_DURATION and not _arm_resetting:
            _close_csv(_csv_file, _csv_path)
            if _cond_idx + 1 < len(CONDITIONS_ORDER):
                state = STATE_CONFIRM_NEXT
            else:
                state = STATE_DONE
                controller.get_logger().info('All conditions complete.')

    elif state == STATE_CONFIRM_NEXT:
        if not _confirm_pending:
            next_cond = CONDITIONS_ORDER[_cond_idx + 1]
            k_next    = _K_INIT_MAP[next_cond]
            _ask_confirm_async(
                f'\n[Confirm] Press ENTER to start next condition: '
                f'{next_cond} (k_tip = {k_next:.0f} N/m) …\n')
        elif _confirm_ready:
            _cond_idx += 1
            _reset_trial()
            state = STATE_MOVING
            controller.get_logger().info(
                f'Starting condition {_cond_idx + 1}/{len(CONDITIONS_ORDER)}: '
                f'{CONDITION}')

    elif state == STATE_DONE:
        pass


# =============================================================================
# Run
# =============================================================================
arm.moveL(UR5_POSE_BOTTLE_START.tolist(), UR5_INIT_SPEED, UR5_INIT_ACCELERATION)
controller.get_logger().info(
    f'Dynamic grasp | order: {" → ".join(CONDITIONS_ORDER)} | '
    f'K_SOFT = {K_SOFT} N/m | K_STIFF = {K_STIFF} N/m | '
    f'close at X+{CLOSE_DISTANCE:.2f} m | speed = {APPROACH_SPEED:.3f} m/s')

input(f'\nPress ENTER to start first condition: {CONDITIONS_ORDER[0]} …\n')

controller.create_timer(1.0 / CONTROL_FREQUENCY, control_callback)

try:
    while rclpy.ok() and state != STATE_DONE:
        rclpy.spin_once(controller, timeout_sec=0.01)
except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
    _close_csv(_csv_file, _csv_path)
finally:
    arm.stopScript()

    vmc_joint.set_stiffness(0.0)
    vmc_joint.set_damping(0.0)
    vmc_task.set_stiffness(0.0)
    vmc_task.set_damping(0.0)
    controller.publish_torques(np.zeros(13))
    controller.get_logger().info('Stiffness zeroed (safe shutdown).')

    arm.disconnect()
    recv.disconnect()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

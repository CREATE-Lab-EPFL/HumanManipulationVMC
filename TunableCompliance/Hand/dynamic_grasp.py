"""
Tunable compliance — dynamic grasping of a bottle (ADAPT Hand).

The UR5 moves horizontally along +X from UR5_POSE_BOTTLE_START.  At
CLOSE_DISTANCE the hand targets instantly shift from home (all zeros) to
the PC1 pose.  Three conditions, selected at startup:

  'soft'     — K_TIP = K_SOFT throughout  (hand barely grips)
  'stiff'    — K_TIP = K_STIFF throughout  (hand grips firmly)
  'adaptive' — K_TIP = K_SOFT at close; switches to K_STIFF after SOFT_DURATION

After the UR5 has traveled TOTAL_DISTANCE the experiment ends, then the
hand returns to home and the UR5 drives back to UR5_POSE_BOTTLE_START.

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
from ModelIDHand.hand_params    import FINGER_TIP_OFFSETS
from UR5_codes.UR5_config       import UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCELERATION
from UR5_codes.UR5_readPose     import UR5Receiver
from hand_config import (
    UR5_POSE_BOTTLE_START,
    PC1_WRIST, PC1_THUMB, PC1_SPREAD, PC1_INDEX, PC1_MIDDLE, PC1_RING, PC1_PINKY,
    HOME_WRIST, HOME_THUMB, HOME_SPREAD, HOME_FINGER,
    FINGERTIPS, CONDITIONS,
    K_SOFT, K_STIFF, SOFT_DURATION, K_RAMP_DURATION,
    K_ROT, K_ROT_FLEX, B_ROT, B_TIP, B_FLEX_DAMP,
    APPROACH_SPEED, TOTAL_DISTANCE, CLOSE_DISTANCE,
)
import rtde_control

# =============================================================================
# Collected data
# =============================================================================
COLLECTED_DATA  = False

LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 30))

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

_K_INIT = {'soft': K_SOFT, 'stiff': K_STIFF, 'adaptive': K_SOFT}
K_INIT = _K_INIT[CONDITION]

# =============================================================================
# UR5 transport endpoint
# =============================================================================
TRANSPORT_END = UR5_POSE_BOTTLE_START + np.array([TOTAL_DISTANCE, 0.0, 0.0, 0.0, 0.0, 0.0])

# Absolute time from transport start at which the hand will close
_CLOSE_TIME_S = CLOSE_DISTANCE / APPROACH_SPEED

# =============================================================================
# FK targets for task spring
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

# =============================================================================
# ROS2 + controller
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

vmc_task.springs['palm'].stiffness = np.zeros(3)
vmc_task.dampers['palm'].damping   = np.full(3, B_TIP)
vmc_task.targets['palm']           = D_REF['palm'].copy()

grav_lim = GravFricLim()
recv     = UR5Receiver()

arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info('UR5 connected')

# =============================================================================
# CSV
# =============================================================================

def _output_path():
    folder = os.path.join(_HERE, 'outputs', 'dynamic_grasp')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'dynamic_grasp_{CONDITION}.csv')


_FIELDNAMES = (
    ['time_s', 'phase', 'k_tip_Npm'] +
    [f'q_{i}'     for i in range(15)] +
    [f'q_dot_{i}' for i in range(15)] +
    [f'tau_{i}'   for i in range(15)]
)

_csv_path   = None
_csv_file   = None
_csv_writer = None
if not COLLECTED_DATA:
    _csv_path   = _output_path()
    _csv_file   = open(_csv_path, 'w', newline='')
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(_FIELDNAMES)
    controller.get_logger().info(f'Saving to: {_csv_path}')
else:
    controller.get_logger().info('Data collection disabled (COLLECTED_DATA = True).')

# =============================================================================
# State machine
# =============================================================================
STATE_MOVING      = 0
STATE_RETURN_HOME = 1
STATE_DONE        = 2

state             = STATE_MOVING
_arm_moving       = False
_hand_closed      = False
_stiffened        = False   # adaptive: True once K ramp completes
_close_time       = None    # wall-clock time when hand closed
_k_ramp_t0        = None    # wall-clock time when K ramp started
_move_start       = None    # wall-clock time when UR5 started moving
_log_tick         = 0
_current_k_tip    = 0.0     # task spring stiffness currently applied
_countdown_said   = set()   # countdown seconds already announced
_return_started   = False   # True once the home-return sequence has been initiated


def _set_task_stiffness(k):
    global _current_k_tip
    _current_k_tip = k
    for _f in FINGERTIPS:
        vmc_task.springs[_f].stiffness = np.full(3, k)
    vmc_task.springs['palm'].stiffness = np.full(3, k)


def _close_hand():
    vmc_joint.wrist             = PC1_WRIST.copy()
    vmc_joint.thumb             = PC1_THUMB.copy()
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.spread[_f]    = np.array([PC1_SPREAD[_f]])
    vmc_joint.index_target      = PC1_INDEX.copy()
    vmc_joint.middle_target     = PC1_MIDDLE.copy()
    vmc_joint.ring_pinky_target = PC1_RING.copy()
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc_joint.stiffness[_f] = np.full(3, K_ROT_FLEX)
        vmc_joint.damping[_f]   = np.full(3, B_FLEX_DAMP)
    _set_task_stiffness(K_INIT)


def _start_transport():
    global _arm_moving, _move_start
    def _run():
        global state, _arm_moving
        arm.moveL(TRANSPORT_END.tolist(), APPROACH_SPEED, UR5_INIT_ACCELERATION)
        state       = STATE_RETURN_HOME
        _arm_moving = False
    _arm_moving = True
    _move_start = time.time()
    threading.Thread(target=_run, daemon=True).start()


def _start_return():
    global _arm_moving
    def _run():
        global state, _arm_moving
        arm.moveL(UR5_POSE_BOTTLE_START.tolist(), APPROACH_SPEED, UR5_INIT_ACCELERATION)
        state       = STATE_DONE
        _arm_moving = False
    _arm_moving = True
    threading.Thread(target=_run, daemon=True).start()

# =============================================================================
# Control callback
# =============================================================================
_experiment_start = time.time()

def control_callback():
    global _hand_closed, _stiffened, _close_time, _k_ramp_t0, _log_tick
    global _countdown_said, _return_started

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
            controller.get_logger().info('Starting transport …')
            _start_transport()
            return

        elapsed_move = now - _move_start

        # 5-second countdown before the closing point.
        if not _hand_closed:
            secs_left = int(np.ceil(_CLOSE_TIME_S - elapsed_move))
            if 1 <= secs_left <= 5 and secs_left not in _countdown_said:
                controller.get_logger().info(f'CLOSING IN {secs_left} s …')
                _countdown_said.add(secs_left)

        # Close hand when UR5 has traveled CLOSE_DISTANCE.
        if not _hand_closed and elapsed_move >= _CLOSE_TIME_S:
            _close_hand()
            _hand_closed = True
            _close_time  = now
            controller.get_logger().info(f'Hand closed — k_tip = {K_INIT:.0f} N/m')

        # Adaptive: ramp K from K_SOFT to K_STIFF after SOFT_DURATION.
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
                    controller.get_logger().info(f'Stiffened — k_tip = {K_STIFF:.0f} N/m')

        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            if not _hand_closed:
                phase = 'open'
            elif CONDITION == 'adaptive' and not _stiffened:
                phase = 'soft' if _k_ramp_t0 is None else 'ramping'
            else:
                phase = 'closed'
            row = ([f'{now - _experiment_start:.4f}', phase, f'{_current_k_tip:.1f}'] +
                   [f'{v:.6f}' for v in q] +
                   [f'{v:.6f}' for v in q_dot] +
                   [f'{v:.6f}' for v in tau_vmc])
            _csv_writer.writerow(row)

    elif state == STATE_RETURN_HOME:
        if not _return_started:
            _set_task_stiffness(0.0)
            vmc_joint.wrist             = HOME_WRIST.copy()
            vmc_joint.thumb             = HOME_THUMB.copy()
            for _f in ['index', 'middle', 'ring', 'pinky']:
                vmc_joint.spread[_f]    = np.array([HOME_SPREAD[_f]])
            vmc_joint.index_target      = HOME_FINGER.copy()
            vmc_joint.middle_target     = HOME_FINGER.copy()
            vmc_joint.ring_pinky_target = HOME_FINGER.copy()
            for _f in ['index', 'middle', 'ring', 'pinky']:
                vmc_joint.stiffness[_f] = np.full(3, K_ROT)
                vmc_joint.damping[_f]   = np.full(3, B_ROT)
            _return_started = True
            controller.get_logger().info('Experiment done — returning hand to home and UR5 to start …')
            _start_return()

    elif state == STATE_DONE:
        pass

# =============================================================================
# Run
# =============================================================================
arm.moveL(UR5_POSE_BOTTLE_START.tolist(), UR5_INIT_SPEED, UR5_INIT_ACCELERATION)
controller.get_logger().info(
    f'Dynamic grasp | condition: {CONDITION} | '
    f'K_SOFT = {K_SOFT} N/m | K_STIFF = {K_STIFF} N/m | '
    f'close at X+{CLOSE_DISTANCE:.2f} m | speed = {APPROACH_SPEED:.3f} m/s')

input('\nPress ENTER to start …\n')

controller.create_timer(1.0 / CONTROL_FREQUENCY, control_callback)

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

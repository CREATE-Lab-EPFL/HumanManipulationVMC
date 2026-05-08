"""
Supplementary — sigmoidal and polynomial stiffness sweep (finger, task space).

A single task-space spring is attached to the fingertip (DIP). Its stiffness
follows a displacement-dependent law updated each control cycle from the UR5
vertical displacement d. MCP and PIP springs are disabled (stiffness = 0).

The DIP target position is computed via FK_DIP(joint_to_motor(FINGER_TARGET)),
i.e., where the fingertip would be if the joints were at FINGER_TARGET.

Select experiment type via EXPERIMENT_MODE:

  EXPERIMENT_MODE = 'sigmoid'
  ─────────────────────────────────────────────────────────────────────────────
    K(d) = K_max − (K_max − K_min) / (1 + exp(−α·(d − d_th)))

    Transitions from K_max (firm approach) to K_min (soft deep contact) centred
    at d_th = D_TH. Sweep parameter: α [m⁻¹] from ALPHA_SWEEP.
      α = small : gradual transition spanning a wide displacement window
      α = large : sharp transition concentrated near D_TH

  EXPERIMENT_MODE = 'polynomial'
  ─────────────────────────────────────────────────────────────────────────────
    K(d) = K_base · (d / d_norm)^n   [clamped to K_min for stability]

    Stiffness grows as a power of displacement normalised by d_norm = D_NORM.
    At d = D_NORM: K = K_base. Sweep parameter: order n from ORDER_SWEEP.
      n = small : slower growth (linear)
      n = large : faster growth (superlinear)

Protocol (per run):
  1. UR5 moves to UR5_POSE; finger lifted (task-space target at FINGER_STRAIGHT).
  2. Wait SETTLE_TIME for finger to lift.
  3. Set run parameter, switch to contact target (FINGER_TARGET), set initial
      DIP stiffness for the chosen law; wait SETTLE_TIME.
  4. UR5 descends UR5_DESCENT — DIP stiffness updates from d each cycle; record.
  5. UR5 returns to UR5_POSE — continue K(d); record.
  6. Repeat for all (param, run) combinations.

Outputs:
  Supplementary/outputs/{sigmoid|polynomial}_stiffness/{param}/{param}_run_N.csv
"""

import numpy as np
import rclpy
from std_msgs.msg import Float64
import sys
import os
import time
import csv
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from VMCFinger.FingerController import FingerController, CONTROL_FREQUENCY
from VMCFinger.FingerVMCTaskSpace import VMC
from VMCFinger.FingerGravFricLim import GravFricLim
from KinematicsFinger.FK_Finger import FK_DIP, joint_to_motor
from ModelIDFinger.finger_params import c
from UR5_codes.UR5_config import (
    UR5_POSE, UR5_IP,
    UR5_INIT_SPEED, UR5_INIT_ACCELERATION,
    UR5_DESCENT, UR5_DESCENT_SPEED,
    FINGER_TARGET, FINGER_STRAIGHT,
    N_RUNS,
)

import rtde_control
import rtde_receive

# ── Experiment mode ───────────────────────────────────────────────────────────
EXPERIMENT_MODE = 'polynomial'   # 'sigmoid' | 'polynomial'

# ── Spring parameters ─────────────────────────────────────────────────────────
K_MIN      = 10.0    # [N/m] minimum / floor stiffness for DIP
K_MAX      = 100.0   # [N/m] maximum stiffness (sigmoid only)
K_BASE     = K_MIN   # [N/m] polynomial stiffness at d = D_NORM
K_APPROACH = 150.0   # [N/m] DIP stiffness during contact search (not recorded)

DAMPING = 2.0   # [N·s/m] task-space damping

# Sigmoid parameters
D_TH        = 0.010          # [m] transition centre displacement
ALPHA_SWEEP = [0, 100, 1000]    # [m⁻¹] transition steepness

# Polynomial parameters
D_NORM      = 0.010          # [m] normalisation displacement
ORDER_SWEEP = [0, 1, 2]         # polynomial orders

# Set to True once data is collected — runs the protocol without saving files.
COLLECTED_DATA = True

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME = 3.0   # [s] settle wait (lift + contact settle)

# ── Experiment queue ──────────────────────────────────────────────────────────
_sweep = ALPHA_SWEEP if EXPERIMENT_MODE == 'sigmoid' else ORDER_SWEEP
experiment_queue  = [(p, run) for p in _sweep for run in range(N_RUNS)]
total_experiments = len(experiment_queue)

# ── ROS2 / finger init ────────────────────────────────────────────────────────
rclpy.init()
controller    = FingerController()
grav_fric_lim = GravFricLim()

controller.create_subscription(
    Float64, '/force_normal',
    lambda msg: setattr(controller, 'force_N', msg.data * 0.00980665), 10)
controller.force_N = 0.0

# Compute DIP target positions from joint angles
q_ref_motor_rad  = joint_to_motor(FINGER_TARGET)
target_DIP       = FK_DIP(q_ref_motor_rad,  np.array([0.0, c, 0.0]))
q_lift_motor_rad = joint_to_motor(FINGER_STRAIGHT)
target_DIP_lift  = FK_DIP(q_lift_motor_rad, np.array([0.0, c, 0.0]))

# Spring attachment point: 10 cm below the lifted fingertip (Z points downward)
target_DIP_spring = target_DIP_lift + np.array([0.0, 0.0, 0.10])

vmc = VMC(
    stiffness  = np.array([0.0, 0.0, 0.0]),
    damping    = np.array([DAMPING, DAMPING, DAMPING]),
    target_DIP = target_DIP_spring,
)
# Only DIP spring is active
vmc.springs['DIP'].stiffness = K_APPROACH

# ── UR5 init ──────────────────────────────────────────────────────────────────
arm  = rtde_control.RTDEControlInterface(UR5_IP)
recv = rtde_receive.RTDEReceiveInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info('UR5 connected')

# ── State machine ─────────────────────────────────────────────────────────────
STATE_INIT_ARM = 0
STATE_LIFT     = 1
STATE_SETTLE   = 2
STATE_DESCEND  = 3
STATE_ASCEND   = 4
STATE_NEXT     = 5
STATE_DONE     = 6

state             = STATE_INIT_ARM
state_start_time  = time.time()
current_exp_idx   = 0
arm_moving        = False

experiment_start_time = None
csv_file              = None
csv_writer            = None
csv_filename          = None
current_param         = _sweep[0] if _sweep else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def sigmoid_k(alpha, d):
    """K(d) = K_max - (K_max - K_min) / (1 + exp(-alpha*(d - D_TH)))"""
    return K_MAX - (K_MAX - K_MIN) / (1.0 + np.exp(-alpha * (d - D_TH)))

def polynomial_k(order, d):
    """K(d) = K_base * (d/D_NORM)^order"""
    return max(K_MIN, K_BASE * (max(d, 0.0) / D_NORM) ** order)

def current_k(param, d):
    return sigmoid_k(param, d) if EXPERIMENT_MODE == 'sigmoid' else polynomial_k(param, d)

def param_label(param):
    return f'alpha_{param}' if EXPERIMENT_MODE == 'sigmoid' else f'order_{param}'

def set_lift_profile():
    """Lifted target and approach stiffness."""
    vmc.targets['DIP'] = target_DIP_spring
    vmc.springs['DIP'].stiffness = K_APPROACH

def set_contact_profile(param):
    """Contact target and initial run stiffness before motion."""
    vmc.targets['DIP'] = target_DIP_spring
    if EXPERIMENT_MODE == 'sigmoid':
        vmc.springs['DIP'].stiffness = K_MAX
    else:
        vmc.springs['DIP'].stiffness = polynomial_k(param, 0.0)

def output_path(param, run):
    folder = os.path.join(
        os.path.dirname(__file__), 'outputs',
        f'{EXPERIMENT_MODE}_stiffness', param_label(param))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{param_label(param)}_run_{run + 1}.csv')

def open_csv(param, run):
    fname = output_path(param, run)
    f = open(fname, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        'Time_s',
        'Phase',
        'Force_N',
        'Motor1_pos_deg', 'Motor2_pos_deg',
        'Motor1_vel_degs', 'Motor2_vel_degs',
        'Motor1_torque_Nm', 'Motor2_torque_Nm',
        'UR5_Z_m',
        'UR5_displacement_m',
        'K_task_Nm',
        'Param',
        'Mode',
    ])
    return f, w, fname

def move_arm_async(target_pose, speed, done_state):
    global state, arm_moving, state_start_time
    def _run():
        global state, arm_moving, state_start_time
        arm.moveL(target_pose.tolist(), speed, UR5_INIT_ACCELERATION)
        state_start_time = time.time()
        state = done_state
        arm_moving = False
    arm_moving = True
    threading.Thread(target=_run, daemon=True).start()


# ── Control callback ──────────────────────────────────────────────────────────

def control_callback():
    global state, state_start_time, current_exp_idx, arm_moving
    global experiment_start_time, current_param
    global csv_file, csv_writer, csv_filename

    q_deg     = controller.get_joint_positions()
    q_dot_deg = controller.get_joint_velocities()
    q_motor   = np.radians(q_deg)
    q_dot_rad = np.radians(q_dot_deg)

    # ── Compute displacement and update DIP stiffness (motion phases) ────────
    disp = None
    ur5_z = None
    if state in (STATE_DESCEND, STATE_ASCEND):
        ur5_z = recv.getActualTCPPose()[2]
        disp = UR5_POSE[2] - ur5_z
        k = current_k(current_param, disp)
        vmc.springs['DIP'].stiffness = k

    tau_vmc   = vmc.finger_torques(q_motor, q_dot_rad)
    tau_comp  = grav_fric_lim.compute_compensation_torques(q_motor, q_dot_rad, tau_vmc)
    tau_total = tau_vmc + tau_comp
    controller.publish_torques(tau_total)

    # ── Recording (descent and ascent) ────────────────────────────────────────
    if not COLLECTED_DATA and state in (STATE_DESCEND, STATE_ASCEND) and experiment_start_time is not None and disp is not None:
        phase = 'descent' if state == STATE_DESCEND else 'ascent'
        elapsed = time.time() - experiment_start_time
        csv_writer.writerow([
            f'{elapsed:.4f}',
            phase,
            f'{controller.force_N:.4f}',
            f'{q_deg[0]:.4f}',     f'{q_deg[1]:.4f}',
            f'{q_dot_deg[0]:.4f}', f'{q_dot_deg[1]:.4f}',
            f'{tau_total[0]:.6f}', f'{tau_total[1]:.6f}',
            f'{ur5_z:.6f}',
            f'{disp:.6f}',
            f'{vmc.springs["DIP"].stiffness:.6f}',
            current_param,
            EXPERIMENT_MODE,
        ])

    # ── State transitions ─────────────────────────────────────────────────────

    if state == STATE_INIT_ARM:
        if not arm_moving:
            controller.get_logger().info('Moving UR5 to UR5_POSE ...')
            move_arm_async(UR5_POSE, UR5_INIT_SPEED, STATE_LIFT)

    elif state == STATE_LIFT:
        set_lift_profile()
        if time.time() - state_start_time >= SETTLE_TIME:
            param, run = experiment_queue[current_exp_idx]
            current_param = param
            set_contact_profile(param)
            controller.get_logger().info(
                f'=== Experiment {current_exp_idx + 1}/{total_experiments}: '
                f'{EXPERIMENT_MODE} {param_label(param)}, run {run + 1}/{N_RUNS} — settling into contact ...')
            state = STATE_SETTLE
            state_start_time = time.time()

    elif state == STATE_SETTLE:
        if time.time() - state_start_time >= SETTLE_TIME:
            param, run = experiment_queue[current_exp_idx]
            if not COLLECTED_DATA:
                csv_file, csv_writer, csv_filename = open_csv(param, run)
                controller.get_logger().info(f'Saving to: {csv_filename}')
            experiment_start_time = time.time()
            descent_target = UR5_POSE.copy()
            descent_target[2] -= UR5_DESCENT
            controller.get_logger().info('Descending ...')
            move_arm_async(descent_target, UR5_DESCENT_SPEED, STATE_ASCEND)
            state = STATE_DESCEND

    elif state == STATE_DESCEND:
        pass

    elif state == STATE_ASCEND:
        if not arm_moving:
            move_arm_async(UR5_POSE, UR5_DESCENT_SPEED, STATE_NEXT)

    elif state == STATE_NEXT:
        if csv_file and not csv_file.closed:
            csv_file.close()
            controller.get_logger().info(f'Data saved: {csv_filename}')
        current_exp_idx += 1
        if current_exp_idx >= total_experiments:
            controller.get_logger().info('=== ALL EXPERIMENTS COMPLETE ===')
            state = STATE_DONE
        else:
            state = STATE_LIFT
            state_start_time = time.time()

    elif state == STATE_DONE:
        pass


# ── Start ─────────────────────────────────────────────────────────────────────

controller.create_timer(1.0 / CONTROL_FREQUENCY, control_callback)
controller.get_logger().info(
    f'{EXPERIMENT_MODE.capitalize()} stiffness: {len(_sweep)} params × {N_RUNS} runs = '
    f'{total_experiments} experiments')

try:
    rclpy.spin(controller)
except KeyboardInterrupt:
    controller.get_logger().info('Interrupted')
finally:
    if csv_file is not None and not csv_file.closed:
        csv_file.close()
    arm.stopScript()
    controller.destroy_node()
    rclpy.shutdown()

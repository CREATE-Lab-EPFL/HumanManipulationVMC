"""
Passive compliance shaping - linear cart stiffness range (finger, vertical cart).

The finger presses against a surface at UR5_POSE while cart stiffness K_cart is
sampled from 40 to 300 N/m using the same biased spacing strategy as
passive_range.py (denser near K_MIN), vertical direction only (0 deg, z-axis),
with N_RUNS repetitions per stiffness.
UR5 descends UR5_DESCENT from UR5_POSE and returns, recording force and
position throughout both descent and ascent.

Protocol (per K_cart):
  1. Finger lifted to FINGER_STRAIGHT; wait SETTLE_TIME.
  2. Set K_cart and target FINGER_TARGET; wait SETTLE_TIME.
  3. UR5 descends UR5_DESCENT — record all signals (Phase='descent').
  4. UR5 returns to UR5_POSE — record all signals (Phase='ascent').
  5. Repeat for all K_cart values and runs.

Outputs: PassiveCompliance/Finger/outputs/stiffness_range_linear/K_XX.XX_run_N.csv
"""

import numpy as np
import rclpy
from std_msgs.msg import Float64
import sys
import os
import time
import csv
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from VMCFinger.FingerController import FingerController, CONTROL_FREQUENCY
from VMCFinger.FingerVMCDirCart import VMC
from VMCFinger.FingerGravFricLim import GravFricLim
from VMC_utils.VirtualModels import LinearSpring, ConstrainedLinearSpring, ConstrainedLinearDamper
from UR5_codes.UR5_config import (
    UR5_POSE, UR5_IP,
    UR5_INIT_SPEED, UR5_INIT_ACCELERATION,
    UR5_DESCENT, UR5_DESCENT_SPEED,
    FINGER_TARGET, FINGER_STRAIGHT,
    BENDING_DAMPING,
    N_RUNS,
)

import rtde_control
import rtde_receive

# ── Experiment parameters ─────────────────────────────────────────────────────
K_MIN = 5.0
K_MAX = 280.0
K_SAMPLES = 10
K_DISTRIBUTION_POWER = 2.0
K_RANGE = list(dict.fromkeys([
    round(float(k), 2)
    for k in (
        K_MIN + (K_MAX - K_MIN) * (np.linspace(0.0, 1.0, K_SAMPLES) ** K_DISTRIBUTION_POWER)
    )
]))
CART_DAMPING = 3.0
CART_OFFSET = 0.10
EXPERIMENT_JOINT_STIFFNESS = 0.0

# Set to True once data is collected — runs the protocol without saving files.
COLLECTED_DATA = True

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME = 3.0

# ── Experiment queue ──────────────────────────────────────────────────────────
experiment_queue  = [(k, run) for k in K_RANGE for run in range(N_RUNS)]
total_experiments = len(experiment_queue)

# ── ROS2 / finger init ────────────────────────────────────────────────────────
rclpy.init()
controller    = FingerController()
grav_fric_lim = GravFricLim()

controller.create_subscription(
    Float64, '/force_normal',
    lambda msg: setattr(controller, 'force_N', msg.data * 0.00980665), 10)
controller.force_N = 0.0

controller.create_subscription(
    Float64, '/force_shear',
    lambda msg: setattr(controller, 'force_shear_N', msg.data * 0.00980665), 10)
controller.force_shear_N = 0.0

vmc = VMC(
    stiffness      = np.array([0.5] * 3),
    damping        = np.array([BENDING_DAMPING] * 3),
    cart_stiffness = 0.0,
    cart_damping   = 0.0,
    target         = FINGER_STRAIGHT,
)
vmc.cart_offset = CART_OFFSET

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def n_vertical():
    return np.array([[0.0], [0.0], [1.0]])


def set_range_stiffness(k_cart):
    n = n_vertical()
    vmc.spring = LinearSpring(np.array([EXPERIMENT_JOINT_STIFFNESS] * 3))
    vmc.cart_spring = ConstrainedLinearSpring(k_cart, n)
    vmc.cart_damper = ConstrainedLinearDamper(CART_DAMPING, n)
    vmc.target_cart = vmc._target_cart_base + vmc.cart_offset * n.flatten()


def output_path(k, run):
    folder = os.path.join(os.path.dirname(__file__), 'outputs', 'stiffness_range_linear')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'K_{k:.2f}_run_{run + 1}.csv')


def open_csv(k, run):
    fname = output_path(k, run)
    f = open(fname, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        'Time_s',
        'Phase',
        'Force_normal_N', 'Force_shear_N',
        'Motor1_pos_deg', 'Motor2_pos_deg',
        'Motor1_vel_degs', 'Motor2_vel_degs',
        'Motor1_torque_Nm', 'Motor2_torque_Nm',
        'UR5_Z_m', 'UR5_displacement_m',
        'CartStiffness_Npm',
        'Run',
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
    global experiment_start_time
    global csv_file, csv_writer, csv_filename

    q_deg     = controller.get_joint_positions()
    q_dot_deg = controller.get_joint_velocities()
    q_motor   = np.radians(q_deg)
    q_dot_rad = np.radians(q_dot_deg)

    tau_vmc   = vmc.finger_torques(q_motor, q_dot_rad)
    tau_comp  = grav_fric_lim.compute_compensation_torques(q_motor, q_dot_rad, tau_vmc)
    tau_total = tau_vmc + tau_comp
    controller.publish_torques(tau_total)

    if not COLLECTED_DATA and state in (STATE_DESCEND, STATE_ASCEND) and experiment_start_time is not None:
        phase   = 'descent' if state == STATE_DESCEND else 'ascent'
        elapsed = time.time() - experiment_start_time
        ur5_z   = recv.getActualTCPPose()[2]
        disp    = UR5_POSE[2] - ur5_z
        k_cur, run_cur = experiment_queue[current_exp_idx]
        csv_writer.writerow([
            f'{elapsed:.4f}',
            phase,
            f'{controller.force_N:.4f}',
            f'{controller.force_shear_N:.4f}',
            f'{q_deg[0]:.4f}',     f'{q_deg[1]:.4f}',
            f'{q_dot_deg[0]:.4f}', f'{q_dot_deg[1]:.4f}',
            f'{tau_total[0]:.6f}', f'{tau_total[1]:.6f}',
            f'{ur5_z:.6f}',        f'{disp:.6f}',
            f'{k_cur:.2f}',
            run_cur + 1,
        ])

    if state == STATE_INIT_ARM:
        if not arm_moving:
            controller.get_logger().info('Moving UR5 to UR5_POSE ...')
            move_arm_async(UR5_POSE, UR5_INIT_SPEED, STATE_LIFT)

    elif state == STATE_LIFT:
        vmc.spring = LinearSpring(np.array([0.5] * 3))
        vmc.cart_spring = ConstrainedLinearSpring(0.0, n_vertical())
        vmc.cart_damper = ConstrainedLinearDamper(0.0, n_vertical())
        vmc.target = FINGER_STRAIGHT
        if time.time() - state_start_time >= SETTLE_TIME:
            k_cur, run = experiment_queue[current_exp_idx]
            set_range_stiffness(k_cur)
            vmc.target = FINGER_TARGET
            controller.get_logger().info(
                f'=== Experiment {current_exp_idx + 1}/{total_experiments}: '
                f'K_cart={k_cur:.1f} N/m, run {run + 1}/{N_RUNS} — settling into contact ...')
            state = STATE_SETTLE
            state_start_time = time.time()

    elif state == STATE_SETTLE:
        if time.time() - state_start_time >= SETTLE_TIME:
            k_cur, run = experiment_queue[current_exp_idx]
            if not COLLECTED_DATA:
                csv_file, csv_writer, csv_filename = open_csv(k_cur, run)
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
    f'Linear cart stiffness range (vertical): K from {K_MIN:.0f} to {K_MAX:.0f} '
    f'with {K_SAMPLES} biased samples (power={K_DISTRIBUTION_POWER}), '
    f'{N_RUNS} runs/K = {total_experiments} experiments')

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

"""
Tunable compliance via VMC - repulsive stiffness shaping (finger).

A repulsive Gaussian virtual spring is superimposed at the fingertip on top of
the joint-space spring (k_d = BENDING_STIFFNESS, constant throughout). The spring
is inactive during the initial approach and activates at depth d_act = UR5_DESCENT / 2.
Once active it repulses the fingertip away from its rest position with a force
that rises then decays with displacement:

    F_Z(Δz) = strength · exp(-Δz² / 2σ²) · Δz

where Δz = z_tip - z_target [m] grows as the finger is pressed.

Protocol (per run):
  1. UR5 moves to UR5_POSE; finger lifted to FINGER_STRAIGHT.
  2. Wait SETTLE_TIME for finger to lift.
  3. Set run strength, keep base spring at BENDING_STIFFNESS,
      set target FINGER_TARGET; wait SETTLE_TIME.
  4. UR5 descends UR5_DESCENT; Gaussian activates at d = UR5_DESCENT/2; record.
  5. UR5 returns to UR5_POSE; Gaussian deactivates at half-ascent; record.
  6. Repeat for all (strength, run) combinations.

Outputs:
  TunableCompliance/Finger/outputs/repulsive_stiffness_shaping/strength_{s}/strength_{s}_run_N.csv
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
from VMCFinger.FingerVMCRepulsiveSpring import VMC
from KinematicsFinger.FK_Finger import joint_to_motor, FK_DIP
from ModelIDFinger.finger_params import c as FINGER_TIP_OFFSET
from VMCFinger.FingerGravFricLim import GravFricLim
from VMC_utils.VirtualModels import LinearSpring
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
BENDING_STIFFNESS = 0.5         # [N·m/rad] local bending stiffness baseline
SIGMA            = 0.010              # [m] Gaussian width - peak force at Δz = SIGMA
ACTIVATION_DEPTH = UR5_DESCENT / 2   # [m] UR5 displacement at which Gaussian activates

STRENGTH_SWEEP = [0, 300, 600]         # [N/m] cart_strength values to sweep

# Set to True once data is collected — runs the protocol without saving files.
COLLECTED_DATA = True

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME = 3.0   # [s] settle wait (used twice per run: finger lift + contact bend)

# ── Experiment queue ──────────────────────────────────────────────────────────
experiment_queue  = [(s, run) for s in STRENGTH_SWEEP for run in range(N_RUNS)]
total_experiments = len(experiment_queue)

# ── ROS2 / finger init ────────────────────────────────────────────────────────
rclpy.init()
controller    = FingerController()
grav_fric_lim = GravFricLim()

controller.create_subscription(
    Float64, '/force_normal',
    lambda msg: setattr(controller, 'force_N', msg.data * 0.00980665), 10)
controller.force_N = 0.0

vmc = VMC(
    stiffness     = np.array([BENDING_STIFFNESS]       * 3),
    damping       = np.array([4 * BENDING_DAMPING]     * 3),
    cart_strength = 0.0,
    cart_sigma    = SIGMA,
    cart_damping  = 4 * BENDING_DAMPING,
    target        = FINGER_STRAIGHT,
)

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
gaussian_activated    = False
gaussian_deactivated_on_ascent = False
current_strength      = STRENGTH_SWEEP[0] if STRENGTH_SWEEP else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def reset_gaussian():
    """Zero the Gaussian spring strength before each descent."""
    vmc.cart_spring.strength = 0.0

def set_bending_stiffness():
    vmc.spring = LinearSpring(np.array([BENDING_STIFFNESS] * 3))
    vmc.cart_spring.strength = 0.0

def output_path(strength, run):
    s_str  = str(strength)
    folder = os.path.join(
        os.path.dirname(__file__), 'outputs', 'repulsive_stiffness_shaping',
        f'strength_{s_str}')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'strength_{s_str}_run_{run + 1}.csv')

def open_csv(strength, run):
    fname = output_path(strength, run)
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
        'CartStrength_N_per_m',
        'Sigma_m',
        'Strength',
    ])
    return f, w, fname

def move_arm_async(target_pose, speed, done_state):
    """Move UR5 in a background thread; set state when done."""
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
    global experiment_start_time, current_strength
    global csv_file, csv_writer, csv_filename
    global gaussian_activated, gaussian_deactivated_on_ascent

    q_deg     = controller.get_joint_positions()
    q_dot_deg = controller.get_joint_velocities()
    q_motor   = np.radians(q_deg)
    q_dot_rad = np.radians(q_dot_deg)

    # ── Compute UR5 displacement during motion phases ─────────────────────────
    disp = None
    ur5_z = None
    if state in (STATE_DESCEND, STATE_ASCEND):
        ur5_z = recv.getActualTCPPose()[2]
        disp = UR5_POSE[2] - ur5_z

    # ── Gaussian activation ────────────────────────────────────────────────────
    if state == STATE_DESCEND and not gaussian_activated and disp is not None:
        if disp >= ACTIVATION_DEPTH:
            vmc.cart_spring.strength = float(current_strength)
            gaussian_activated = True
            controller.get_logger().info(
                f'Gaussian activated: strength={current_strength} N/m '
                f'at disp={disp * 1e3:.1f} mm')

    # ── Gaussian deactivation (half-ascent) ──────────────────────────────────
    if (
        state == STATE_ASCEND
        and arm_moving
        and gaussian_activated
        and not gaussian_deactivated_on_ascent
        and disp is not None
    ):
        if disp <= (UR5_DESCENT / 2):
            vmc.cart_spring.strength = 0.0
            gaussian_deactivated_on_ascent = True
            controller.get_logger().info('Gaussian deactivated at half-ascent')

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
            f'{vmc.cart_spring.strength:.4f}',
            f'{SIGMA:.4f}',
            current_strength,
        ])

    # ── State transitions ─────────────────────────────────────────────────────

    if state == STATE_INIT_ARM:
        if not arm_moving:
            controller.get_logger().info('Moving UR5 to UR5_POSE ...')
            move_arm_async(UR5_POSE, UR5_INIT_SPEED, STATE_LIFT)

    elif state == STATE_LIFT:
        vmc.target = FINGER_STRAIGHT
        if time.time() - state_start_time >= SETTLE_TIME:
            strength, run = experiment_queue[current_exp_idx]
            current_strength = strength
            set_bending_stiffness()
            vmc.target = FINGER_TARGET
            vmc.target_cart = FK_DIP(joint_to_motor(FINGER_TARGET), np.array([0, FINGER_TIP_OFFSET, 0]))
            controller.get_logger().info(
                f'=== Experiment {current_exp_idx + 1}/{total_experiments}: '
                f'strength={strength} N/m, run {run + 1}/{N_RUNS} — settling into contact ...')
            state = STATE_SETTLE
            state_start_time = time.time()

    elif state == STATE_SETTLE:
        if time.time() - state_start_time >= SETTLE_TIME:
            strength, run = experiment_queue[current_exp_idx]
            if not COLLECTED_DATA:
                csv_file, csv_writer, csv_filename = open_csv(strength, run)
                controller.get_logger().info(f'Saving to: {csv_filename}')
            experiment_start_time = time.time()
            gaussian_activated    = False
            gaussian_deactivated_on_ascent = False
            vmc.cart_spring.strength = 0.0
            descent_target = UR5_POSE.copy()
            descent_target[2] -= UR5_DESCENT
            controller.get_logger().info('Descending ...')
            move_arm_async(descent_target, UR5_DESCENT_SPEED, STATE_ASCEND)
            state = STATE_DESCEND

    elif state == STATE_DESCEND:
        pass  # activation and recording handled above

    elif state == STATE_ASCEND:
        if not arm_moving:
            move_arm_async(UR5_POSE, UR5_DESCENT_SPEED, STATE_NEXT)

    elif state == STATE_NEXT:
        if csv_file and not csv_file.closed:
            csv_file.close()
            controller.get_logger().info(f'Data saved: {csv_filename}')
        set_bending_stiffness()
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
    f'Repulsive stiffness shaping: {len(STRENGTH_SWEEP)} strengths × {N_RUNS} runs = '
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

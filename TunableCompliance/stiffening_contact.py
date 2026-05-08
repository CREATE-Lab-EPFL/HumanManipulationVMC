"""
Tunable compliance via VMC - stiffening on contact (finger).

The protocol mirrors passive_stiffness_sweep: move to UR5_POSE, lift finger,
settle, set run profile and target, settle, descend, ascend, repeat.
The only behavioral difference is the online stiffness update:

    K(F) = K_min + (K_max - K_min) · (1 - exp(-alpha · F))

where F is the measured normal contact force [N].

  - alpha = 0       : K = K_min throughout  (no stiffening - passive baseline)
  - alpha = small   : gradual stiffening  (F_half = ln(2)/alpha)
  - alpha = large   : near-immediate stiffening at first contact

Alpha values in ALPHA_SWEEP are spaced so that F_half = ln(2)/alpha spans
the expected contact-force range logarithmically.

Protocol (per run):
  1. UR5 moves to UR5_POSE; finger lifted to FINGER_STRAIGHT.
  2. Wait SETTLE_TIME for finger to lift.
  3. Set run alpha, initialise K_d to K_min, set target FINGER_TARGET;
      wait SETTLE_TIME.
  4. UR5 descends UR5_DESCENT - update K_d online from measured force; record.
  5. UR5 returns to UR5_POSE - continue K_d = f(F); record.
  6. Repeat for all (alpha, run) combinations.

Outputs:
  TunableCompliance/outputs/stiffening_contact/alpha_X/alpha_X_run_N.csv
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
from VMCFinger.FingerVMCFingerSpace import VMC
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
K_MIN = 0.10    # [N·m/rad] stiffness at zero contact force
K_MAX = 0.60    # [N·m/rad] stiffness at saturation

# alpha [1/N]
ALPHA_SWEEP = [0.0, 0.3, 2.0]

# Set to True once data is collected — runs the protocol without saving files.
COLLECTED_DATA = True

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME = 3.0   # [s] settle wait (used twice per run: finger lift + contact bend)

# ── Experiment queue ──────────────────────────────────────────────────────────
experiment_queue  = [(alpha, run) for alpha in ALPHA_SWEEP for run in range(N_RUNS)]
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
    stiffness = np.array([K_MIN] * 3),
    damping   = np.array([BENDING_DAMPING]   * 3),
    target    = FINGER_STRAIGHT,
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
STATE_SETTLE_A = 2
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
current_alpha         = ALPHA_SWEEP[0] if ALPHA_SWEEP else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def stiffening_k(alpha, force):
    """K(F) = K_min + (K_max - K_min) * (1 - exp(-alpha * F))."""
    if alpha == 0:
        return K_MIN
    return K_MIN + (K_MAX - K_MIN) * (1.0 - np.exp(-alpha * max(force, 0.0)))

def set_experiment_stiffness():
    """Initialise spring to K_min before run (same for all alphas)."""
    vmc.spring = LinearSpring(np.array([K_MIN] * 3))

def output_path(alpha, run):
    folder = os.path.join(
        os.path.dirname(__file__), 'outputs', 'stiffening_contact', f'alpha_{alpha}')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'alpha_{alpha}_run_{run + 1}.csv')

def open_csv(alpha, run):
    fname = output_path(alpha, run)
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
        'K_d_Nmrad',
        'Alpha',
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
    global experiment_start_time, current_alpha
    global csv_file, csv_writer, csv_filename

    q_deg     = controller.get_joint_positions()
    q_dot_deg = controller.get_joint_velocities()
    q_motor   = np.radians(q_deg)
    q_dot_rad = np.radians(q_dot_deg)

    # ── Stiffness update (motion phases only) ─────────────────────────────────
    if state in (STATE_DESCEND, STATE_ASCEND):
        k = stiffening_k(current_alpha, controller.force_N)
        vmc.spring.stiffness = np.array([k] * 3)

    tau_vmc   = vmc.finger_torques(q_motor, q_dot_rad)
    tau_comp  = grav_fric_lim.compute_compensation_torques(q_motor, q_dot_rad, tau_vmc)
    tau_total = tau_vmc + tau_comp
    controller.publish_torques(tau_total)

    # ── Recording (descent and ascent) ────────────────────────────────────────
    if not COLLECTED_DATA and state in (STATE_DESCEND, STATE_ASCEND) and experiment_start_time is not None:
        phase    = 'descent' if state == STATE_DESCEND else 'ascent'
        elapsed  = time.time() - experiment_start_time
        ur5_z    = recv.getActualTCPPose()[2]
        disp     = UR5_POSE[2] - ur5_z
        k_now    = vmc.spring.stiffness[0]
        csv_writer.writerow([
            f'{elapsed:.4f}',
            phase,
            f'{controller.force_N:.4f}',
            f'{q_deg[0]:.4f}',    f'{q_deg[1]:.4f}',
            f'{q_dot_deg[0]:.4f}', f'{q_dot_deg[1]:.4f}',
            f'{tau_total[0]:.6f}', f'{tau_total[1]:.6f}',
            f'{ur5_z:.6f}',
            f'{disp:.6f}',
            f'{k_now:.6f}',
            current_alpha,
        ])

    # ── State transitions ─────────────────────────────────────────────────────

    if state == STATE_INIT_ARM:
        if not arm_moving:
            controller.get_logger().info('Moving UR5 to UR5_POSE ...')
            move_arm_async(UR5_POSE, UR5_INIT_SPEED, STATE_LIFT)

    elif state == STATE_LIFT:
        vmc.target = FINGER_STRAIGHT
        if time.time() - state_start_time >= SETTLE_TIME:
            alpha, run = experiment_queue[current_exp_idx]
            current_alpha = alpha
            set_experiment_stiffness()
            vmc.target = FINGER_TARGET
            controller.get_logger().info(
                f'=== Experiment {current_exp_idx + 1}/{total_experiments}: '
                f'alpha={alpha}, run {run + 1}/{N_RUNS} — settling into contact ...')
            state = STATE_SETTLE_A
            state_start_time = time.time()

    elif state == STATE_SETTLE_A:
        if time.time() - state_start_time >= SETTLE_TIME:
            alpha, run = experiment_queue[current_exp_idx]
            if not COLLECTED_DATA:
                csv_file, csv_writer, csv_filename = open_csv(alpha, run)
                controller.get_logger().info(f'Saving to: {csv_filename}')
            experiment_start_time = time.time()
            descent_target = UR5_POSE.copy()
            descent_target[2] -= UR5_DESCENT
            controller.get_logger().info('Descending ...')
            move_arm_async(descent_target, UR5_DESCENT_SPEED, STATE_ASCEND)
            state = STATE_DESCEND

    elif state == STATE_DESCEND:
        pass  # stiffness update and recording handled above

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
    f'Stiffening contact: {len(ALPHA_SWEEP)} alpha values × {N_RUNS} runs = '
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

"""
Passive compliance shaping - directional stiffness (finger).

The finger presses against a surface while the cart spring direction n is swept
over ANGLE_SWEEP angles in the Y-Z plane using vertical pressing:
    - Vertical:  UR5 descends in Z (normal press) from UR5_POSE.

The UR5 starts already at the contact pose. No contact detection is performed.

Protocol (per run):
    1. Arm moves to start pose UR5_POSE;
     finger lifts to FINGER_STRAIGHT; wait SETTLE_TIME.
  2. Set cart stiffness at angle n; finger to FINGER_TARGET; wait SETTLE_TIME.
  3. UR5 presses UR5_DESCENT from start pose — record (Phase='descent').
  4. UR5 returns to start pose — record (Phase='ascent').
    5. Repeat for all (angle, run) combinations.

Outputs:
    PassiveCompliance/outputs/directional_stiffness/angle_{deg}/angle_{deg}_run_{n}.csv
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
BENDING_STIFFNESS           = 0.5    # [N·m/rad] local bending stiffness baseline
EXPERIMENT_JOINT_STIFFNESS = 0.0     # [N·m/rad] joint-level spring during press (cart-only)
CART_STIFFNESS             = 100.0   # [N/m]     cart spring stiffness
CART_DAMPING               = 1.0     # [N·s/m]   cart damper
CART_OFFSET                = 0.10    # [m]       cart application point offset along n

ANGLE_SWEEP  = [0, -30, -60]              # [deg] cart angles from Z in the Y-Z plane

# Set to True once data is collected — runs the protocol without saving files.
COLLECTED_DATA = True

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME = 3.0   # [s] settle wait (used twice per run: finger lift + contact bend)

# ── Experiment queue ──────────────────────────────────────────────────────────
experiment_queue = [
    (angle, run)
    for angle in ANGLE_SWEEP
    for run in range(N_RUNS)
]
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
    stiffness      = np.array([BENDING_STIFFNESS] * 3),
    damping        = np.array([BENDING_DAMPING]   * 3),
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
STATE_LIFT      = 0   # finger straight + arm to start pose
STATE_SETTLE_A  = 1   # wait SETTLE_TIME with finger straight
STATE_SETTLE_B  = 2   # wait SETTLE_TIME with finger at target (contact settle)
STATE_DESCEND   = 3
STATE_ASCEND    = 4
STATE_NEXT      = 5
STATE_DONE      = 6

state            = STATE_LIFT
state_start_time = time.time()
current_exp_idx  = 0
arm_moving       = False

experiment_start_time = None
csv_file              = None
csv_writer            = None
csv_filename          = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def n_from_angle(angle_deg):
    """Cart normal vector at angle_deg from Z in the Y-Z plane."""
    a = np.radians(angle_deg)
    return np.array([[0], [np.sin(a)], [np.cos(a)]])

def start_pose():
    return UR5_POSE

def set_bending_stiffness():
    vmc.spring      = LinearSpring(np.array([BENDING_STIFFNESS] * 3))
    vmc.cart_spring = ConstrainedLinearSpring(0.0)
    vmc.cart_damper = ConstrainedLinearDamper(0.0)

def set_experiment_stiffness(angle_deg):
    n = n_from_angle(angle_deg)
    vmc.spring      = LinearSpring(np.array([EXPERIMENT_JOINT_STIFFNESS] * 3))
    vmc.cart_spring = ConstrainedLinearSpring(CART_STIFFNESS, n)
    vmc.cart_damper = ConstrainedLinearDamper(CART_DAMPING,   n)
    vmc.target_cart = vmc._target_cart_base + vmc.cart_offset * n.flatten()

def output_path(angle_deg, run):
    folder = os.path.join(
        os.path.dirname(__file__), 'outputs', 'directional_stiffness',
        f'angle_{angle_deg}')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'angle_{angle_deg}_run_{run + 1}.csv')

def open_csv(angle_deg, run):
    fname = output_path(angle_deg, run)
    f = open(fname, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        'Time_s',
        'Phase',
        'Force_normal_N', 'Force_shear_N',
        'Motor1_pos_deg', 'Motor2_pos_deg',
        'Motor1_vel_degs', 'Motor2_vel_degs',
        'Motor1_torque_Nm', 'Motor2_torque_Nm',
        'UR5_displacement_m',
        'Angle_deg',
    ])
    return f, w, fname

def displacement(tcp_pose):
    """Vertical displacement from start pose."""
    return UR5_POSE[2] - tcp_pose[2]

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

    # ── Recording (descent and ascent) ────────────────────────────────────────
    if not COLLECTED_DATA and state in (STATE_DESCEND, STATE_ASCEND) and experiment_start_time is not None:
        angle_deg, _ = experiment_queue[current_exp_idx]
        phase    = 'descent' if state == STATE_DESCEND else 'ascent'
        elapsed  = time.time() - experiment_start_time
        tcp_pose = recv.getActualTCPPose()
        disp     = displacement(tcp_pose)
        csv_writer.writerow([
            f'{elapsed:.4f}',
            phase,
            f'{controller.force_N:.4f}',       f'{controller.force_shear_N:.4f}',
            f'{q_deg[0]:.4f}',                 f'{q_deg[1]:.4f}',
            f'{q_dot_deg[0]:.4f}',             f'{q_dot_deg[1]:.4f}',
            f'{tau_total[0]:.6f}',             f'{tau_total[1]:.6f}',
            f'{disp:.6f}',
            angle_deg,
        ])

    # ── State transitions ─────────────────────────────────────────────────────

    if state == STATE_LIFT:
        set_bending_stiffness()
        vmc.target = FINGER_STRAIGHT
        if not arm_moving:
            controller.get_logger().info('Moving arm to vertical start pose ...')
            move_arm_async(start_pose(), UR5_INIT_SPEED, STATE_SETTLE_A)

    elif state == STATE_SETTLE_A:
        # Arm at start pose, finger settling to FINGER_STRAIGHT
        if time.time() - state_start_time >= SETTLE_TIME:
            angle_deg, run = experiment_queue[current_exp_idx]
            set_experiment_stiffness(angle_deg)
            vmc.target = FINGER_TARGET
            controller.get_logger().info(
                f'=== Experiment {current_exp_idx + 1}/{total_experiments}: '
                f'vertical, angle={angle_deg}°, run {run + 1}/{N_RUNS} — settling into contact ...')
            state = STATE_SETTLE_B
            state_start_time = time.time()

    elif state == STATE_SETTLE_B:
        # Finger bending into contact, wait to settle
        if time.time() - state_start_time >= SETTLE_TIME:
            angle_deg, run = experiment_queue[current_exp_idx]
            if not COLLECTED_DATA:
                csv_file, csv_writer, csv_filename = open_csv(angle_deg, run)
                controller.get_logger().info(f'Saving to: {csv_filename}')
            experiment_start_time = time.time()
            sp = start_pose().copy()
            sp[2] -= UR5_DESCENT
            controller.get_logger().info('Pressing ...')
            move_arm_async(sp, UR5_DESCENT_SPEED, STATE_ASCEND)
            state = STATE_DESCEND

    elif state == STATE_DESCEND:
        pass  # recording handled above; arm thread transitions to STATE_ASCEND

    elif state == STATE_ASCEND:
        if not arm_moving:
            move_arm_async(start_pose(), UR5_DESCENT_SPEED, STATE_NEXT)

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
    f'Directional stiffness (vertical only): {len(ANGLE_SWEEP)} angles × '
    f'{N_RUNS} runs = {total_experiments} experiments')

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

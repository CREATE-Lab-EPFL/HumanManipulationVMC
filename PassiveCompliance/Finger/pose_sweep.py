"""
Passive compliance — pose sweep (finger).

The finger presses against a surface across two K_d values and three UR5
starting Z offsets (+5 mm, 0 mm, −5 mm relative to UR5_POSE).
In each condition the arm descends HALF_DESCENT (10 mm) and returns,
recording force and position throughout.

Experiments whose output file already exists are skipped automatically,
so re-running the script only collects missing conditions.

Outputs: PassiveCompliance/Finger/outputs/pose_sweep/kd_{kd}/offset_Xmm/offset_Xmm_run_N.csv
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
from VMCFinger.FingerVMCFingerSpace import VMC
from VMCFinger.FingerGravFricLim import GravFricLim
from VMC_utils.VirtualModels import LinearSpring
from UR5_codes.UR5_config import (
    UR5_POSE, UR5_IP,
    UR5_INIT_SPEED, UR5_INIT_ACCELERATION,
    UR5_DESCENT_SPEED,
    FINGER_TARGET, FINGER_STRAIGHT,
    N_RUNS,
)

import rtde_control
import rtde_receive

# ── Experiment parameters ─────────────────────────────────────────────────────
KD_VALUES    = [0.1, 0.6]               # [N·m/rad] virtual stiffness values
DAMPING      = 0.003                    # [N·m·s/rad]
HALF_DESCENT = 0.01                     # [m] = UR5_DESCENT / 2
POSE_OFFSETS = [+0.005, 0.0, -0.005]   # [m] Z offsets relative to UR5_POSE

# Set to True once all data is collected — runs the protocol without saving files.
COLLECTED_DATA = True

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME = 3.0   # [s]

# ── Helpers ───────────────────────────────────────────────────────────────────

def kd_label(kd):
    return f'kd_{kd:.1f}'


def offset_label(offset_m):
    mm   = int(round(offset_m * 1000))
    sign = '+' if mm >= 0 else ''
    return f'offset_{sign}{mm}mm'


def start_pose_for(offset_m):
    p    = UR5_POSE.copy()
    p[2] += offset_m
    return p


def output_path(kd, offset_m, run):
    folder = os.path.join(
        os.path.dirname(__file__), 'outputs', 'pose_sweep',
        kd_label(kd), offset_label(offset_m))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{offset_label(offset_m)}_run_{run + 1}.csv')


def open_csv(kd, offset_m, run):
    fname = output_path(kd, offset_m, run)
    f = open(fname, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        'Time_s',
        'Phase',
        'Force_N',
        'Motor1_pos_deg', 'Motor2_pos_deg',
        'Motor1_vel_degs', 'Motor2_vel_degs',
        'Motor1_torque_Nm', 'Motor2_torque_Nm',
        'UR5_Z_m', 'UR5_displacement_m',
        'Stiffness_Nmrad',
        'Offset_mm',
    ])
    return f, w, fname


# ── Experiment queue ──────────────────────────────────────────────────────────
# COLLECTED_DATA = True: one pass with the smallest K_d across all three offsets.
# COLLECTED_DATA = False: full sweep (all K_d × offsets × runs).
if COLLECTED_DATA:
    experiment_queue = [(KD_VALUES[-1], off, 0) for off in POSE_OFFSETS]
else:
    experiment_queue = [(kd, off, run)
                        for off in POSE_OFFSETS
                        for kd  in KD_VALUES
                        for run in range(N_RUNS)]
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
    stiffness = np.array([KD_VALUES[0]] * 3),
    damping   = np.array([DAMPING] * 3),
    target    = FINGER_STRAIGHT,
)

# ── UR5 init ──────────────────────────────────────────────────────────────────
arm  = rtde_control.RTDEControlInterface(UR5_IP)
recv = rtde_receive.RTDEReceiveInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info('UR5 connected')

# ── State machine ─────────────────────────────────────────────────────────────
STATE_GOTO_POSE = 0
STATE_LIFT      = 1
STATE_SETTLE    = 2
STATE_DESCEND   = 3
STATE_ASCEND    = 4
STATE_NEXT      = 5
STATE_DONE      = 6

state            = STATE_GOTO_POSE
state_start_time = time.time()
current_exp_idx  = 0
arm_moving       = False

experiment_start_time = None
csv_file              = None
csv_writer            = None
csv_filename          = None


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
    if not COLLECTED_DATA and state in (STATE_DESCEND, STATE_ASCEND) and experiment_start_time is not None and csv_writer is not None:
        kd, offset, _ = experiment_queue[current_exp_idx]
        start_z    = UR5_POSE[2] + offset
        phase      = 'descent' if state == STATE_DESCEND else 'ascent'
        elapsed    = time.time() - experiment_start_time
        ur5_z      = recv.getActualTCPPose()[2]
        disp       = start_z - ur5_z
        offset_mm  = int(round(offset * 1000))
        csv_writer.writerow([
            f'{elapsed:.4f}',
            phase,
            f'{controller.force_N:.4f}',
            f'{q_deg[0]:.4f}',     f'{q_deg[1]:.4f}',
            f'{q_dot_deg[0]:.4f}', f'{q_dot_deg[1]:.4f}',
            f'{tau_total[0]:.6f}', f'{tau_total[1]:.6f}',
            f'{ur5_z:.6f}',        f'{disp:.6f}',
            f'{kd:.4f}',
            f'{offset_mm:+d}',
        ])

    # ── State transitions ─────────────────────────────────────────────────────

    if state == STATE_GOTO_POSE:
        if not arm_moving:
            kd, offset, run = experiment_queue[current_exp_idx]
            if not COLLECTED_DATA and os.path.exists(output_path(kd, offset, run)):
                controller.get_logger().info(
                    f'Skipping existing: K_d={kd}  offset={offset * 1000:+.0f} mm  run {run + 1}')
                state = STATE_NEXT
                state_start_time = time.time()
                return
            target = start_pose_for(offset)
            controller.get_logger().info(
                f'Moving to start pose  K_d={kd}  offset={offset * 1000:+.0f} mm ...')
            move_arm_async(target, UR5_INIT_SPEED, STATE_LIFT)

    elif state == STATE_LIFT:
        vmc.target = FINGER_STRAIGHT
        if time.time() - state_start_time >= SETTLE_TIME:
            kd, offset, run = experiment_queue[current_exp_idx]
            vmc.spring = LinearSpring(np.array([kd] * 3))
            vmc.target = FINGER_TARGET
            controller.get_logger().info(
                f'=== Experiment {current_exp_idx + 1}/{total_experiments}: '
                f'K_d={kd}  offset={offset * 1000:+.0f} mm  run {run + 1}/{N_RUNS} — settling ...')
            state = STATE_SETTLE
            state_start_time = time.time()

    elif state == STATE_SETTLE:
        if time.time() - state_start_time >= SETTLE_TIME:
            kd, offset, run = experiment_queue[current_exp_idx]
            if not COLLECTED_DATA:
                csv_file, csv_writer, csv_filename = open_csv(kd, offset, run)
                controller.get_logger().info(f'Saving to: {csv_filename}')
            experiment_start_time = time.time()
            descent_target    = start_pose_for(offset)
            descent_target[2] -= HALF_DESCENT
            controller.get_logger().info('Descending ...')
            move_arm_async(descent_target, UR5_DESCENT_SPEED, STATE_ASCEND)
            state = STATE_DESCEND

    elif state == STATE_DESCEND:
        pass  # recording handled above; arm thread transitions to STATE_ASCEND

    elif state == STATE_ASCEND:
        if not arm_moving:
            kd, offset, _ = experiment_queue[current_exp_idx]
            move_arm_async(start_pose_for(offset), UR5_DESCENT_SPEED, STATE_NEXT)

    elif state == STATE_NEXT:
        if csv_file and not csv_file.closed:
            csv_file.close()
            controller.get_logger().info(f'Data saved: {csv_filename}')
        csv_file = csv_writer = csv_filename = None
        current_exp_idx += 1
        if current_exp_idx >= total_experiments:
            controller.get_logger().info('=== ALL EXPERIMENTS COMPLETE ===')
            state = STATE_DONE
        else:
            state = STATE_GOTO_POSE
            state_start_time = time.time()

    elif state == STATE_DONE:
        pass


# ── Start ─────────────────────────────────────────────────────────────────────

controller.create_timer(1.0 / CONTROL_FREQUENCY, control_callback)
controller.get_logger().info(
    f'Pose sweep: {len(KD_VALUES)} K_d × {len(POSE_OFFSETS)} offsets × {N_RUNS} runs = '
    f'{total_experiments} experiments  |  K_d = {KD_VALUES}  |  descent = {HALF_DESCENT * 1e3:.0f} mm')

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

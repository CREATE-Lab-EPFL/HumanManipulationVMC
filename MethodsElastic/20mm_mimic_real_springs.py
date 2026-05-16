"""
Methods — Mimic real spring stiffness using VMC (finger).

Runs the VMC-controlled finger at the K_d values identified by
plot_elastic_band.ipynb for the soft and hard elastic bands, producing
F–d curves for direct comparison.

Set K_SOFT and K_HARD to the values printed by the notebook.

Protocol (per K, per run):
  1. Finger lifted to FINGER_STRAIGHT; wait SETTLE_TIME.
  2. Set K and target FINGER_TARGET; wait SETTLE_TIME to settle into contact.
  3. UR5 descends UR5_DESCENT — record Phase='descent'.
  4. UR5 returns to UR5_POSE  — record Phase='ascent'.

Outputs: MethodsElastic/outputs/mimic_springs/K_X.XXXX/K_X.XXXX_run_N.csv
"""

import numpy as np
import rclpy
from std_msgs.msg import Float64
import sys
import os
import json
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
    N_RUNS,
)

import rtde_control
import rtde_receive

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Identified K_d values — loaded from plot_elastic_band.ipynb output ────────
_KD_JSON = os.path.join(_HERE, 'outputs', 'mimic_springs', 'identified_kd.json')
with open(_KD_JSON) as _f:
    _kd = json.load(_f)
K_SOFT = _kd['soft']   # [N·m/rad]
K_HARD = _kd['hard']   # [N·m/rad]

DAMPING    = 0.003   # [N·m·s/rad]
SETTLE_TIME = 3.0    # [s]

# Set to True once data is collected — runs protocol without saving files.
COLLECTED_DATA = False

# ── Experiment queue ──────────────────────────────────────────────────────────
K_VALUES = [K_SOFT, K_HARD]
experiment_queue  = [(k, run) for k in K_VALUES for run in range(N_RUNS)]
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
    stiffness = np.array([K_VALUES[0]] * 3),
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

def set_stiffness(k):
    vmc.spring = LinearSpring(np.array([k] * 3))

def output_path(k, run):
    folder = os.path.join(_HERE, 'outputs', 'mimic_springs', f'K_{k:.4f}')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'K_{k:.4f}_run_{run + 1}.csv')

def open_csv(k, run):
    fname = output_path(k, run)
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

    # ── Recording ─────────────────────────────────────────────────────────────
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
            f'{q_deg[0]:.4f}',     f'{q_deg[1]:.4f}',
            f'{q_dot_deg[0]:.4f}', f'{q_dot_deg[1]:.4f}',
            f'{tau_total[0]:.6f}', f'{tau_total[1]:.6f}',
            f'{ur5_z:.6f}',        f'{disp:.6f}',
            f'{k_cur:.4f}',
            run_cur + 1,
        ])

    # ── State transitions ─────────────────────────────────────────────────────

    if state == STATE_INIT_ARM:
        if not arm_moving:
            controller.get_logger().info('Moving UR5 to UR5_POSE ...')
            move_arm_async(UR5_POSE, UR5_INIT_SPEED, STATE_LIFT)

    elif state == STATE_LIFT:
        vmc.target = FINGER_STRAIGHT
        if time.time() - state_start_time >= SETTLE_TIME:
            k_cur, run = experiment_queue[current_exp_idx]
            set_stiffness(k_cur)
            vmc.target = FINGER_TARGET
            controller.get_logger().info(
                f'=== Experiment {current_exp_idx + 1}/{total_experiments}: '
                f'K={k_cur:.4f} N·m/rad, run {run + 1}/{N_RUNS} — settling ...')
            state = STATE_SETTLE
            state_start_time = time.time()

    elif state == STATE_SETTLE:
        if time.time() - state_start_time >= SETTLE_TIME:
            k_cur, run = experiment_queue[current_exp_idx]
            if not COLLECTED_DATA:
                csv_file, csv_writer, csv_filename = open_csv(k_cur, run)
                controller.get_logger().info(f'Saving to: {csv_filename}')
            experiment_start_time = time.time()
            descent_target    = UR5_POSE.copy()
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
    f'Mimicking springs: K_soft={K_SOFT:.4f}, K_hard={K_HARD:.4f} N·m/rad, '
    f'{N_RUNS} runs each = {total_experiments} experiments')

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

"""
Proprioceptive sensing — motor transmission efficiency identification (finger).

Identifies the tendon transmission efficiency η = diag(η_MCP, η_PIP) by
recording (q, K, F_measured) triplets across N_SAMPLES random stiffness
configurations. η is fitted offline in plot_finger_eta.ipynb.

The UR5 is stationary at UR5_POSE; only the finger stiffness is varied
across N_SAMPLES random configurations drawn independently for MCP, PIP, DIP.

Protocol:
  1. UR5 moves to UR5_POSE; finger held at LIFTED_TARGET.
  2. Wait SETTLE_TIME.
  3. Set first K = BENDING_STIFFNESS (subsequent samples: random K).
  4. Lower finger to FINGER_TARGET; wait LOWER_SETTLE_TIME.
  5. Record for RECORD_INTERVAL.
  6. Lift to LIFTED_TARGET; wait LIFT_SETTLE_TIME.
  7. Repeat steps 3-6 for N_SAMPLES total configurations.

Outputs:
  ProprioceptiveSensing/outputs/finger_eta/finger_eta_run_N.csv

η is fitted offline in plot_finger_eta.ipynb.
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
from UR5_codes.UR5_config import (
    UR5_POSE, UR5_IP,
    UR5_INIT_SPEED, UR5_INIT_ACCELERATION,
    FINGER_TARGET,
    BENDING_DAMPING,
)

import rtde_control
import rtde_receive

# ── Experiment parameters ─────────────────────────────────────────────────────
N_SAMPLES    = 200              # random K configurations
BENDING_STIFFNESS = 0.5         # [N·m/rad] local bending stiffness baseline

K_MIN_SWEEP  = 0.05    # [N·m/rad] lower bound of random stiffness range
K_MAX_SWEEP  = 1.00    # [N·m/rad] upper bound of random stiffness range

# Set to True once data is collected — runs the protocol without saving files.
COLLECTED_DATA = True

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME       = 3.0   # [s] wait after UR5 reaches pose
LOWER_SETTLE_TIME = 3.0   # [s] wait after finger lowers before recording
RECORD_INTERVAL   = 5.0   # [s] recording duration per K configuration
LIFT_SETTLE_TIME  = 2.0   # [s] wait after finger lifts before changing K

# ── Lifted position (finger clear of contact) ─────────────────────────────────
LIFTED_TARGET = np.array([np.deg2rad(10.0)] * 3)

# ── ROS2 / finger init ────────────────────────────────────────────────────────
rclpy.init()
controller    = FingerController()
grav_fric_lim = GravFricLim()

for topic, attr in [('/force',        'force_total'),
                    ('/force_normal', 'force_normal'),
                    ('/force_shear',  'force_shear')]:
    controller.create_subscription(
        Float64, topic,
        lambda msg, a=attr: setattr(controller, a, msg.data * 0.00980665), 10)
controller.force_total  = 0.0
controller.force_normal = 0.0
controller.force_shear  = 0.0

vmc = VMC(
    stiffness = np.array([BENDING_STIFFNESS] * 3),
    damping   = np.array([BENDING_DAMPING]   * 3),
    target    = LIFTED_TARGET,
)

# ── UR5 init ──────────────────────────────────────────────────────────────────
arm  = rtde_control.RTDEControlInterface(UR5_IP)
recv = rtde_receive.RTDEReceiveInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info('UR5 connected')

# ── State machine ─────────────────────────────────────────────────────────────
STATE_INIT_ARM   = 0   # move UR5 to UR5_POSE
STATE_SETTLE     = 1   # wait at pose, then set first K and lower finger
STATE_LOWERING   = 2   # finger moving to FINGER_TARGET, waiting to settle
STATE_RUNNING    = 3   # recording for RECORD_INTERVAL
STATE_LIFTING    = 4   # finger lifting; then change K or finish
STATE_DONE       = 5

state             = STATE_INIT_ARM
state_start_time  = time.time()
arm_moving        = False

sample_count      = 0
current_stiffness = np.array([BENDING_STIFFNESS] * 3)
record_start_time = None
experiment_start  = time.time()

csv_file     = None
csv_writer   = None
csv_filename = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_stiffness():
    """Draw independent uniform random stiffness for MCP, PIP, DIP."""
    return np.random.uniform(K_MIN_SWEEP, K_MAX_SWEEP, 3)


def output_path():
    folder = os.path.join(os.path.dirname(__file__), 'outputs', 'finger_eta')
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder)
                if f.startswith('finger_eta_run_') and f.endswith('.csv')]
    if existing:
        nums = [int(f.replace('finger_eta_run_', '').replace('.csv', ''))
                for f in existing]
        n = max(nums) + 1
    else:
        n = 1
    return os.path.join(folder, f'finger_eta_run_{n}.csv')


def open_csv():
    fname = output_path()
    f = open(fname, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        'Sample_Num',
        'Time_s',
        'Force_Total_N', 'Force_Normal_N', 'Force_Shear_N',
        'Motor1_pos_deg', 'Motor2_pos_deg',
        'Stiffness_MCP', 'Stiffness_PIP', 'Stiffness_DIP',
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


# ── Open CSV ──────────────────────────────────────────────────────────────────
if not COLLECTED_DATA:
    csv_file, csv_writer, csv_filename = open_csv()
    controller.get_logger().info(f'Saving to: {csv_filename}')


# ── Control callback ──────────────────────────────────────────────────────────

def control_callback():
    global state, state_start_time, arm_moving
    global sample_count, current_stiffness
    global record_start_time, experiment_start
    global csv_file, csv_writer, csv_filename

    q_deg     = controller.get_joint_positions()
    q_dot_deg = controller.get_joint_velocities()
    q_motor   = np.radians(q_deg)
    q_dot_rad = np.radians(q_dot_deg)

    tau_vmc  = vmc.finger_torques(q_motor, q_dot_rad)
    tau_comp = grav_fric_lim.compute_compensation_torques(q_motor, q_dot_rad, tau_vmc)
    controller.publish_torques(tau_vmc + tau_comp)

    now = time.time()

    # ── Recording (STATE_RUNNING only) ─────────────────────────────────────────
    if not COLLECTED_DATA and state == STATE_RUNNING and record_start_time is not None:
        elapsed = now - experiment_start
        csv_writer.writerow([
            sample_count,
            f'{elapsed:.4f}',
            f'{controller.force_total:.6f}',
            f'{controller.force_normal:.6f}',
            f'{controller.force_shear:.6f}',
            f'{q_deg[0]:.4f}',
            f'{q_deg[1]:.4f}',
            f'{current_stiffness[0]:.4f}',
            f'{current_stiffness[1]:.4f}',
            f'{current_stiffness[2]:.4f}',
        ])

    # ── State transitions ─────────────────────────────────────────────────────

    if state == STATE_INIT_ARM:
        if not arm_moving:
            controller.get_logger().info('Moving UR5 to start pose …')
            move_arm_async(UR5_POSE, UR5_INIT_SPEED, STATE_SETTLE)

    elif state == STATE_SETTLE:
        if now - state_start_time >= SETTLE_TIME:
            current_stiffness = np.array([BENDING_STIFFNESS] * 3)
            vmc.spring.stiffness = current_stiffness
            sample_count = 1
            controller.get_logger().info(
                f'Sample 1/{N_SAMPLES} - '
                f'K=[{current_stiffness[0]:.3f}, {current_stiffness[1]:.3f}, '
                f'{current_stiffness[2]:.3f}] N·m/rad')
            vmc.target = FINGER_TARGET
            state = STATE_LOWERING
            state_start_time = now

    elif state == STATE_LOWERING:
        if now - state_start_time >= LOWER_SETTLE_TIME:
            record_start_time = now
            state = STATE_RUNNING
            state_start_time = now

    elif state == STATE_RUNNING:
        if now - record_start_time >= RECORD_INTERVAL:
            vmc.target = LIFTED_TARGET
            state = STATE_LIFTING
            state_start_time = now

    elif state == STATE_LIFTING:
        if now - state_start_time >= LIFT_SETTLE_TIME:
            if sample_count >= N_SAMPLES:
                controller.get_logger().info(
                    f'=== ALL {N_SAMPLES} SAMPLES COMPLETE ===')
                if not COLLECTED_DATA and csv_file and not csv_file.closed:
                    csv_file.close()
                    controller.get_logger().info(f'Data saved: {csv_filename}')
                state = STATE_DONE
            else:
                current_stiffness = random_stiffness()
                vmc.spring.stiffness = current_stiffness
                sample_count += 1
                controller.get_logger().info(
                    f'Sample {sample_count}/{N_SAMPLES} - '
                    f'K=[{current_stiffness[0]:.3f}, {current_stiffness[1]:.3f}, '
                    f'{current_stiffness[2]:.3f}] N·m/rad')
                vmc.target = FINGER_TARGET
                state = STATE_LOWERING
                state_start_time = now

    elif state == STATE_DONE:
        pass


# ── Start ─────────────────────────────────────────────────────────────────────

controller.create_timer(1.0 / CONTROL_FREQUENCY, control_callback)
controller.get_logger().info(
    f'η identification: {N_SAMPLES} samples')

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

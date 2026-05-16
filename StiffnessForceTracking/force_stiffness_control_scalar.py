"""
Stiffness and force tracking — scalar baseline for force+stiffness control (finger).

Scalar (model-free) baseline for force_stiffness_control.py.
theta_ref is updated by a direct proportional rule broadcasting the error equally
to all three joint targets:

  θ_ref(i+1) = θ_ref(i) - α · (f_meas - f_ref) · 1_3

No sensitivity matrix or stiffness model is evaluated during the control loop.
K_d remains fixed. Used for comparison against the model-based ref gradient descent.
Runs N_RUNS repetitions: between runs the finger lifts (FINGER_STRAIGHT) to reset contact.
"""

import numpy as np
import rclpy
from std_msgs.msg import Float64
import sys
import os
import csv
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from VMC_utils.VirtualModels import LinearSpring
from VMCFinger.FingerController import FingerController, CONTROL_FREQUENCY
import VMCFinger.FingerVMCFingerSpace as FingerSpaceVMC
from VMCFinger.FingerGravFricLim import GravFricLim
from UR5_codes.UR5_config import (
    FINGER_TARGET,
    FINGER_STRAIGHT,
    UR5_IP,
    UR5_POSE,
    UR5_INIT_SPEED,
    UR5_INIT_ACCELERATION,
    N_RUNS,
)
from StiffnessForceTracking.experiment_config import (
    FORCE_LEVELS, STEP_HOLD, INITIAL_SETTLE_TIME, TRACK_DURATION,
    LR_FORCE_STIFF, BENDING_DAMPING,
)

import rtde_control

# Set to True once data is collected — runs the protocol without saving files.
COLLECTED_DATA = True
BENDING_STIFFNESS = 0.5  # [N·m/rad] local bending stiffness baseline

K_fixed = np.diag([BENDING_STIFFNESS, BENDING_STIFFNESS, BENDING_STIFFNESS])


def reference_force(t):
    """Square wave over FORCE_LEVELS, STEP_HOLD s per level, repeating."""
    period = len(FORCE_LEVELS) * STEP_HOLD
    idx = int((t % period) / STEP_HOLD)
    return FORCE_LEVELS[idx]


def scalar_ref_step(theta_ref_rad, f_ref, f_measured):
    """
    Scalar proportional update on theta_ref:

      θ_ref(i+1) = θ_ref(i) - α · (f_meas - f_ref) · 1_3

    No model. Error broadcast equally to all three joint targets.
    """
    return theta_ref_rad - (LR_FORCE_STIFF / 10) * (f_measured - f_ref) * np.ones(3)


# =============================================================================
# CSV helpers
# =============================================================================
def setup_csv(experiment_name):
    folder = os.path.join(_HERE, "outputs", experiment_name)
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder)
                if f.startswith(experiment_name) and f.endswith('.csv')]
    num = max([int(f.replace(experiment_name + '_', '').replace('.csv', ''))
               for f in existing], default=0) + 1
    path = os.path.join(folder, f"{experiment_name}_{num}.csv")
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        'Time_s', 'Force_Ref_N', 'Force_Normal_N', 'Force_Total_N',
        'Force_Shear_N', 'Motor1_pos_deg', 'Motor2_pos_deg',
        'ThetaRef1_deg', 'ThetaRef2_deg', 'ThetaRef3_deg',
    ])
    return f, w, path


# =============================================================================
# ROS2 setup
# =============================================================================
rclpy.init()
controller    = FingerController()
vmc           = FingerSpaceVMC.VMC(damping=np.array([BENDING_DAMPING] * 3))
grav_fric_lim = GravFricLim()

arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info("Moving UR5 to UR5_POSE...")
arm.moveL(UR5_POSE.tolist(), UR5_INIT_SPEED, UR5_INIT_ACCELERATION)
controller.get_logger().info("UR5 at UR5_POSE")

theta_ref_current = FINGER_TARGET.copy()   # (3,) joint-space [rad]
vmc.spring = LinearSpring(K_fixed)
vmc.target = theta_ref_current

controller.get_logger().info(f"Force Stiffness Control — Scalar baseline — {N_RUNS} runs planned")
controller.get_logger().info(f"Reference: {FORCE_LEVELS} N, {STEP_HOLD} s/level, {TRACK_DURATION:.0f} s total")
controller.get_logger().info(f"lr={LR_FORCE_STIFF}  K_fixed (diagonal)={np.diag(K_fixed)}")
controller.get_logger().info("Update rule: theta_ref(i+1) = theta_ref(i) - lr*(f_meas - f_des)*1_3")

for topic, attr in [('/force', 'force_total'), ('/force_normal', 'force_normal'), ('/force_shear', 'force_shear')]:
    controller.create_subscription(
        Float64, topic,
        lambda msg, a=attr: setattr(controller, a, msg.data * 0.00980665), 10
    )
# State machine
STATE_INITIAL_SETTLE = 0
STATE_TRACK          = 1
STATE_LIFT           = 2
STATE_DONE           = 3

state               = STATE_INITIAL_SETTLE
state_start_time    = time.time()
experiment_start    = None
current_run         = 0
_last_progress_step = -1

csv_file   = None
csv_writer = None
csv_path   = None
if not COLLECTED_DATA:
    csv_file, csv_writer, csv_path = setup_csv("force_stiffness_control_scalar")
    controller.get_logger().info(f"Run 1/{N_RUNS} — saving to: {csv_path}")


def control_callback():
    global state, state_start_time, experiment_start, theta_ref_current, _last_progress_step
    global current_run, csv_file, csv_writer, csv_path

    q_deg       = controller.get_joint_positions()
    q_dot_deg   = controller.get_joint_velocities()

    q_motor     = np.radians(q_deg)
    q_dot_motor = np.radians(q_dot_deg)

    tau_vmc          = vmc.finger_torques(q_motor, q_dot_motor)
    tau_compensation = grav_fric_lim.compute_compensation_torques(
                           q_motor, q_dot_motor, tau_vmc)
    controller.publish_torques(tau_vmc + tau_compensation)

    now = time.time()

    if state == STATE_INITIAL_SETTLE:
        if now - state_start_time >= INITIAL_SETTLE_TIME:
            controller.get_logger().info(
                f"Run {current_run + 1}/{N_RUNS} — settling complete, starting scalar ref tracking.")
            experiment_start    = now
            state               = STATE_TRACK
            state_start_time    = now
            _last_progress_step = -1

    elif state == STATE_TRACK:
        elapsed = now - experiment_start

        step = int(elapsed // 30)
        if step != _last_progress_step:
            _last_progress_step = step
            controller.get_logger().info(
                f"Run {current_run + 1}/{N_RUNS} — {int(elapsed):3d} / {TRACK_DURATION:.0f} s")

        f_ref = reference_force(elapsed)
        theta_ref_new = scalar_ref_step(theta_ref_current, f_ref, controller.force_normal)
        vmc.target        = theta_ref_new
        theta_ref_current = theta_ref_new

        if not COLLECTED_DATA:
            theta_ref_deg = np.degrees(theta_ref_current)
            csv_writer.writerow([
                f"{elapsed:.4f}", f"{f_ref:.4f}",
                f"{controller.force_normal:.6f}", f"{controller.force_total:.6f}",
                f"{controller.force_shear:.6f}", f"{q_deg[0]:.4f}", f"{q_deg[1]:.4f}",
                f"{theta_ref_deg[0]:.6f}", f"{theta_ref_deg[1]:.6f}", f"{theta_ref_deg[2]:.6f}",
            ])

        if elapsed >= TRACK_DURATION:
            controller.get_logger().info(f"Run {current_run + 1}/{N_RUNS} complete.")
            if csv_file and not csv_file.closed:
                csv_file.close()
                controller.get_logger().info(f"Data saved to: {csv_path}")
            current_run += 1
            if current_run >= N_RUNS:
                controller.get_logger().info("All runs complete.")
                state = STATE_DONE
            else:
                theta_ref_current = FINGER_TARGET.copy()
                vmc.target       = FINGER_STRAIGHT
                state            = STATE_LIFT
                state_start_time = now

    elif state == STATE_LIFT:
        if now - state_start_time >= INITIAL_SETTLE_TIME:
            vmc.target = FINGER_TARGET
            if not COLLECTED_DATA:
                csv_file, csv_writer, csv_path = setup_csv("force_stiffness_control_scalar")
                controller.get_logger().info(
                    f"Run {current_run + 1}/{N_RUNS} — saving to: {csv_path}")
            state            = STATE_INITIAL_SETTLE
            state_start_time = now

    elif state == STATE_DONE:
        pass


# =============================================================================
# Run
# =============================================================================
timer_period = 1.0 / CONTROL_FREQUENCY
controller.create_timer(timer_period, control_callback)
controller.get_logger().info(
    f"Starting control loop — {N_RUNS} runs × {TRACK_DURATION:.0f} s each")

try:
    while rclpy.ok() and state != STATE_DONE:
        rclpy.spin_once(controller, timeout_sec=0.01)
except KeyboardInterrupt:
    controller.get_logger().info("Interrupted.")
finally:
    if csv_file is not None and not csv_file.closed:
        csv_file.close()
        controller.get_logger().info(f"Data saved to: {csv_path}")
    arm.stopScript()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

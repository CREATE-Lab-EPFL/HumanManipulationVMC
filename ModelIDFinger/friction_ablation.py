"""
Model ID (finger) — friction ablation study.

Step and ramp response tracking with and without friction compensation,
used to validate and identify Stribeck friction model parameters.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
import sys
import os
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from VMC_utils.GravityCompensation import *
from KinematicsFinger.FK_Finger import *
from KinematicsFinger.JacobiansFinger import *
from VMCFinger.FingerController import FingerController, CONTROL_FREQUENCY
from VMCFinger.FingerVMCFingerSpace import VMC
from VMCFinger.FingerGravFricLim import GravFricLim

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

EXPERIMENT_DURATION = 6.0
HOLD_DURATION = 1.0
RAMP_DURATION = 5.0
TARGET_ANGLES_DEG = np.array([30.0, 30.0, 30.0])
STIFFNESS = np.array([0.6, 0.6, 0.6])  # N·m/rad
RUN = "step_no_friction"
    
# Experiments configuration
EXPERIMENTS = {
    "step_with_friction": {
        "trajectory_type": "step",
        "use_friction_compensation": True,
        "description": "Step response with friction compensation"
    },
    "step_no_friction": {
        "trajectory_type": "step",
        "use_friction_compensation": False,
        "description": "Step response without friction compensation (gravity only)"
    },
    "ramp_with_friction": {
        "trajectory_type": "ramp",
        "use_friction_compensation": True,
        "description": "Ramp response with friction compensation"
    },
    "ramp_no_friction": {
        "trajectory_type": "ramp",
        "use_friction_compensation": False,
        "description": "Ramp response without friction compensation (gravity only)"
    },
}

# =============================================================================
# TRAJECTORY GENERATOR
# =============================================================================

class TrajectoryGenerator:
    """
    Generates step and ramp reference trajectories for joint angles.
    """

    def __init__(self, trajectory_type, target_angles_deg):
        """
        Args:
            trajectory_type: "step" or "ramp"
            target_angles_deg: Target joint angles in degrees [MCP, PIP, DIP]
        """

        self.trajectory_type = trajectory_type
        self.target_angles_rad = np.deg2rad(target_angles_deg)
        self.zero_angles = np.zeros(3)

    def get_reference(self, elapsed_time):
        """
        Get reference joint angles based on elapsed time.

        Args:
            elapsed_time: Time since experiment start (seconds)

        Returns:
            reference_angles: Reference joint angles in radians [MCP, PIP, DIP]
        """

        if elapsed_time < HOLD_DURATION:
            return self.zero_angles.copy()

        time_since_start = elapsed_time - HOLD_DURATION

        if self.trajectory_type == "step":
            return self.target_angles_rad.copy()

        elif self.trajectory_type == "ramp":
            if time_since_start >= RAMP_DURATION:
                return self.target_angles_rad.copy()
            else:
                alpha = time_since_start / RAMP_DURATION
                return self.zero_angles + alpha * self.target_angles_rad

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_csv(experiment_name):
    """
    Setup CSV file for logging experiment data.
    """

    output_folder = os.path.join(os.path.dirname(__file__), "outputs", "friction_ablation")
    os.makedirs(output_folder, exist_ok=True)
    csv_filename = os.path.join(output_folder, f"{experiment_name}.csv")
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    headers = [
        'Time_s',
        'Ref_MCP_deg', 'Ref_PIP_deg',
        'MCP_deg', 'PIP_deg'
    ]
    csv_writer.writerow(headers)

    return csv_file, csv_writer, csv_filename

# =============================================================================
# INITIALIZATION
# =============================================================================

rclpy.init()
controller = FingerController()

vmc = VMC()
vmc.spring.stiffness = STIFFNESS
grav_fric_lim = GravFricLim()

controller.get_logger().info("VMC Friction Ablation Controller started")
controller.get_logger().info(f"Control frequency: {CONTROL_FREQUENCY} Hz")
controller.get_logger().info(f"Stiffness: {STIFFNESS} N·m/rad")

exp_config = EXPERIMENTS[RUN]

trajectory_gen = TrajectoryGenerator(exp_config["trajectory_type"], TARGET_ANGLES_DEG)
csv_file, csv_writer, csv_filename = setup_csv(RUN)

controller.get_logger().info(f"=== Starting experiment: {RUN} ===")
controller.get_logger().info(f"Description: {exp_config['description']}")
controller.get_logger().info(f"Friction compensation: {exp_config['use_friction_compensation']}")
controller.get_logger().info(f"Trajectory: {exp_config['trajectory_type']}")
controller.get_logger().info(f"Saving to: {csv_filename}")

# Control state
experiment_start_time = time.time()
control_iteration = 0
LOG_EVERY_N_ITERATIONS = 100

# =============================================================================
# CONTROL CALLBACK
# =============================================================================

def control_callback():
    global experiment_start_time, control_iteration
    global csv_file, csv_writer, csv_filename
    global trajectory_gen, exp_config

    current_time = time.time()
    elapsed_time = current_time - experiment_start_time

    # Check if experiment duration exceeded
    if elapsed_time >= EXPERIMENT_DURATION:
        # IMPORTANT: Set torques to zero before shutting down
        controller.publish_torques(np.zeros(2))
        csv_file.close()
        controller.get_logger().info(f"Experiment completed. Data saved to: {csv_filename}")
        controller.get_logger().info("=== EXPERIMENT COMPLETE ===")
        rclpy.shutdown()
        return

    # Get reference trajectory
    reference_angles_rad = trajectory_gen.get_reference(elapsed_time)

    # Get current motor states
    q_deg = controller.get_joint_positions()
    q_dot_deg = controller.get_joint_velocities()
    q_motor = np.radians(q_deg)
    q_dot_motor = np.radians(q_dot_deg)

    # Update VMC target
    vmc.target = reference_angles_rad

    # Compute VMC torques
    tau_vmc = vmc.finger_torques(q_motor, q_dot_motor)

    # Compute compensation torques
    tau_gravity = grav_fric_lim.gravity_compensation(q_motor)
    tau_limits = grav_fric_lim.deadzone_spring_torques(q_motor)

    # Conditionally apply friction compensation based on experiment config
    if exp_config["use_friction_compensation"]:
        tau_friction = grav_fric_lim.friction_compensation(q_dot_motor, tau_gravity + tau_limits + tau_vmc)
    else:
        tau_friction = np.zeros(2)  # No friction compensation

    # Total torque
    tau_total = tau_vmc + tau_gravity + tau_limits + tau_friction

    # Publish torques
    controller.publish_torques(tau_total)

    # Log data to CSV
    reference_angles_deg = np.rad2deg(reference_angles_rad)
    angles_deg = np.rad2deg(motor_to_joint(q_motor))
    csv_writer.writerow([
        f"{elapsed_time:.4f}",
        f"{reference_angles_deg[0]:.4f}", f"{reference_angles_deg[1]:.4f}",
        f"{angles_deg[0]:.4f}", f"{angles_deg[1]:.4f}"
    ])
    
    # Log status periodically
    control_iteration += 1
    if control_iteration % LOG_EVERY_N_ITERATIONS == 0:
        controller.get_logger().info(
            f"Time: {elapsed_time:.2f}s | "
            f"Ref: [{reference_angles_deg[0]:.1f}, {reference_angles_deg[1]:.1f}]° | "
            f"Real: [{angles_deg[0]:.1f}, {angles_deg[1]:.1f}]° | "
            f"Friction: {'ON' if exp_config['use_friction_compensation'] else 'OFF'}"
        )

# =============================================================================
# MAIN LOOP
# =============================================================================

timer_period = 1.0 / CONTROL_FREQUENCY
controller.create_timer(timer_period, control_callback)

controller.get_logger().info(f"Duration: {EXPERIMENT_DURATION}s ({HOLD_DURATION}s hold + {EXPERIMENT_DURATION - HOLD_DURATION}s trajectory)")
controller.get_logger().info(f"Target: {TARGET_ANGLES_DEG[0]}° MCP, {TARGET_ANGLES_DEG[1]}° PIP")

try:
    rclpy.spin(controller)
except KeyboardInterrupt:
    controller.get_logger().info("Shutting down")
finally:
    # Zero torques on shutdown for safety
    try:
        controller.publish_torques(np.zeros(2))
    except:
        pass
    if csv_file is not None and not csv_file.closed:
        csv_file.close()
        controller.get_logger().info(f"Data saved to: {csv_filename}")
    controller.destroy_node()
    rclpy.shutdown()

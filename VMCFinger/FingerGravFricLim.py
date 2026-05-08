## Gravity, Friction, and Joint Limit Compensation for Finger VMC
## =====================
## To get motor positions:         /joint_positions (degrees)
## To get motor velocities:        /joint_velocities (deg/s)
## To publish torques:             /goal_torque (N·m)
## Always run before:              ros2 run dynamixel_interface dynamixel_node
## Speed:                   sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
##
## All units are SI:
## - Lengths: meters [m]
## - Angles: radians [rad] (ROS topics use degrees, converted internally)
## - Forces: Newtons [N]
## - Torques: Newton-meters [N·m]
## - Velocities: meters/second [m/s], radians/second [rad/s]

import numpy as np
from VMC_utils.VirtualModels import *
from VMC_utils.GravityCompensation import *
from KinematicsFinger.FK_Finger import *
from KinematicsFinger.JacobiansFinger import *
from ModelIDFinger.finger_params import (
    R_world_to_finger,
    mass_MCP, mass_PIP, mass_DIP,
    cog_MCP, cog_PIP, cog_DIP,
    joint_limits, friction_max, friction_vlim, limit_stiffness,
)

class GravFricLim:
    """
    Gravity, Friction, and Joint Limit compensation for finger VMC.
    Applies gravity compensation, friction compensation using Stribeck's model,
    and joint limit torques to prevent exceeding joint limits.
    """

    def __init__(self):
        """Initialize VMC with finger parameters and compensation components."""

        # Gravity compensation for each link
        self.gravity_comp = {
            'MCP': GravityCompensation(mass=mass_MCP, R_world_to_system=R_world_to_finger),
            'PIP': GravityCompensation(mass=mass_PIP, R_world_to_system=R_world_to_finger),
            'DIP': GravityCompensation(mass=mass_DIP, R_world_to_system=R_world_to_finger)
        }

        # Center of gravity for each link (local coordinates in m, from URDF)
        self.cog_points = {'MCP': cog_MCP, 'PIP': cog_PIP, 'DIP': cog_DIP}

        # Initialize Jacobian functions
        self.J_motors = q2joint()
        self.J_MCP = q2MCP()
        self.J_PIP = q2PIP()
        self.J_DIP = q2DIP()

        # Joint limit springs (DeadzoneLimitSpring for each joint)
        self.limit_springs = {
            'MCP': DeadzoneLimitSpring(limit_stiffness, joint_limits['MCP'][0], joint_limits['MCP'][1]),
            'PIP': DeadzoneLimitSpring(limit_stiffness, joint_limits['PIP'][0], joint_limits['PIP'][1]),
            'DIP': DeadzoneLimitSpring(limit_stiffness, joint_limits['DIP'][0], joint_limits['DIP'][1])
        }

    def compute_jacobian_cog(self, q_motor, link_name):
        """
        Compute Jacobian for a point on specified link cog.

        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)
            link_name: "MCP", "PIP", or "DIP"

        Returns:
            J: 3x2 Jacobian matrix
        """

        r_local = self.cog_points[link_name]

        if link_name == "MCP":
            J = self.J_MCP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
        elif link_name == "PIP":
            J = self.J_PIP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
        elif link_name == "DIP":
            J = self.J_DIP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
        else:
            raise ValueError(f"Unknown link: {link_name}")

        return np.array(J)

    def friction_compensation(self, q_dot_motor, tau_motor):
        """
        Compute Stribeck friction compensation torques for the motors.

        Model: τ_friction = F_max * exp(-(v/v_lim)²) * sign(τ_commanded)

        Args:
            q_dot_motor: [q_dot_MCP, q_dot_PIP] motor velocities (rad/s)
            tau_motor: [tau_MCP, tau_PIP] current motor torques (N·m)

        Returns:
            tau_friction: [tau_MCP, tau_PIP] friction compensation torques (N·m)
        """

        return friction_max * np.exp(-(q_dot_motor / friction_vlim) ** 2) * np.sign(tau_motor)

    def gravity_compensation(self, q_motor):
        """
        Compute gravity compensation torques for the finger.

        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)

        Returns:
            tau_gravity: [tau_MCP, tau_PIP] motor torques (N·m)
        """

        tau_gravity = np.zeros(2)

        for link_name in ['MCP', 'PIP', 'DIP']:
            J_cog = self.compute_jacobian_cog(q_motor, link_name)
            F_gravity = self.gravity_comp[link_name].compute_force()
            tau_gravity += J_cog.T @ F_gravity

        return tau_gravity

    def deadzone_spring_torques(self, q_motor):
            """
            Compute torques from joint limit deadzone springs.

            Args:
                q_motor: [q_MCP, q_PIP] motor angles (rad)

            Returns:
                tau_limits: [tau_MCP, tau_PIP] motor torques (N·m)
            """

            theta = motor_to_joint(q_motor)

            # Compactly compute torques for MCP(0), PIP(1), and DIP(2)
            tau_limits_joint = np.array([
                self.limit_springs[joint].compute_force(theta[i])
                for i, joint in enumerate(['MCP', 'PIP', 'DIP'])
            ])

            # Map joint torques back to motor torques
            # Note: *q_motor unpacks the array into the two arguments required by J_motors
            return self.J_motors(*q_motor).T @ tau_limits_joint

    def compute_compensation_torques(self, q_motor, q_dot_motor, tau_vmc):
        """
        Compute compensation control torques for the finger motors.

        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)
            q_dot_motor: [q_dot_MCP, q_dot_PIP] motor velocities (rad/s)
            tau_vmc: [tau_MCP, tau_PIP] torques from the virtual model controller (N·m)
        Returns:
            tau_total: [tau_MCP, tau_PIP] total motor torques including compensation (N·m)
        """

        tau_gravity = self.gravity_compensation(q_motor)
        tau_limits = self.deadzone_spring_torques(q_motor)
        tau_friction = self.friction_compensation(q_dot_motor, tau_gravity + tau_limits + tau_vmc)

        return tau_gravity + tau_limits + tau_friction

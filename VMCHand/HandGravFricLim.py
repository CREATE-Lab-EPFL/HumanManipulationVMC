## Gravity, Friction, and Joint Limit Compensation for Hand VMC
## =====================
## To get motor positions:         /joint_positions (degrees)
## To get motor velocities:        /joint_velocities (deg/s)
## To publish torques:             /goal_torque (N·m)
## Always run before:              ros2 run dynamixel_interface dynamixel_node --ros-args -p baudrate:=2000000 -p motor_ids:=[0,1,2,3,4,5,6,7,8,9,10,11,12] -p position_motor_ids:=[13,14]
## Speed:                          sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
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
from KinematicsHand.FK_Hand import *
from KinematicsHand.JacobiansHand import *
from ModelIDHand.hand_params import (
    MASSES, COG, JOINT_LIMITS,
    friction_max, friction_vlim, limit_stiffness,
    HAND_MOUNTING_ANGLE,
)

class GravFricLim:
    """
    Gravity, friction, and joint limit compensation for the 15-DOF hand model.

    PLA masses (g):
    - Wrist/Palm: Wrist_joint=10.4g, Palm=80.8g
    - Thumb: base=3.5g, 2dof=7.0g, proximal=6.8g, middle=4.3g, distal=3.5g
    - Index/Ring: base=1.4g, 2dof=1.1g, proximal=5.7g, middle=4.0g, distal=2.5g
    - Middle: base=1.4g, 2dof=1.1g, proximal=6.6g, middle=4.0g, distal=2.5g
    - Pinky: base=1.4g, 2dof=1.1g, proximal=4.3g, middle=2.2g, distal=2.5g
    """

    def __init__(self, R_world2hand=None):
        """
        Initialize compensation with hand parameters.

        Args:
            R_world2hand: 3x3 rotation matrix mapping world vectors to hand base frame.
                          Use get_tcp_rotation_matrix().T when the hand is on the UR5.
                          Defaults to identity (gravity along -z of hand frame).
        """

        self.R_world2hand = np.eye(3) if R_world2hand is None else np.asarray(R_world2hand)

        # Max static-friction compensation torque [N·m]. Scalar or [13] per-motor array.
        # Override per-task (e.g. lower/zero it to kill stiction limit-cycles on
        # unsprung fingers). Defaults to the model value from hand_params.
        self.friction_max = friction_max

        # Fixed rotation from UR5 flange frame to hand base frame (from hand_params)
        self._R_mounting = rot_axis(np.array([0.0, 0.0, 1.0]), HAND_MOUNTING_ANGLE)

        # Gravity compensation for each link
        self.gravity_comp = {
            link: GravityCompensation(mass=MASSES[link], R_world_to_system=self.R_world2hand)
            for link in MASSES
        }

        # Center of gravity for each link
        self.cog_points = COG

        # Joint limit springs
        self.limit_springs = {
            joint: DeadzoneLimitSpring(limit_stiffness, limits[0], limits[1])
            for joint, limits in JOINT_LIMITS.items()
        }

        # Jacobian functions
        self.jac = HandJacobians()

    def set_orientation(self, R_world2hand):
        """Update hand orientation and propagate to all GravityCompensation objects."""
        self.R_world2hand = np.asarray(R_world2hand)
        for gc in self.gravity_comp.values():
            gc.R_world_to_system = self.R_world2hand

    def gravity_compensation(self, q_motor):
        """
        Compute gravity compensation torques for the entire hand.

        Args:
            q_motor: [13] motor angles (radians).

        Returns:
            tau_gravity: [13] gravity compensation motor torques (N·m).
        """

        tau = np.zeros(13)

        # wrist_joint and palm have zero Jacobian (rigid wrist) → skip them
        THUMB_MAP  = {'base': 'CMC1', '2dof_joint': 'CMC1', 'proximal': 'CMC2', 'middle': 'MCP', 'distal': 'IP'}
        FINGER_MAP = {'base': 'MCP', '2dof_joint': 'MCP', 'proximal': 'MCP', 'middle': 'PIP', 'distal': 'DIP'}

        for link, cog in self.cog_points.items():
            if link in ('wrist_joint', 'palm'):
                continue  # constant position with rigid wrist → zero contribution
            elif link.startswith('thumb_'):
                J = self.jac.get_thumb_jacobian(THUMB_MAP[link.split('_', 1)[1]], q_motor, cog)
            else:
                finger, part = link.split('_', 1)
                J = self.jac.get_finger_jacobian(finger, FINGER_MAP[part], q_motor, cog)

            tau += np.array(J).T @ self.gravity_comp[link].compute_force()

        return tau

    def friction_compensation(self, q_dot_motor, tau_motor):
        """
        Compute Stribeck friction compensation torques for the entire hand.

        Model: τ_friction = F_max * exp(-(v/v_lim)²) * sign(τ_commanded)

        Args:
            q_dot_motor: [13] motor velocities (radians/second).
            tau_motor:   [13] commanded motor torques (N·m).

        Returns:
            tau_friction: [13] friction compensation motor torques (N·m).
        """
        return self.friction_max * np.exp(-(q_dot_motor / friction_vlim) ** 2) * np.sign(tau_motor)

    def joint_limit_torques(self, q_motor):
        """
        Compute joint limit motor torques from deadzone springs.

        Args:
            q_motor: [13] motor angles (radians).

        Returns:
            tau_limits: [13] joint limit motor torques (N·m).
        """
        tau = np.zeros(13)

        # Thumb: [CMC1, CMC2, MCP, IP]
        theta = FK_motor2thumb(q_motor)
        J = np.array(self.jac.get_angles_jacobian('thumb', q_motor))
        tau_joint = np.array([self.limit_springs[f'thumb_{j}'].compute_force(theta[i])
                              for i, j in enumerate(['CMC1', 'CMC2', 'MCP', 'IP'])])
        tau += J.T @ tau_joint

        # Fingers: spread + [MCP, PIP, DIP]
        for finger in ['index', 'middle', 'ring', 'pinky']:
            theta_s = FK_motor2spread(q_motor, finger)
            J_s = np.array(self.jac.get_spread_jacobian(finger, q_motor))
            tau += J_s.flatten() * self.limit_springs[f'{finger}_spread'].compute_force(theta_s)

            theta_f = FK_motor2finger(q_motor, finger)
            J_f = np.array(self.jac.get_angles_jacobian(finger, q_motor))
            tau_joint = np.array([self.limit_springs[f'{finger}_{j}'].compute_force(theta_f[i])
                                  for i, j in enumerate(['MCP', 'PIP', 'DIP'])])
            tau += J_f.T @ tau_joint

        return tau

    def compute_compensation_torques(self, q_motor, q_dot_motor, tau_motor, R_tcp=None):
        """
        Compute total compensation control torques for the hand motors.

        Args:
            q_motor:     [13] motor angles (radians).
            q_dot_motor: [13] motor velocities (radians/second).
            tau_motor:   [13] torques from the Virtual Model Controller (N·m).
            R_tcp: optional 3x3 rotation matrix of the UR5 TCP frame in world
                   (i.e. recv.get_tcp_rotation_matrix(), NOT pre-transposed).
                   Combined with HAND_MOUNTING_ANGLE from hand_params to give
                   the true world→hand rotation: (R_tcp @ R_mounting).T

        Returns:
            tau_compensation: [13] total motor compensation torques (N·m).
        """

        if R_tcp is not None:
            self.set_orientation((R_tcp @ self._R_mounting).T)

        tau_gravity = self.gravity_compensation(q_motor)
        tau_limits = self.joint_limit_torques(q_motor)
        tau_friction = self.friction_compensation(q_dot_motor, tau_motor + tau_gravity + tau_limits)
        tau_compensation = tau_gravity + tau_friction + tau_limits

        return tau_compensation

"""
    VMC Controller for Finger in Finger Space.
    Implements a linear spring by default, but can be substituted
    by more complex virtual components.
"""

## To get motor positions:         /joint_positions (degrees)
## To get motor velocities:        /joint_velocities (deg/s)
## To publish torques:             /goal_torque (N·m)
## Always run before:              ros2 run dynamixel_interface dynamixel_node
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
from KinematicsFinger.FK_Finger import *
from KinematicsFinger.JacobiansFinger import *

# VMC Finger Controller Class
class VMC:
    """
    Virtual Model Controller for finger in Finger Space.

    Applies virtual springs and dampers to tune the finger's response,
    and computes joint torques via Jacobian transpose method.
    """

    def __init__(self, stiffness=np.array([0.0, 0.0, 0.0]), damping=np.array([0.0, 0.0, 0.0]), target=np.array([0.0, 0.0, 0.0])):
        """Initialize VMC with finger parameters and virtual components."""

        # Initialize Jacobian functions
        self.J_motors = q2joint()

        # Virtual springs and dampers for each joint
        self.stiffness = stiffness
        self.damping = damping
        self.spring = LinearSpring(self.stiffness)
        self.damper = LinearDamper(self.damping)
        self.target = target

    def compute_jacobian_joint(self, q_motor):
        """
        Compute Jacobian for specified link joint angles.

        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)

        Returns:
            J: 3x2 Jacobian matrix
        """

        return self.J_motors(q_motor[0], q_motor[1])
    
    def finger_torques(self, q_motor, q_dot_motor):
        """
        Compute torques from virtual springs and dampers at attachment points.

        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)
            q_dot_motor: [q_dot_MCP, q_dot_PIP] motor angular velocities (rad/s)

        Returns:
            tau_finger: [tau_MCP, tau_PIP] motor torques (N*m)
        """

        # Initialize finger torques
        tau_finger = np.zeros(2)

        # TIP: spring and damper forces
        J = self.compute_jacobian_joint(q_motor)
        angle_current = motor_to_joint(q_motor)
        vel_current = J @ q_dot_motor
        tau_spring = self.spring.compute_force(angle_current, self.target)
        tau_damper = self.damper.compute_force(vel_current)

        # Consider all contributions
        tau_finger += J.T @ (tau_spring + tau_damper)

        return tau_finger
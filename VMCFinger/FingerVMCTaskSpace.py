"""
    VMC Controller for Finger in Task Space.
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
from ModelIDFinger.finger_params import a, b, c

# VMC Finger Controller Class
class VMC:
    """
    Virtual Model Controller for finger in Task Space.

    Applies virtual springs and dampers to tune the finger's response,
    and computes joint torques via Jacobian transpose method.
    """

    JOINTS = ['MCP', 'PIP', 'DIP']

    def __init__(self, stiffness=np.array([0.0, 0.0, 0.0]), damping=np.array([0.0, 0.0, 0.0]), target_MCP=np.array([0.0, a, 0.0]), target_PIP=np.array([0.0, a+b, 0.0]), target_DIP=np.array([0.0, a+b+c, 0.0])):
        """Initialize VMC with finger parameters and virtual components."""

        # Jacobian functions per joint
        self.J_motors = q2joint()
        self.J_funcs = {'MCP': q2MCP(), 'PIP': q2PIP(), 'DIP': q2DIP()}

        # Virtual attachment points in link frames
        self.attachment_points = {
            'MCP': np.array([0, a, 0]),  # Proximal end of Proximal phalanx
            'PIP': np.array([0, b, 0]),  # Middle of Proximal phalanx
            'DIP': np.array([0, c, 0])   # Fingertip of Distal phalanx
        }

        # Virtual springs and dampers per joint
        self.springs = {j: LinearSpring(stiffness) for j in self.JOINTS}
        self.dampers = {j: LinearDamper(damping) for j in self.JOINTS}

        # Target positions per joint
        self.targets = {'MCP': target_MCP, 'PIP': target_PIP, 'DIP': target_DIP}

    def compute_jacobian(self, q_motor, link_name):
        """
        Compute Jacobian for a point on specified link.
        
        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)
            link_name: "MCP", "PIP", or "DIP"
        
        Returns:
            J: 3x2 Jacobian matrix
        """

        r = self.attachment_points[link_name]
        J = self.J_funcs[link_name](q_motor[0], q_motor[1], r[0], r[1], r[2])
        return np.array(J)

    def finger_torques(self, q_motor, q_dot_motor):
        """
        Compute torques from virtual springs and dampers at attachment points.

        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)
            q_dot_motor: [q_dot_MCP, q_dot_PIP] motor velocities (rad/s)

        Returns:
            tau_finger: [tau_MCP, tau_PIP] motor torques (N*m)
        """

        tau_finger = np.zeros(2)

        for j in self.JOINTS:
            J = self.compute_jacobian(q_motor, j)
            pos = FK(q_motor, j, self.attachment_points[j])
            vel = J @ q_dot_motor
            F_spring = self.springs[j].compute_force(pos, self.targets[j])
            F_damper = self.dampers[j].compute_force(vel)
            tau_finger += J.T @ (F_spring + F_damper)

        return tau_finger
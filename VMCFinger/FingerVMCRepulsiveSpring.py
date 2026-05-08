"""
    VMC Controller for Finger - finger-space spring + cart repulsive spring for overall stiffness shaping.

    Two virtual elements act simultaneously:
      1. Linear spring + damper in joint (finger) space - gives baseline stiffness
         to all joint motions (k_d [N·m/rad]).
      2. ConstrainedGaussianSpring + ConstrainedLinearDamper at the fingertip in
         Cartesian space, constrained to the Z direction via a horizontal cart -
         adds stiffness only in the pressing direction (k_cart [N/m]).

    The cart constraint (P = n @ n.T, n = [0,0,1] by default) zeroes the X and Y
    components of the spring force, so only vertical displacement at the fingertip
    drives a repulsive force.    
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
from VMC_utils.VirtualModels import LinearSpring, LinearDamper, ConstrainedGaussianSpring, ConstrainedLinearDamper
from KinematicsFinger.FK_Finger import motor_to_joint, joint_to_motor, FK_DIP
from KinematicsFinger.JacobiansFinger import q2joint, q2DIP
from ModelIDFinger.finger_params import c


class VMC:
    """
    Virtual Model Controller - finger-space spring and directional cart repulsive spring.

    Combines:
      - Joint-space spring/damper (k_d) for baseline finger stiffness.
      - Cartesian cart repulsive spring/damper (k_cart) at the fingertip, constrained
        along n (default Z), for directional stiffness reduction in the pressing direction.
    """

    def __init__(self,
                 stiffness      = np.array([0.8, 0.8, 0.8]),
                 damping        = np.array([0.0, 0.0, 0.0]),
                 cart_strength  = 10.0,
                 cart_sigma     = 0.05,
                 cart_damping   = 0.0,
                 target         = np.array([0.0, 0.0, 0.0]),
                 n              = np.array([[0], [0], [1]])):
        """
        Args:
            stiffness:      joint-space spring stiffness [N·m/rad] - scalar, vector, or matrix
            damping:        joint-space damper coefficient [N·m·s/rad] - scalar, vector, or matrix
            cart_strength:  cart repulsive spring strength [N] at zero displacement
            cart_sigma:     cart repulsive spring width [m] - controls how quickly force falls
            cart_damping:   cart damper coefficient [N·s/m]
            target:         joint angle target [theta_MCP, theta_PIP, theta_DIP] (rad)
            n:              unit normal to the cart plane - spring/damper act along this direction
        """

        # Jacobian functions
        self.J_motors = q2joint()
        self.J_DIP    = q2DIP()

        # Fingertip position in the DIP link frame
        self.r_tip = np.array([0, c, 0])

        # Finger-space virtual elements
        self.stiffness = stiffness
        self.damping = damping
        self.spring = LinearSpring(self.stiffness)
        self.damper = LinearDamper(self.damping)

        # Cart-space virtual elements
        self.cart_strength = cart_strength
        self.cart_sigma = cart_sigma
        self.cart_damping = cart_damping
        self.cart_spring = ConstrainedGaussianSpring(self.cart_strength, self.cart_sigma, n)
        self.cart_damper = ConstrainedLinearDamper(self.cart_damping, n)

        # Targets
        self.target = target
        q_target_motor = joint_to_motor(target)
        self.target_cart = FK_DIP(q_target_motor, self.r_tip)  # full 3D; only Z used by cart

    def compute_jacobian_joint(self, q_motor):
        """
        Jacobian mapping motor velocities to joint angle velocities (3×2).

        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)
        
        Returns:
            J: 3×2 matrix 
        """
        return self.J_motors(q_motor[0], q_motor[1])
    
    def compute_jacobian_tip(self, q_motor):
        """
        Jacobian mapping motor velocities to fingertip Cartesian velocities (3×2).

        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)
        
        Returns:
            J: 3×2 matrix 
        """
        return np.array(self.J_DIP(q_motor[0], q_motor[1], *self.r_tip))
    
    def finger_torques(self, q_motor, q_dot_motor):
        """
        Compute finger torques from virtual model forces.

        Args:
            q_motor: [q_MCP, q_PIP] motor angles (rad)
            q_dot_motor: [q_dot_MCP, q_dot_PIP] motor angular velocities (rad/s)

        Returns:
            tau_finger: [tau_MCP, tau_PIP] motor torques (N·m)
        """

        # ── Finger-space spring + damper ──────────────────────────────────────
        J_joint = self.compute_jacobian_joint(q_motor)
        angle_current = motor_to_joint(q_motor)
        vel_joint = J_joint @ q_dot_motor

        tau_spring = self.spring.compute_force(angle_current, self.target)
        tau_damper = self.damper.compute_force(vel_joint)

        # ── Cart-space repulsive spring + damper ──────────────────────────────
        J_tip = self.compute_jacobian_tip(q_motor)
        p_tip = FK_DIP(q_motor, self.r_tip)
        vel_cart = J_tip @ q_dot_motor

        F_cart_spring = self.cart_spring.compute_force(p_tip, self.target_cart)
        F_cart_damper = self.cart_damper.compute_force(vel_cart)

        # ── Total motor torques ───────────────────────────────────────────────
        tau_finger = J_joint.T @ (tau_spring + tau_damper)
        tau_finger += J_tip.T @ (F_cart_spring + F_cart_damper)

        return tau_finger
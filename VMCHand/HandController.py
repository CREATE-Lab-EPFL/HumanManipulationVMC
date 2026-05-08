## VMC Hand Controller
## =====================
## To get motor positions:         /joint_positions (degrees)
## To get motor velocities:        /joint_velocities (deg/s)
## To publish torques:             /goal_torque (N·m)
## Always run before:              ros2 run dynamixel_interface dynamixel_node --ros-args -p baudrate:=2000000 -p motor_ids:=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
## Speed:                          sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
##
## All units are SI:
## - Lengths: meters [m]
## - Angles: radians [rad] (ROS topics use degrees, converted internally)
## - Forces: Newtons [N]
## - Torques: Newton-meters [N·m]
## - Velocities: meters/second [m/s], radians/second [rad/s]

import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from ModelIDHand.motor_config import hardware_to_software, software_to_hardware
from ModelIDHand.hand_params import goal_limit_torque, goal_limit_torque_wrist

# Define ROS2 frequency
CONTROL_FREQUENCY = 330  # Hz

# Per-motor torque limits in software order: wrist_motor1, wrist_motor2 use the
# wrist limit; all finger/thumb motors use the standard limit.
TORQUE_LIMITS = np.array(
    [goal_limit_torque_wrist] * 2 + [goal_limit_torque] * 13
)


class HandController(Node):
    """
    ROS2 node for hand control.

    Handles hardware/software motor ordering conversion internally.
    All public methods use software ordering.
    """

    def __init__(self):
        super().__init__('hand_controller')

        # Publisher for torque commands (in Nm)
        self.torque_pub = self.create_publisher(Float64MultiArray, '/goal_torque', 10)

        # Subscribers for joint states - positions in degrees, velocities in deg/s
        self.position_sub = self.create_subscription(
            Float64MultiArray, '/joint_positions', self.position_callback, 10)
        self.velocity_sub = self.create_subscription(
            Float64MultiArray, '/joint_velocities', self.velocity_callback, 10)

        # Current joint states in SOFTWARE order (radians and rad/s)
        self.joint_positions = np.zeros(15)
        self.joint_velocities = np.zeros(15)

    def position_callback(self, msg):
        """Callback for joint position updates (hardware order, degrees)."""
        q_hw_deg = np.array(msg.data)
        self.joint_positions = np.radians(hardware_to_software(q_hw_deg))

    def velocity_callback(self, msg):
        """Callback for joint velocity updates (hardware order, deg/s)."""
        q_dot_hw_deg = np.array(msg.data)
        self.joint_velocities = np.radians(hardware_to_software(q_dot_hw_deg))

    def get_joint_positions(self):
        """
        Get current joint positions in software order.

        Returns:
            q_motor: [15] motor angles (radians).
        """
        return self.joint_positions

    def get_joint_velocities(self):
        """
        Get current joint velocities in software order.

        Returns:
            q_dot_motor: [15] motor velocities (radians/second).
        """
        return self.joint_velocities

    def publish_torques(self, torques):
        """
        Publish torque commands to motors.

        Args:
            torques: [15] motor torques in software order (N·m).
        """
        torques = np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)
        tau_hw = software_to_hardware(torques)

        # Debug: print torques being sent (every 100 calls to avoid spam)
        if not hasattr(self, '_pub_count'):
            self._pub_count = 0
        self._pub_count += 1
        if self._pub_count % 100 == 0:
            self.get_logger().info(f'Torques (Nm): {np.round(tau_hw, 3).tolist()}')

        msg = Float64MultiArray()
        msg.data = list(tau_hw)
        self.torque_pub.publish(msg)

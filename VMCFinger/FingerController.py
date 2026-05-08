## VMC Finger Controller
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
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from ModelIDFinger.finger_params import goal_torque_limit

CONTROL_FREQUENCY = 900  # Hz

class FingerController(Node):
    """ROS2 node interface for the finger. Reads joint states, publishes torques."""

    def __init__(self):
        super().__init__('finger_controller')

        self.torque_pub   = self.create_publisher(Float64MultiArray, '/goal_torque', 10)
        self.position_sub = self.create_subscription(Float64MultiArray, '/joint_positions',
                                                     self._pos_cb, 10)
        self.velocity_sub = self.create_subscription(Float64MultiArray, '/joint_velocities',
                                                     self._vel_cb, 10)

        self.joint_positions  = np.zeros(2)  # [deg]
        self.joint_velocities = np.zeros(2)  # [deg/s]

    def _pos_cb(self, msg): self.joint_positions  = np.array(msg.data)
    def _vel_cb(self, msg): self.joint_velocities = np.array(msg.data)

    def get_joint_positions(self):  return self.joint_positions
    def get_joint_velocities(self): return self.joint_velocities

    def publish_torques(self, torques):
        torques = np.clip(torques, -goal_torque_limit, goal_torque_limit)
        msg = Float64MultiArray()
        msg.data = list(torques)
        self.torque_pub.publish(msg)

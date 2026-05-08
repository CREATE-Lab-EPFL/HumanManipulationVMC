"""
ROS2 Arduino Force Publisher

This module reads force data from Arduino load cells and publishes ROS2 messages.
"""

import serial
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class ForcePublisher(Node):
    """ROS2 node that publishes load cell force data"""

    def __init__(self):
        super().__init__('force_publisher')
        # Publish individual and total forces
        self.pub_shear = self.create_publisher(Float64, '/force_shear', 10)
        self.pub_normal = self.create_publisher(Float64, '/force_normal', 10)
        self.pub_total = self.create_publisher(Float64, '/force', 10)

        # Initialize serial connection with zero timeout for truly non-blocking reads
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0)
        time.sleep(3)  # Wait for Arduino to initialize
        self.ser.reset_input_buffer()  # Clear any stale data

        # Buffer for incomplete lines
        self.line_buffer = ""

        # Diagnostics
        self.last_publish_time = time.time()
        self.max_buffer_size = 0

        # Create timer for non-blocking serial reads (200 Hz - faster than Arduino's 100 Hz)
        self.timer = self.create_timer(0.005, self.timer_callback)

        self.get_logger().info('Force publisher node started')
        self.get_logger().info('Waiting for Arduino data...')

    def publish_forces(self, force_shear, force_normal):
        """Publish shear, normal, and total force values"""
        # Compute total force magnitude
        force_total = (force_shear**2 + force_normal**2)**0.5

        # Publish all three
        msg_shear = Float64()
        msg_shear.data = - force_shear
        self.pub_shear.publish(msg_shear)

        msg_normal = Float64()
        msg_normal.data = force_normal
        self.pub_normal.publish(msg_normal)

        msg_total = Float64()
        msg_total.data = force_total
        self.pub_total.publish(msg_total)

    def timer_callback(self):
        """Non-blocking timer callback to read serial data - ALWAYS reads freshest data only"""
        try:
            bytes_waiting = self.ser.in_waiting
            if bytes_waiting > 0:
                # FLUSH old buffered data - we only want the CURRENT value
                # Read and discard everything except the most recent data
                if bytes_waiting > 100:  # If buffer is accumulating, it means Arduino was blocked
                    self.get_logger().warning(f"Flushing {bytes_waiting} bytes of stale data")
                    # Read everything to clear the buffer
                    chunk = self.ser.read(bytes_waiting).decode('utf-8', errors='ignore')
                    self.line_buffer = chunk  # Replace buffer entirely with fresh data
                else:
                    # Normal operation - accumulate
                    chunk = self.ser.read(bytes_waiting).decode('utf-8', errors='ignore')
                    self.line_buffer += chunk

                # Extract ONLY the most recent complete line
                if '\n' in self.line_buffer:
                    # Find the LAST complete line (most recent data)
                    last_newline_idx = self.line_buffer.rfind('\n')

                    # Check if there's another newline before it (to get complete line)
                    prev_newline_idx = self.line_buffer.rfind('\n', 0, last_newline_idx)

                    if prev_newline_idx != -1:
                        # Extract the last complete line
                        latest_line = self.line_buffer[prev_newline_idx + 1:last_newline_idx].strip()
                    else:
                        # Only one complete line - use everything before last newline
                        latest_line = self.line_buffer[:last_newline_idx].strip()

                    # Keep only data after the last newline (incomplete line for next iteration)
                    self.line_buffer = self.line_buffer[last_newline_idx + 1:]

                    # Parse and publish ONLY the freshest data
                    if latest_line and ',' in latest_line:
                        # Handle potential negative numbers
                        if latest_line[0].isdigit() or latest_line[0] == '-':
                            parts = latest_line.split(',')
                            if len(parts) == 2:
                                try:
                                    force_shear = float(parts[0])
                                    force_normal = float(parts[1])

                                    # Publish all forces
                                    self.publish_forces(force_shear, force_normal)

                                    # Track publish rate
                                    now = time.time()
                                    dt = now - self.last_publish_time
                                    if dt > 0.2:  # More than 200ms between publishes
                                        self.get_logger().warning(f"Slow publish rate: {dt*1000:.1f} ms gap (Arduino blocking)")
                                    self.last_publish_time = now

                                except ValueError:
                                    self.get_logger().warning(f"Invalid data: {latest_line}")

                # Hard limit on buffer size - flush completely if exceeded
                if len(self.line_buffer) > 200:
                    self.get_logger().error("Buffer overflow - HARD FLUSH")
                    self.line_buffer = ""
                    self.ser.reset_input_buffer()

        except Exception as e:
            self.get_logger().error(f"Serial read error: {e}")

    def destroy_node(self):
        """Clean up serial connection"""
        self.ser.close()
        super().destroy_node()

if __name__ == "__main__":
    rclpy.init()
    force_publisher = ForcePublisher()

    try:
        rclpy.spin(force_publisher)
    except KeyboardInterrupt:
        force_publisher.get_logger().info("Shutting down force publisher")
    finally:
        force_publisher.destroy_node()
        rclpy.shutdown()
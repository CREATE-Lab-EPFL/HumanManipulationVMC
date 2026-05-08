import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import sys
import tty
import termios
import threading

# In another terminal:
# ros2 run dynamixel_interface dynamixel_node --ros-args -p control_mode:=position
# Run DynamixelPositionController

# To increase Dynamixel communication frequency:
# sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer

class DynamixelPositionController(Node):
    def __init__(self):
        super().__init__('dynamixel_position_controller')
        
        self.position_pub = self.create_publisher(Float64MultiArray, '/goal_position', 10)
        self.position_sub = self.create_subscription(
            Float64MultiArray, '/joint_positions', self.position_callback, 10)
        
        self.current_positions = [0.0, 0.0]
        self.target_positions = [0.0, 0.0]
        
    def position_callback(self, msg):
        self.current_positions = list(msg.data)
        
    def send_position(self, positions):
        msg = Float64MultiArray()
        msg.data = positions
        self.position_pub.publish(msg)
        
    def increment_motor(self, motor_idx, delta):
        self.target_positions[motor_idx] += delta
        self.send_position(self.target_positions)
        self.get_logger().info(f'Target: {self.target_positions}, Current: {self.current_positions}')

def get_key():
    """
    Reads a single keypress from the user without requiring them to press Enter.
    """

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def get_motor_index():
    """
    Read motor index (supports 0-1, press Enter to confirm).
    """

    print("Enter motor index (0, 1), press Enter to confirm: ", end='', flush=True)
    digit = ""
    while True:
        key = get_key()
        if key == '\r' or key == '\n':  # Enter key
            break
        elif key == '\x7f' or key == '\x08':  # Backspace
            if digit:
                digit = ""
                print(f"\rEnter motor index (0, 1), press Enter to confirm:  ", end='', flush=True)
        elif key in ['0', '1']:  # Only accept 0 or 1
            digit = key
            print(f"\rEnter motor index (0, 1), press Enter to confirm: {digit}", end='', flush=True)
        elif key == '\x03':  # Ctrl-C
            raise KeyboardInterrupt
    print()  # newline
    return int(digit) if digit else 0

def keyboard_thread_position(controller):
    """
    First chooses motor index to control, then uses 'q'/'a' to increase/decrease position.
    """

    print("\nKeyboard Control (Position Mode):")
    motor_idx = get_motor_index()
    print(f"Controlling motor {motor_idx}. Use 'q' to increase position, 'a' to decrease position, 'm' to change motor.")

    while rclpy.ok():
        key = get_key()
        if key == 'q':
            controller.increment_motor(motor_idx, 1.0)
        elif key == 'a':
            controller.increment_motor(motor_idx, -1.0)
        elif key == 'm':
            motor_idx = get_motor_index()
            print(f"Controlling motor {motor_idx}. Use 'q' to increase position, 'a' to decrease position, 'm' to change motor.")
        elif key == '\x03':  # Ctrl-C
            break

def main_position():
    rclpy.init()
    controller = DynamixelPositionController()

    kb_thread = threading.Thread(target=keyboard_thread_position, args=(controller,))
    kb_thread.start()

    rclpy.spin(controller)

    controller.destroy_node()
    rclpy.shutdown()
    kb_thread.join()

if __name__ == '__main__':
    main_position()
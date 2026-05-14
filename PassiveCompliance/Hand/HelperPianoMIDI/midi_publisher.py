"""
MIDI keyboard → ROS2 publisher.

Reads live MIDI events from a physical keyboard connected via USB/MIDI and
publishes them on two topics:

  /midi/note_on   (std_msgs/String)  — "note=<0-127> velocity=<0-127>"
  /midi/note_off  (std_msgs/String)  — "note=<0-127>"
  /midi/control   (std_msgs/String)  — "cc=<0-127> value=<0-127>"

Usage
-----
    python3 midi_publisher.py [--port <partial-name>]

If --port is omitted the script lists available ports and picks the first one.
Requires: rclpy, mido, python-rtmidi
    pip install mido python-rtmidi
"""

import argparse
import sys

import mido
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MidiPublisher(Node):
    def __init__(self, port_name: str | None):
        super().__init__('midi_publisher')

        self._pub_on  = self.create_publisher(String, '/midi/note_on',  10)
        self._pub_off = self.create_publisher(String, '/midi/note_off', 10)
        self._pub_cc  = self.create_publisher(String, '/midi/control',  10)

        available = mido.get_input_names()
        if not available:
            self.get_logger().fatal('No MIDI input ports found. Connect your keyboard and retry.')
            raise SystemExit(1)

        if port_name is None:
            chosen = available[0]
            self.get_logger().info(f'Available ports: {available}')
            self.get_logger().info(f'Using first port: "{chosen}"')
        else:
            matches = [p for p in available if port_name.lower() in p.lower()]
            if not matches:
                self.get_logger().fatal(
                    f'No port matching "{port_name}". Available: {available}'
                )
                raise SystemExit(1)
            chosen = matches[0]
            self.get_logger().info(f'Using port: "{chosen}"')

        self._port = mido.open_input(chosen, callback=self._midi_callback)
        self.get_logger().info('Listening for MIDI events — press Ctrl+C to stop.')

    def _midi_callback(self, msg: mido.Message) -> None:
        if msg.type == 'note_on' and msg.velocity > 0:
            out = String()
            out.data = f'note={msg.note} velocity={msg.velocity}'
            self._pub_on.publish(out)
            self.get_logger().debug(f'note_on  {out.data}')

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            out = String()
            out.data = f'note={msg.note}'
            self._pub_off.publish(out)
            self.get_logger().debug(f'note_off {out.data}')

        elif msg.type == 'control_change':
            out = String()
            out.data = f'cc={msg.control} value={msg.value}'
            self._pub_cc.publish(out)
            self.get_logger().debug(f'control  {out.data}')

    def destroy_node(self):
        self._port.close()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description='MIDI keyboard → ROS2 publisher')
    parser.add_argument('--port', default=None, help='Partial name of the MIDI input port')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = MidiPublisher(args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
ROS2 MIDI subscriber — prints note/control events to the terminal.

Subscribes to:
  /midi/note_on   (std_msgs/String)
  /midi/note_off  (std_msgs/String)
  /midi/control   (std_msgs/String)

Usage
-----
    python3 midi_subscriber.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_note_to_name(note: int) -> str:
    return f'{NOTE_NAMES[note % 12]}{note // 12 - 1}'


class MidiSubscriber(Node):
    def __init__(self):
        super().__init__('midi_subscriber')

        self.create_subscription(String, '/midi/note_on',  self._on_note_on,  10)
        self.create_subscription(String, '/midi/note_off', self._on_note_off, 10)
        self.create_subscription(String, '/midi/control',  self._on_control,  10)

        self.get_logger().info('Subscribed to /midi/note_on, /midi/note_off, /midi/control')

    def _parse(self, data: str) -> dict:
        return dict(pair.split('=') for pair in data.split())

    def _on_note_on(self, msg: String) -> None:
        fields = self._parse(msg.data)
        note     = int(fields['note'])
        velocity = int(fields['velocity'])
        name     = midi_note_to_name(note)
        print(f'[NOTE ON ] {name:4s} (MIDI {note:3d})  velocity={velocity:3d}')

    def _on_note_off(self, msg: String) -> None:
        fields = self._parse(msg.data)
        note = int(fields['note'])
        name = midi_note_to_name(note)
        print(f'[NOTE OFF] {name:4s} (MIDI {note:3d})')

    def _on_control(self, msg: String) -> None:
        fields = self._parse(msg.data)
        cc    = int(fields['cc'])
        value = int(fields['value'])
        print(f'[CONTROL ] CC {cc:3d}  value={value:3d}')


def main():
    rclpy.init()
    node = MidiSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

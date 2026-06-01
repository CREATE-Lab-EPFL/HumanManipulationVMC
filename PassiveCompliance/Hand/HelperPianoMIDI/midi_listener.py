"""
Standalone MIDI listener — prints note and CC events to the terminal.

Replaces the old midi_publisher.py + midi_subscriber.py pair.
No ROS2 required.

Usage:
    python3 midi_listener.py                    # auto-selects first port
    python3 midi_listener.py --port Arturia     # partial name match
    python3 midi_listener.py --list             # list available ports
"""

import argparse
import time
from midi_controller import MidiController

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note_name(n: int) -> str:
    return f'{NOTE_NAMES[n % 12]}{n // 12 - 1}'


def main():
    parser = argparse.ArgumentParser(description='MIDI terminal listener')
    parser.add_argument('--port', default=None, help='Partial port name')
    parser.add_argument('--list', action='store_true', help='List ports and exit')
    args = parser.parse_args()

    if args.list:
        ports = MidiController.list_ports()
        if ports:
            for p in ports:
                print(p)
        else:
            print('No MIDI ports found.')
        return

    def on_note_on(note: int, velocity: int) -> None:
        print(f'[NOTE ON ] {note_name(note):4s} (MIDI {note:3d})  velocity={velocity:3d}')

    def on_note_off(note: int) -> None:
        print(f'[NOTE OFF] {note_name(note):4s} (MIDI {note:3d})')

    with MidiController(port_name=args.port,
                        on_note_on=on_note_on, on_note_off=on_note_off) as ctrl:
        print(f'Listening on: "{ctrl.port_name}"  —  press Ctrl+C to stop.')
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()

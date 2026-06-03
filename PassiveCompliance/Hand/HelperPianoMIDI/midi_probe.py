"""
Raw MIDI probe — diagnose whether a keyboard reports real touch dynamics.

Unlike midi_listener.py (which only decodes note-on/off), this dumps EVERY
incoming MIDI message with its raw bytes, and keeps a running summary of the
distinct note-on velocities seen. Use it to answer one question:

    "Does my keyboard actually vary velocity, or send a fixed value?"

Usage:
    python3 midi_probe.py                 # auto-select first port
    python3 midi_probe.py --port Arturia  # partial name match
    python3 midi_probe.py --list

Then press a key by hand: first as gently as possible, then as hard/fast as
you can, a few times each.

  * Velocities spread out (e.g. 20 ... 110)  -> keyboard IS velocity-sensitive;
    the flat 8 in the experiment is mechanical (presses too slow/soft) — raise
    PRESS_ACCEL and check the strike actually moves the key.
  * Velocity stays fixed (always 8)          -> keyboard is NOT sending touch
    velocity (fixed-velocity / non-weighted, or a 'velocity curve = fixed'
    setting). No code/accel change will help — change the keyboard's velocity
    setting or use a velocity-sensitive one.
  * Note-on is fixed BUT you see 0xA0/0xD0 (aftertouch) or 0xB0 (CC) messages
    change as you press harder -> dynamics live in aftertouch/CC, not velocity;
    capture that channel instead.
"""

import argparse
import time
import rtmidi

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _note_name(n: int) -> str:
    return f'{_NOTE_NAMES[n % 12]}{n // 12 - 1}'


def _decode(msg):
    """Human-readable label for a raw MIDI message."""
    if not msg:
        return 'empty'
    status = msg[0] & 0xF0
    ch = (msg[0] & 0x0F) + 1
    if status == 0x90 and len(msg) >= 3 and msg[2] > 0:
        return f'NOTE ON  ch{ch}  {_note_name(msg[1]):4s}  vel={msg[2]:3d}'
    if status == 0x80 or (status == 0x90 and len(msg) >= 3 and msg[2] == 0):
        return f'NOTE OFF ch{ch}  {_note_name(msg[1]):4s}'
    if status == 0xA0:
        return f'POLY AFTERTOUCH ch{ch}  {_note_name(msg[1]):4s}  pressure={msg[2]:3d}'
    if status == 0xD0:
        return f'CHANNEL AFTERTOUCH ch{ch}  pressure={msg[1]:3d}'
    if status == 0xB0:
        return f'CONTROL CHANGE ch{ch}  cc={msg[1]:3d}  val={msg[2]:3d}'
    if status == 0xE0:
        return f'PITCH BEND ch{ch}'
    return f'status=0x{msg[0]:02X}'


def main():
    parser = argparse.ArgumentParser(description='Raw MIDI probe / velocity diagnostic')
    parser.add_argument('--port', default=None, help='Partial port name')
    parser.add_argument('--list', action='store_true', help='List ports and exit')
    args = parser.parse_args()

    midi_in = rtmidi.MidiIn()
    ports = midi_in.get_ports()
    if args.list:
        print('\n'.join(ports) if ports else 'No MIDI ports found.')
        return
    if not ports:
        print('No MIDI input ports found. Connect a device and try again.')
        return

    # pick port (skip ALSA "Through" loopback unless it is the only one)
    idx = 0
    if args.port:
        idx = next((i for i, p in enumerate(ports) if args.port.lower() in p.lower()), 0)
    else:
        idx = next((i for i, p in enumerate(ports) if 'through' not in p.lower()), 0)
    midi_in.open_port(idx)
    # IMPORTANT: do NOT ignore aftertouch — we want to see it if present
    midi_in.ignore_types(sysex=True, timing=True, active_sense=True)
    print(f'Probing: "{ports[idx]}"  —  press keys (soft, then hard). Ctrl+C to stop.\n')

    velocities = set()
    vmin, vmax = 127, 0
    try:
        while True:
            ev = midi_in.get_message()
            if ev is None:
                time.sleep(0.001)
                continue
            msg, _dt = ev
            label = _decode(msg)
            raw = ' '.join(f'{b:3d}' for b in msg)
            print(f'  [{raw:<12}]  {label}')
            if (msg[0] & 0xF0) == 0x90 and len(msg) >= 3 and msg[2] > 0:
                v = msg[2]
                velocities.add(v)
                vmin, vmax = min(vmin, v), max(vmax, v)
                print(f'      -> velocity range so far: {vmin}..{vmax}   '
                      f'distinct values: {sorted(velocities)}')
    except KeyboardInterrupt:
        print('\n' + '=' * 60)
        if not velocities:
            print('No note-on messages captured.')
        elif len(velocities) == 1:
            print(f'VERDICT: velocity was CONSTANT at {velocities.pop()} — the keyboard '
                  'is not reporting touch velocity. Change its velocity-curve setting '
                  'or use a velocity-sensitive keyboard.')
        else:
            print(f'VERDICT: velocity VARIED ({vmin}..{vmax}) — the keyboard is '
                  'velocity-sensitive. The flat experiment value is mechanical: make '
                  'the robot strike the key faster/harder.')
        print('=' * 60)
    finally:
        midi_in.close_port()


if __name__ == '__main__':
    main()

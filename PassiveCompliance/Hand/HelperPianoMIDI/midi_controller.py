"""
Real-time MIDI controller interface.

Listens to a physical MIDI input device and tracks the pressed/released
state of every note. State updates arrive via an rtmidi callback on a
background thread; reads are always non-blocking.

Typical usage
-------------
    from midi_controller import MidiController

    ctrl = MidiController()          # auto-selects first available port
    # or: ctrl = MidiController("Arturia KeyLab 49")

    pressed = ctrl.get_pressed(midi_min=48, midi_max=83)
    # {48: False, 49: True, ..., 83: False}

    ctrl.close()
    # or: use as a context manager  (with MidiController() as ctrl: ...)

Event callbacks
---------------
    def on_note(note, velocity):
        print(f'note {note} pressed with velocity {velocity}')

    ctrl = MidiController(on_note_on=on_note, on_note_off=lambda n: ...)
"""

import threading
import rtmidi


class MidiController:
    """Non-blocking MIDI input wrapper.

    Opens one MIDI input port and maintains a per-note pressed state,
    updated asynchronously via an rtmidi callback.

    Attributes:
        port_name: Human-readable name of the opened port.
    """

    _NOTE_ON     = 0x90
    _NOTE_OFF    = 0x80
    _STATUS_MASK = 0xF0

    def __init__(self, port_name: str = None,
                 on_note_on=None, on_note_off=None):
        """Open a MIDI input port and start listening.

        Args:
            port_name:   Substring to match against available port names
                         (case-insensitive). If None, the first port is used.
            on_note_on:  Optional callable(note: int, velocity: int) invoked
                         on every note-on event (on the rtmidi background thread).
            on_note_off: Optional callable(note: int) invoked on every note-off.

        Raises:
            RuntimeError: If no MIDI input ports are available.
        """
        self._lock  = threading.Lock()
        self._state: dict[int, bool] = {note: False for note in range(128)}
        self._on_note_on  = on_note_on
        self._on_note_off = on_note_off

        self._midi_in = rtmidi.MidiIn()
        ports = self._midi_in.get_ports()

        if not ports:
            raise RuntimeError(
                'No MIDI input ports found. '
                'Connect a MIDI device and try again.'
            )

        port_index = self._select_port(ports, port_name)
        self._midi_in.open_port(port_index)
        self._midi_in.set_callback(self._callback)
        self._midi_in.ignore_types(sysex=True, timing=True, active_sense=True)

        self.port_name = ports[port_index]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pressed(self, midi_min: int = 0, midi_max: int = 127) -> dict[int, bool]:
        """Return the pressed state of all notes in [midi_min, midi_max]."""
        with self._lock:
            return {note: self._state[note] for note in range(midi_min, midi_max + 1)}

    def is_pressed(self, midi_note: int) -> bool:
        """Return whether a single note is currently pressed."""
        with self._lock:
            return self._state.get(midi_note, False)

    def reset(self):
        """Mark all notes as released (useful after a pause/reconnect)."""
        with self._lock:
            for note in range(128):
                self._state[note] = False

    def close(self):
        """Release the MIDI port."""
        self._midi_in.close_port()

    @staticmethod
    def list_ports() -> list[str]:
        """Return the names of all available MIDI input ports."""
        tmp = rtmidi.MidiIn()
        ports = tmp.get_ports()
        del tmp
        return ports

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_port(ports: list[str], name_hint: str | None) -> int:
        if name_hint is not None:
            hint = name_hint.lower()
            for i, p in enumerate(ports):
                if hint in p.lower():
                    return i
        return 0

    def _callback(self, event, _data):
        message, _delta_time = event
        if len(message) < 3:
            return

        status   = message[0] & self._STATUS_MASK
        note     = message[1]
        velocity = message[2]

        if status == self._NOTE_ON and velocity > 0:
            with self._lock:
                self._state[note] = True
            if self._on_note_on is not None:
                self._on_note_on(note, velocity)
        elif status == self._NOTE_OFF or (status == self._NOTE_ON and velocity == 0):
            with self._lock:
                self._state[note] = False
            if self._on_note_off is not None:
                self._on_note_off(note)

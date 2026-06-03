"""
Real-time microphone controller for the guitar experiment.

Streams audio from an input device on a background thread and detects pluck
onsets (the audio analogue of a MIDI note-on), reporting an intensity for each.
Reads are always non-blocking — this mirrors HelperPianoMIDI/midi_controller.py.

Typical usage
-------------
    from mic_controller import MicrophoneController

    def on_pluck(intensity, t):
        print(f'pluck  intensity={intensity:.3f}  at t={t:.2f}s')

    mic = MicrophoneController(on_onset=on_pluck)     # default input device
    # or: mic = MicrophoneController("USB", on_onset=on_pluck)  # name substring
    level = mic.get_level()                            # current RMS, non-blocking
    mic.close()
    # or use as a context manager:  with MicrophoneController() as mic: ...

Onset detection
---------------
Simple energy-based detector: each audio block's RMS is compared to
`onset_threshold`; a rising edge (with a refractory gap and hysteresis) is a
pluck. The reported intensity is the block's peak amplitude (0..1) — the
loudness proxy, analogous to MIDI note-on velocity.

Requires the `sounddevice` package (PortAudio backend):
    pip install sounddevice
"""

import threading
import time

import numpy as np

try:
    import sounddevice as _sd
except Exception as _exc:          # pragma: no cover - import guard
    _sd = None
    _SD_IMPORT_ERROR = _exc


class MicrophoneController:
    """Non-blocking microphone capture with energy-based onset detection.

    Opens one audio input stream and runs an RMS pluck detector on every
    incoming block. Onsets are delivered via an optional callback and also kept
    in a thread-safe list; the latest level is always readable.

    Attributes:
        samplerate, channels, blocksize: stream configuration.
        device_name: human-readable name of the opened input device.
        onsets: list of (time_s, intensity) for every detected pluck.
    """

    def __init__(self, port_name=None, on_onset=None, *,
                 samplerate: int = 44100, channels: int = 1,
                 blocksize: int = 1024, onset_threshold: float = 0.05,
                 refractory: float = 0.10):
        """Open a microphone input stream and start listening.

        Args:
            port_name:       Substring to match against input device names
                             (case-insensitive). If None, the default device is used.
            on_onset:        Optional callable(intensity: float, time_s: float)
                             invoked on every detected pluck (on the audio thread).
            samplerate:      Sample rate [Hz].
            channels:        Number of input channels (only channel 0 is analysed).
            blocksize:       Frames per callback block.
            onset_threshold: RMS level (0..1) above which a pluck is registered.
            refractory:      Minimum seconds between two detected onsets.

        Raises:
            ImportError:  If the sounddevice package is not installed.
            RuntimeError: If no audio input device is available.
        """
        if _sd is None:
            raise ImportError(
                'sounddevice is required for MicrophoneController. '
                'Install it with:  pip install sounddevice') from _SD_IMPORT_ERROR

        self.samplerate      = samplerate
        self.channels        = channels
        self.blocksize       = blocksize
        self.onset_threshold = onset_threshold
        self.refractory      = refractory
        self._on_onset       = on_onset

        self._lock       = threading.Lock()
        self._level      = 0.0
        self._t0         = time.time()
        self._last_onset = -1e9
        self._above      = False          # hysteresis state (currently above threshold)
        self.onsets: list[tuple[float, float]] = []

        device = self._select_device(port_name)
        info = _sd.query_devices(device if device is not None else None, 'input')
        if info['max_input_channels'] < 1:
            raise RuntimeError('No audio input device available.')
        self.device_name = info['name']

        self._stream = _sd.InputStream(
            device=device, channels=channels, samplerate=samplerate,
            blocksize=blocksize, callback=self._callback)
        self._stream.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_level(self) -> float:
        """Return the most recent block RMS level (0..1), non-blocking."""
        with self._lock:
            return self._level

    def reset(self) -> None:
        """Clear the onset history (e.g. between experiment runs)."""
        with self._lock:
            self.onsets.clear()
            self._above = False

    def close(self) -> None:
        """Stop and release the audio stream."""
        self._stream.stop()
        self._stream.close()

    @staticmethod
    def list_devices() -> list[str]:
        """Return the names of all available audio input devices."""
        if _sd is None:
            return []
        return [d['name'] for d in _sd.query_devices() if d['max_input_channels'] > 0]

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
    def _select_device(name_hint):
        if _sd is None or name_hint is None:
            return None
        hint = str(name_hint).lower()
        for i, d in enumerate(_sd.query_devices()):
            if d['max_input_channels'] > 0 and hint in d['name'].lower():
                return i
        return None

    def _callback(self, indata, frames, time_info, status):
        x = indata[:, 0] if indata.ndim > 1 else indata
        if x.size == 0:
            return
        x = x.astype(np.float64)
        rms  = float(np.sqrt(np.mean(x * x)))
        peak = float(np.max(np.abs(x)))
        now  = time.time() - self._t0

        callback = None
        with self._lock:
            self._level = rms
            if (not self._above and rms >= self.onset_threshold
                    and (now - self._last_onset) >= self.refractory):
                # rising edge -> new pluck onset
                self._above = True
                self._last_onset = now
                self.onsets.append((now, peak))
                callback = self._on_onset
            elif rms < 0.5 * self.onset_threshold:
                self._above = False          # reset hysteresis once the sound decays

        if callback is not None:
            callback(peak, now)


# ──────────────────────────────────────────────────────────────────────────────
# CLI — quick mic check (list devices, or print live levels / pluck onsets)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Microphone onset probe')
    parser.add_argument('--device', default=None, help='Partial input device name')
    parser.add_argument('--list', action='store_true', help='List input devices and exit')
    parser.add_argument('--threshold', type=float, default=0.05, help='Onset RMS threshold')
    args = parser.parse_args()

    if args.list:
        devs = MicrophoneController.list_devices()
        print('\n'.join(devs) if devs else 'No audio input devices found.')
        return

    def on_pluck(intensity, t):
        print(f'[PLUCK] intensity={intensity:6.3f}  t={t:6.2f}s')

    with MicrophoneController(args.device, on_onset=on_pluck,
                              onset_threshold=args.threshold) as mic:
        print(f'Listening on: "{mic.device_name}"  —  pluck a string. Ctrl+C to stop.')
        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()

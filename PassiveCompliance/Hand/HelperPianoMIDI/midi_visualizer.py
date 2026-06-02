"""
Live MIDI piano-roll visualizer.

    python3 midi_visualizer.py                         # auto-selects first port
    python3 midi_visualizer.py --port Arturia          # partial port name
    python3 midi_visualizer.py --midi-min 36 --midi-max 96
    python3 midi_visualizer.py --light                 # white theme for paper figures
    python3 midi_visualizer.py --persist               # auto-reconnect on device loss
    python3 midi_visualizer.py --verbose               # also print events to terminal
    python3 midi_visualizer.py --list                  # list available ports

This script opens its own MIDI connection — do NOT run midi_listener.py at
the same time or ALSA will split events between the two processes.

Keys (while the window is open):
    s — save current frame as PNG + PDF (paper-quality, 300 DPI)
    q — quit
"""

import argparse
import collections
import os
import sys
import time
import threading

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from midi_controller import MidiController

# ── Timing / layout ───────────────────────────────────────────────────────────
ROLL_WINDOW  = 8.0   # seconds visible in the scrolling piano roll
FPS          = 25
SAVE_DPI     = 300

# ── Themes ────────────────────────────────────────────────────────────────────
_DARK = dict(
    bg='#0d1117', panel='#161b22',
    white_key='#c8ccd0', black_key='#1a1f27',
    grid_oct='#2d3339', grid_minor='#1e232a',
    text='#8b949e', label='#c9d1d9',
    cmap=plt.cm.plasma,
    roll_alpha=0.88,
)
_LIGHT = dict(
    bg='#ffffff', panel='#f6f8fa',
    white_key='#ffffff', black_key='#2a2a2a',
    grid_oct='#d0d7de', grid_minor='#eaeef2',
    text='#57606a', label='#24292f',
    cmap=plt.cm.viridis,
    roll_alpha=0.80,
)

# ── Note helpers ──────────────────────────────────────────────────────────────
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_WHITE_SET  = {0, 2, 4, 5, 7, 9, 11}


def is_white(n: int) -> bool:
    return n % 12 in _WHITE_SET


def note_name(n: int) -> str:
    return f'{_NOTE_NAMES[n % 12]}{n // 12 - 1}'


# ── Key position layout ───────────────────────────────────────────────────────
def _build_key_positions(midi_min: int, midi_max: int):
    """Map each note to an x position in the keyboard panel."""
    white_x, black_x = {}, {}
    wx = 0
    for note in range(midi_min, midi_max + 1):
        if is_white(note):
            white_x[note] = wx
            wx += 1
        else:
            black_x[note] = wx - 0.5  # centred between last and next white
    return white_x, black_x, wx   # wx = total white key count


# ── Thread-safe note event store ──────────────────────────────────────────────
class _NoteEvent:
    __slots__ = ('note', 'velocity', 'start', 'end')

    def __init__(self, note, velocity, start):
        self.note, self.velocity = note, velocity
        self.start = start
        self.end   = None   # None while still pressed


_events_lock = threading.Lock()
_active: dict          = {}                      # note → _NoteEvent
_finished              = collections.deque(maxlen=3000)
_t0                    = time.time()
_verbose               = False   # set by run_visualizer

_NOTE_NAMES_FULL = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _note_name_full(n: int) -> str:
    return f'{_NOTE_NAMES_FULL[n % 12]}{n // 12 - 1}'


def _on_note_on(note: int, velocity: int) -> None:
    with _events_lock:
        _active[note] = _NoteEvent(note, velocity, time.time() - _t0)
    if _verbose:
        print(f'[NOTE ON ] {_note_name_full(note):4s} (MIDI {note:3d})  velocity={velocity:3d}')


def _on_note_off(note: int) -> None:
    t = time.time() - _t0
    with _events_lock:
        if note in _active:
            ev = _active.pop(note)
            ev.end = t
            _finished.append(ev)
    if _verbose:
        print(f'[NOTE OFF] {_note_name_full(note):4s} (MIDI {note:3d})')


# ── Figure builder ────────────────────────────────────────────────────────────
def _build_figure(midi_min: int, midi_max: int, theme: dict):
    white_x, black_x, n_whites = _build_key_positions(midi_min, midi_max)

    fig = plt.figure(figsize=(15, 8), facecolor=theme['bg'])
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1],
                           hspace=0.0, left=0.07, right=0.985,
                           top=0.96, bottom=0.03)
    ax_roll = fig.add_subplot(gs[0])
    ax_keys = fig.add_subplot(gs[1])

    # ── Piano roll ────────────────────────────────────────────────────────────
    ax_roll.set_facecolor(theme['panel'])
    for spine in ax_roll.spines.values():
        spine.set_color(theme['grid_oct'])
    ax_roll.tick_params(colors=theme['text'], labelsize=8)
    ax_roll.set_ylim(midi_min - 0.5, midi_max + 0.5)
    ax_roll.set_ylabel('Note', color=theme['text'], fontsize=9)

    for note in range(midi_min, midi_max + 1):
        if note % 12 == 0:
            ax_roll.axhline(note, color=theme['grid_oct'],   linewidth=0.6, zorder=0)
        else:
            ax_roll.axhline(note, color=theme['grid_minor'],  linewidth=0.2, zorder=0)

    c_notes = [n for n in range(midi_min, midi_max + 1) if n % 12 == 0]
    ax_roll.set_yticks(c_notes)
    ax_roll.set_yticklabels([note_name(n) for n in c_notes],
                            color=theme['text'], fontsize=8)

    # ── Keyboard panel ────────────────────────────────────────────────────────
    ax_keys.set_facecolor(theme['black_key'])
    ax_keys.set_xlim(-0.5, n_whites - 0.5)
    ax_keys.set_ylim(0, 1)
    ax_keys.axis('off')

    key_rects: dict[int, mpatches.FancyBboxPatch] = {}

    for note, x in white_x.items():
        rect = mpatches.FancyBboxPatch(
            (x - 0.45, 0.03), 0.90, 0.95,
            boxstyle='round,pad=0.01',
            facecolor=theme['white_key'],
            edgecolor=theme['grid_oct'], linewidth=0.8, zorder=1)
        ax_keys.add_patch(rect)
        key_rects[note] = rect
        if note % 12 == 0:
            ax_keys.text(x, 0.07, note_name(note), ha='center', va='bottom',
                         fontsize=6, color=theme['text'], zorder=5)

    for note, x in black_x.items():
        rect = mpatches.FancyBboxPatch(
            (x - 0.28, 0.43), 0.56, 0.55,
            boxstyle='round,pad=0.01',
            facecolor=theme['black_key'],
            edgecolor=theme['grid_oct'], linewidth=0.5, zorder=3)
        ax_keys.add_patch(rect)
        key_rects[note] = rect

    return fig, ax_roll, ax_keys, key_rects, white_x, black_x


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_visualizer(midi_min: int, midi_max: int, port_name, save_dir: str,
                   theme: dict, persist: bool, verbose: bool = False) -> None:
    global _verbose
    _verbose = verbose

    # MIDI connection (with optional retry loop)
    midi_ctrl = None
    while midi_ctrl is None:
        try:
            midi_ctrl = MidiController(port_name=port_name,
                                       on_note_on=_on_note_on,
                                       on_note_off=_on_note_off)
        except RuntimeError as exc:
            if not persist:
                raise
            print(f'{exc}  Retrying in 2 s …')
            time.sleep(2.0)

    print(f'Connected: "{midi_ctrl.port_name}"')
    print('Press  s  to save figure    q  to quit.')

    fig, ax_roll, ax_keys, key_rects, white_x, black_x = \
        _build_figure(midi_min, midi_max, theme)

    fig.suptitle(f'MIDI — {midi_ctrl.port_name}',
                 color=theme['label'], fontsize=10, y=0.99)

    cmap        = theme['cmap']
    roll_alpha  = theme['roll_alpha']
    roll_patches: list = []

    def _key_color(velocity: int):
        return cmap(0.15 + 0.85 * velocity / 127)

    def update(_frame):
        nonlocal roll_patches
        now = time.time() - _t0
        ax_roll.set_xlim(now - ROLL_WINDOW, now)

        for p in roll_patches:
            p.remove()
        roll_patches.clear()

        with _events_lock:
            finished_snap = list(_finished)
            active_snap   = [(ev.note, ev.velocity, ev.start)
                             for ev in _active.values()]

        cutoff = now - ROLL_WINDOW
        pressed = {note: vel for note, vel, _ in active_snap}

        # Draw note bars in the roll
        def _bar(note, start, end, velocity):
            if end < cutoff or start > now:
                return
            x0 = max(start, cutoff)
            r  = mpatches.Rectangle(
                (x0, note - 0.44), end - x0, 0.88,
                facecolor=_key_color(velocity), edgecolor='none',
                linewidth=0, alpha=roll_alpha, zorder=2)
            ax_roll.add_patch(r)
            roll_patches.append(r)

        for ev in finished_snap:
            _bar(ev.note, ev.start, ev.end, ev.velocity)
        for note, vel, start in active_snap:
            _bar(note, start, now, vel)

        # Update keyboard colours
        for note, rect in key_rects.items():
            if note in pressed:
                col = _key_color(pressed[note])
                rect.set_facecolor(col)
                rect.set_zorder(4 if not is_white(note) else 2)
            else:
                rect.set_facecolor(
                    theme['white_key'] if is_white(note) else theme['black_key'])
                rect.set_zorder(1 if is_white(note) else 3)

    _save_idx = [0]

    def _on_key(event):
        if event.key == 'q':
            plt.close('all')
        elif event.key == 's':
            _save_idx[0] += 1
            stem = os.path.join(save_dir, f'midi_snapshot_{_save_idx[0]:03d}')
            fig.savefig(stem + '.png', dpi=SAVE_DPI, bbox_inches='tight',
                        facecolor=theme['bg'])
            fig.savefig(stem + '.pdf', bbox_inches='tight',
                        facecolor=theme['bg'])
            print(f'Saved  {stem}.png  and  {stem}.pdf')

    fig.canvas.mpl_connect('key_press_event', _on_key)

    plt.ion()
    plt.show(block=False)
    _dt = 1.0 / FPS
    while plt.fignum_exists(fig.number):
        update(None)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(_dt)

    midi_ctrl.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Live MIDI piano-roll visualizer')
    parser.add_argument('--port',     default=None,
                        help='Partial MIDI port name (case-insensitive match)')
    parser.add_argument('--midi-min', type=int, default=28,
                        help='Lowest MIDI note displayed (default 28 = E1)')
    parser.add_argument('--midi-max', type=int, default=96,
                        help='Highest MIDI note displayed (default 96 = C7)')
    parser.add_argument('--light',    action='store_true',
                        help='Use light theme (better for paper figures)')
    parser.add_argument('--persist',  action='store_true',
                        help='Keep retrying if no MIDI device is connected')
    parser.add_argument('--save-dir', default='.',
                        help='Directory for saved figures (default: current dir)')
    parser.add_argument('--verbose',  action='store_true',
                        help='Also print note events to terminal (replaces midi_listener.py)')
    parser.add_argument('--list',     action='store_true',
                        help='List available MIDI ports and exit')
    args = parser.parse_args()

    if args.list:
        ports = MidiController.list_ports()
        print('\n'.join(ports) if ports else 'No MIDI ports found.')
        return

    os.makedirs(args.save_dir, exist_ok=True)
    theme = _LIGHT if args.light else _DARK

    run_visualizer(args.midi_min, args.midi_max, args.port,
                   args.save_dir, theme, args.persist, args.verbose)


if __name__ == '__main__':
    main()

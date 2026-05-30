"""
Fusion 360 add-in entry point.

Starts a background thread that fires a custom event every POLL_INTERVAL
seconds. The event handler (running on the Fusion main thread) calls
JointHandler to read the bridge file, apply joint targets, and write back
current values.
"""

import adsk.core
import adsk.fusion
import os
import sys
import threading
import traceback

# Make sibling modules importable regardless of Fusion's working directory.
_addin_dir = os.path.dirname(os.path.realpath(__file__))
if _addin_dir not in sys.path:
    sys.path.insert(0, _addin_dir)

import JointHandler

POLL_INTERVAL   = 0.2          # seconds between bridge file polls
CUSTOM_EVENT_ID = "FusionJointBridgeEvent"

_app       = None
_ui        = None
_handlers  = []
_stop_flag = threading.Event()
_thread    = None


# ---------------------------------------------------------------------------
# Custom-event handler (runs on the Fusion main thread)
# ---------------------------------------------------------------------------

class _BridgeEventHandler(adsk.core.CustomEventHandler):
    def notify(self, args):
        try:
            JointHandler.apply_and_readback(_app)
        except Exception:
            pass   # errors are logged inside apply_and_readback


# ---------------------------------------------------------------------------
# Background polling thread
# ---------------------------------------------------------------------------

def _poll_loop() -> None:
    while not _stop_flag.wait(POLL_INTERVAL):
        try:
            _app.fireCustomEvent(CUSTOM_EVENT_ID)
        except Exception:
            break   # app is shutting down


# ---------------------------------------------------------------------------
# Add-in lifecycle
# ---------------------------------------------------------------------------

def run(context):
    global _app, _ui, _thread

    _app = adsk.core.Application.get()
    _ui  = _app.userInterface

    try:
        event   = _app.registerCustomEvent(CUSTOM_EVENT_ID)
        handler = _BridgeEventHandler()
        event.add(handler)
        _handlers.append(handler)

        _stop_flag.clear()
        _thread = threading.Thread(target=_poll_loop, daemon=True)
        _thread.start()

        _ui.messageBox(
            "FusionJointBridge started.\n"
            f"Polling every {int(POLL_INTERVAL * 1000)} ms.\n"
            f"Bridge file: {JointHandler.BRIDGE_PATH}"
        )

    except Exception:
        if _ui:
            _ui.messageBox(traceback.format_exc())


def stop(context):
    global _thread

    _stop_flag.set()
    if _thread:
        _thread.join(timeout=1.0)
        _thread = None

    try:
        _app.unregisterCustomEvent(CUSTOM_EVENT_ID)
    except Exception:
        pass

    _handlers.clear()

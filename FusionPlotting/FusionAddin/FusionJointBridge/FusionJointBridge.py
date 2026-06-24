"""
Fusion 360 add-in entry point.

On run():
  1. Discovers all joints in the active design → ~/FusionBridge/joints_discovery.json
  2. Starts a background thread that fires a custom event every POLL_INTERVAL seconds.
     The event handler applies joint targets from the bridge file only when the file
     has actually changed since the last apply (cheap no-op otherwise).
"""

import adsk.core
import adsk.fusion
import os
import sys
import threading
import traceback

_addin_dir = os.path.dirname(os.path.realpath(__file__))
if _addin_dir not in sys.path:
    sys.path.insert(0, _addin_dir)

import JointHandler

POLL_INTERVAL   = 2.0   # seconds — only applies when bridge file changed, so low rate is fine
CUSTOM_EVENT_ID = "FusionJointBridgeEvent"

_app       = None
_ui        = None
_handlers  = []
_stop_flag = threading.Event()
_thread    = None


class _BridgeEventHandler(adsk.core.CustomEventHandler):
    def notify(self, args):
        try:
            JointHandler.apply_and_readback(_app)
        except Exception:
            pass


def _poll_loop() -> None:
    while not _stop_flag.wait(POLL_INTERVAL):
        try:
            _app.fireCustomEvent(CUSTOM_EVENT_ID)
        except Exception:
            break


def run(context):
    global _app, _ui, _thread

    _app = adsk.core.Application.get()
    _ui  = _app.userInterface

    try:
        JointHandler.discover_all_joints(_app)

        event   = _app.registerCustomEvent(CUSTOM_EVENT_ID)
        handler = _BridgeEventHandler()
        event.add(handler)
        _handlers.append(handler)

        _stop_flag.clear()
        _thread = threading.Thread(target=_poll_loop, daemon=True)
        _thread.start()

        _ui.messageBox(
            "FusionJointBridge started.\n\n"
            f"Polling every {int(POLL_INTERVAL)} s (applies only on new commands).\n\n"
            f"Bridge file:    {JointHandler.BRIDGE_PATH}\n"
            f"Discovery file: {JointHandler.DISCOVERY_PATH}\n"
            f"Log file:       {JointHandler.LOG_PATH}"
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

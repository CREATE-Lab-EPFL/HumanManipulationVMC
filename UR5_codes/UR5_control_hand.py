"""
UR5_control_hand.py
===================
Move the UR5 (and only the UR5) to the initial pose of a selected hand
experiment. Run the script, choose an experiment from the terminal menu,
and the arm performs a linear move to that pose.

Poses are imported directly from the original experiment configs (no
duplication), so any change upstream is picked up automatically. importlib
is used because (a) ADAPT-StiffControl contains a hyphen, which isn't a
valid Python identifier, and (b) several experiments share the module
name `hand_config.py`, so a plain sys.path import would collide.
"""

import importlib.util
from pathlib import Path

import numpy as np
import rtde_control
import rtde_receive

from UR5_config import UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCELERATION


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(module_alias, relative_path):
    """Load a module from a file path under REPO_ROOT, with a unique alias."""
    full_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_alias, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_piano    = _load("piano_config_hand",
                  "PassiveCompliance/Hand/HelperPianoMIDI/piano_config.py")
_tunable  = _load("tunable_hand_config",
                  "TunableCompliance/Hand/hand_config.py")
_proprio  = _load("proprio_hand_config",
                  "ProprioceptiveSensing/Hand/hand_config.py")
_adapt    = _load("adapt_hand_config",
                  "ADAPT-StiffControl/hand_config.py")


# ── Hand-experiment initial poses (imported from source configs) ─────────────
# Key = experiment script name; value = pose used by that experiment.
HAND_POSES = {
    # PassiveCompliance/Hand/piano_playing_hand.py
    "piano_playing_hand":     _piano.UR5_POSE_PIANO,
    # PassiveCompliance/Hand/piano_glissando.py
    "piano_glissando":        _piano.UR5_POSE_GLISSANDO_START,
    # TunableCompliance/Hand/inhand_manipulation.py
    "inhand_manipulation":    _tunable.UR5_POSE_INHAND,
    # TunableCompliance/Hand/dynamic_grasp.py
    "dynamic_grasp":          _tunable.UR5_POSE_BOTTLE_START,
    # ProprioceptiveSensing/Hand/object_stiffness_hand.py
    "object_stiffness_hand":  _proprio.UR5_POSE_SQUEEZING,
    # ADAPT-StiffControl/grasp_adaptation.py (per-object grasp pose)
    "grasp_adaptation_hard":  _adapt.UR5_POSE_GRASP_OBJ["hard_obj"],
    "grasp_adaptation_soft":  _adapt.UR5_POSE_GRASP_OBJ["soft_obj"],
    # PoseControl/position_tracker.py (UR5 stays still; arm parked at squeezing pose)
    "position_tracking":      _proprio.UR5_POSE_SQUEEZING,
}


class URHandController:

    def __init__(self):
        self.ctrl = rtde_control.RTDEControlInterface(UR5_IP)
        self.recv = rtde_receive.RTDEReceiveInterface(UR5_IP)

        self.ctrl.setTcp([0, 0, 0, 0, 0, 0])
        self.ctrl.endTeachMode()

    def move_to(self, target_pose):
        current_pose = self.recv.getActualTCPPose()
        print("Current TCP Pose:", np.round(current_pose, 4).tolist())
        print("Target  TCP Pose:", np.round(target_pose, 4).tolist())

        self.ctrl.moveL(np.asarray(target_pose).tolist(),
                        speed=UR5_INIT_SPEED,
                        acceleration=UR5_INIT_ACCELERATION)
        print("UR5 movement complete!")


def prompt_experiment():
    names = list(HAND_POSES.keys())

    print("\nAvailable hand experiments:")
    for i, name in enumerate(names, start=1):
        pose = np.asarray(HAND_POSES[name])
        print(f"  [{i}] {name:<22}  pose = {np.round(pose, 4).tolist()}")

    while True:
        choice = input(f"\nSelect experiment [1-{len(names)}] or name: ").strip()
        if not choice:
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(names):
                return names[idx - 1]
        elif choice in HAND_POSES:
            return choice

        print(f"Invalid choice: {choice!r}. Try again.")


if __name__ == "__main__":
    experiment = prompt_experiment()
    target = np.asarray(HAND_POSES[experiment])
    print(f"\nMoving UR5 to '{experiment}' pose ...")

    controller = URHandController()
    controller.move_to(target)

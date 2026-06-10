# UR5_codes — UR5 Robot Arm Interface

Interface to the UR5 robot arm used to mount surfaces and apply controlled
displacements in passive compliance and stiffness characterisation experiments.
Communicates via RTDE (Real-Time Data Exchange) at IP `192.168.1.10`.

## Network setup

```bash
sudo ip addr flush dev eno1
sudo ip addr add 192.168.1.11/24 dev eno1
sudo ip link set eno1 up
```

## Files

### `UR5_config.py`

Configuration constants: IP address, TCP pose offsets, joint limits, speeds.

### `UR5_control.py`

Moves the UR5 to `UR5_POSE` (the finger-experiment fixture defined in
`UR5_config.py`). Single-shot linear move at `UR5_INIT_SPEED` /
`UR5_INIT_ACCELERATION`. No hand actuation.

### `UR5_control_hand.py`

Terminal-menu version of `UR5_control.py` for hand experiments. Lists the
initial pose of every hand experiment and moves the UR5 (only) to the
chosen one. Poses are imported directly from each experiment's own config
(`importlib` is used because `ADAPT-StiffControl` has a hyphen and several
folders share the module name `hand_config.py`), so any upstream pose
change is picked up automatically.

| Menu key                 | Source config                                          | Variable                   |
|--------------------------|--------------------------------------------------------|----------------------------|
| `position_tracking`      | `ProprioceptiveSensing/Hand/hand_config.py`            | `UR5_POSE_SQUEEZING`       |
| `guitar_playing_hand`    | `PassiveCompliance/Hand/hand_config.py`                | `UR5_POSE_GUITAR`          |
| `weight_compliance_hand` | `PassiveCompliance/Hand/hand_config.py`                | `UR5_POSE_WEIGHT`          |
| `inhand_manipulation`    | `TunableCompliance/Hand/hand_config.py`                | `UR5_POSE_INHAND`          |
| `dynamic_grasp`          | `TunableCompliance/Hand/hand_config.py`                | `UR5_POSE_BOTTLE_START`    |
| `object_stiffness_hand`  | `ProprioceptiveSensing/Hand/hand_config.py`            | `UR5_POSE_SQUEEZING`       |
| `grasp_adaptation`       | `ADAPT-StiffControl/hand_config.py`                    | `UR5_POSE_GRASP`           |

### `UR5_readPose.py`

Reads and logs the current TCP pose (position + orientation) from the UR5.

### `UR5_keyboard.py`

Keyboard-driven teleoperation of the UR5 TCP for manual positioning during
experiment setup.

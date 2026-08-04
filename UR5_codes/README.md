# UR5_codes — UR5 Robot Arm Interface

Interface to the UR5 robot arm used to mount surfaces and apply controlled displacements in passive compliance and stiffness characterisation experiments. Communicates via RTDE at IP `192.168.1.10`.

## Network setup

```bash
sudo ip addr flush dev eno1
sudo ip addr add 192.168.1.11/24 dev eno1
sudo ip link set eno1 up
```

## Files

| File | Description |
|------|-------------|
| `UR5_config.py` | Configuration constants: IP address, TCP pose offsets, joint limits, speeds |
| `UR5_control.py` | Single-shot move to `UR5_POSE` (the finger-experiment fixture) — no hand actuation |
| `UR5_control_hand.py` | Terminal-menu version: moves the UR5 (only) to the initial pose of any hand experiment, imported live via `importlib` from each experiment's own config so upstream pose changes are picked up automatically |
| `UR5_readPose.py` | Reads and logs the current TCP pose (position + orientation) |
| `UR5_keyboard.py` | Keyboard-driven teleoperation of the UR5 TCP for manual setup |

`UR5_control_hand.py` menu:

| Menu key | Source config | Variable |
|----------|----------------|----------|
| `position_tracking` | `ProprioceptiveSensing/Hand/hand_config.py` | `UR5_POSE_SQUEEZING` |
| `guitar_playing_hand` | `PassiveCompliance/Hand/hand_config.py` | `UR5_POSE_GUITAR` |
| `weight_compliance_hand` | `PassiveCompliance/Hand/hand_config.py` | `UR5_POSE_WEIGHT` |
| `inhand_manipulation` | `TunableCompliance/Hand/hand_config.py` | `UR5_POSE_INHAND` |
| `dynamic_grasp` | `TunableCompliance/Hand/hand_config.py` | `UR5_POSE_BOTTLE_START` |
| `object_stiffness_hand` | `ProprioceptiveSensing/Hand/hand_config.py` | `UR5_POSE_SQUEEZING` |
| `grasp_adaptation` | `ADAPT-StiffControl/hand_config.py` | `UR5_POSE_GRASP` |

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

Main control interface using `rtde_control` and `rtde_receive`.
Provides move functions (moveJ, moveL) and TCP force/torque reading.

### `UR5_readPose.py`

Reads and logs the current TCP pose (position + orientation) from the UR5.

### `UR5_keyboard.py`

Keyboard-driven teleoperation of the UR5 TCP for manual positioning during
experiment setup.

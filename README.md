# HumanManipulationVMC

Virtual Model Control (VMC) framework for programmable passivity and stiffness
modulation in tendon-driven compliant robotic hands.

**Platforms**:
- **Single finger** - tendon-driven finger with MCP and PIP, with mimic DIP
- **ADAPT Hand** - anthropomorphic hand with wrist, thumb, fingers, and spread

---

## Experimental Areas

`[finger]` = single finger testbed, `[hand]` = ADAPT Hand

---

### Passive Compliance Shaping - `PassiveCompliance/`

Validates that VMC generates diverse, predictable stiffness profiles at the fingertip
and palm through virtual springs alone. Compliance acts as a bidirectional filter:
outward (absorbs noise and impacts), inward (provides adaptability).

The scripts sweep virtual stiffness settings while an external motion applies a
controlled displacement. Logged tip force and displacement yield force-displacement
curves across contact directions and starting poses. The analysis relates virtual
spring settings to apparent compliance in task space and highlights how direction
and pose change the measured response.

**Finger/** - single finger testbed

| File | Description |
|------|-------------|
| `Finger/passive_stiffness_sweep.py` | Sweep virtual joint stiffness and record force versus displacement |
| `Finger/passive_range.py` | Dense sweep with one run per stiffness setting |
| `Finger/passive_stiffness_sweep_linear.py` | Sweep task-space stiffness in a pressing direction and record force versus displacement |
| `Finger/passive_range_linear.py` | Dense sweep in task space for the pressing direction |
| `Finger/directional_stiffness.py` | Task-space stiffness in multiple contact directions in a plane |
| `Finger/pose_sweep.py` | Sweep starting poses and record force versus displacement |

**Hand/** - ADAPT Hand compliance experiments

| File | Description |
|------|-------------|
| `Hand/guitar_playing_hand.py` | Three fingers (index, middle, ring) held closed while the UR5 performs a single upward stroke across the strings. One stiffness condition per run; audio captured externally. |
| `Hand/weight_compliance_hand.py` | Four fingers hold a fixed pose under joint-space springs. Weights are added incrementally by the operator; deflection is logged per (stiffness, weight) pair. |
| `Hand/AudioExtraction/` | Extracts and normalises audio from the three guitar experiment recordings |

---

### Tunable Compliance via VMC - `TunableCompliance/`

Shows that VMC enables real-time stiffness modulation across the full soft-to-rigid
spectrum, adapting to contact events and object properties — without hardware changes.
Includes single-finger characterisation and full-hand in-hand manipulation and
dynamic-grasp studies.

Finger controllers detect contact from deformation and force estimates, then
switch stiffness schedules in real time. The repulsive shaping variant adds a
nonlinear task-space element to increase apparent stiffness above baseline while
preserving passivity.

**Finger/** | `stiffening_contact.py`, `repulsive_stiffness_shaping.py` - contact-triggered stiffness updates and repulsive shaping in task space

Hand experiments use asymmetric fingertip stiffness patterns to bias object
motion and a schedule-based controller to transition between soft and stiff
phases during transport. Logged joint state and task events support comparison
between uniform and asymmetric conditions.

**Hand/** | `inhand_manipulation.py` - in-hand reorientation via asymmetric tip stiffness; `dynamic_grasp.py` - dynamic grasping during UR5 transport under soft, stiff, and adaptive schedules

---

### Proprioceptive Sensing - `ProprioceptiveSensing/`

Contact force and object stiffness estimated from kinematics and virtual stiffness
alone — no external force sensors. The deformation that absorbs impacts encodes
force.

The sensing pipeline infers contact force from virtual spring deformation and
kinematics. Finger experiments identify motor efficiency to map commanded torque
to contact force, then validate the estimate against external measurements.

**Finger/** | `finger_eta.py` - motor efficiency identification and force estimation validation

Hand experiments squeeze objects with two compliance settings and use an additive
compliance model to solve for object stiffness without external sensors.

**Hand/** | `object_stiffness_hand.py` - object stiffness estimation by squeezing with paired compliance settings

---

### Stiffness and Force Tracking - `StiffnessForceTracking/`

Model-based closed-loop control over the full force-displacement space.
Independent optimization pathways:
- **Reference modulation** - gradient descent on d_ref to move the operating point while keeping stiffness independent
- **Stiffness modulation** - gradient descent on K_d to change the slope of the force-displacement response

Controllers compute gradients from the stiffness model and Jacobians, then update
their parameters to track a desired force profile. Closed-loop variants use the
load cell as the feedback signal; open-loop variants replace the feedback with
model-predicted tip force for comparison.

| File | Platform | Description |
|------|----------|-------------|
| `experiment_config.py` | - | Shared config: force targets, timing, learning rates |
| `FORCE_CONTROL.py` | - | Batch runner: executes the controller variants sequentially |
| `force_position_control.py` | finger | Force and position control via K gradient descent (model-based) |
| `force_position_control_scalar.py` | finger | Scalar k*I baseline for force and position control |
| `force_stiffness_control.py` | finger | Force and stiffness control via d_ref gradient descent (model-based) |
| `force_stiffness_control_scalar.py` | finger | Scalar theta_ref baseline for force and stiffness control |
| `force_position_control_openloop.py` | finger | K descent driven by model-predicted force |
| `force_stiffness_control_openloop.py` | finger | d_ref descent driven by model-predicted force |
| `plot_force_position_control.ipynb` | — | Plot force position control experiment |
| `plot_force_stiffness_control.ipynb` | — | Plot force stiffness control experiment |
| `plot_openloop_tracking.ipynb` | — | Plot openloop tracking experiment |

---

### Pose Control - `PoseControl/`

Position tracking validation for the ADAPT Hand using joint-space VMC.
The hand is commanded through target poses inspired by hand synergies
(power grasp and precision pinch), and joint convergence is recorded for each.
The controller applies per-joint spring-damper behavior and logs tracking error
and convergence over time.

| File | Platform | Description |
|------|----------|-------------|
| `position_tracker.py` | hand | Commands the pose targets in sequence, logs joint convergence per pose |
| `plot_position_tracker.ipynb` | hand | Plot position tracker experiment |

---

### ADAPT Hand - Force Adaptation - `ADAPT-StiffControl/`

End-to-end closed-loop force adaptation on the full hand. The operator selects
a target force level (hard / medium / soft); the hand closes to a PC1 grasp pose
and runs stiffness-descent gradient descent on all five fingertips until analytic
tip force converges to the target. The arm then presses down to demonstrate the
exerted force.

| File | Platform | Description |
|------|----------|-------------|
| `grasp_adaptation.py` | hand | Select force level, GD stiffness adaptation, press demonstration |
| `plot_grasp_adaptation.ipynb` | hand | Plot GD convergence, stiffness evolution, and hand pose |

---

### Methods - Elastic Joint Validation - `MethodsElastic/`

Validates the VMC stiffness composition theorem experimentally. TPU joints are
attached as ground-truth torsional springs at all finger joints. Motors can be
disconnected or engaged depending on the script. The UR5 presses the fingertip
while the load cell records contact force, which is compared against model
predictions to confirm the stiffness mapping. Mimic controllers either apply a
fixed virtual stiffness or update it online via feed-forward inversion and
reference descent.

| File | Platform | Description |
|------|----------|-------------|
| `elastic_band_sweep.py` | finger | UR5 pressing with motors disengaged and elastic bands on all joints |
| `20mm_mimic_real_springs.py` | finger | UR5 pressing with motors engaged and a fixed virtual joint stiffness matched to the elastic-band average |
| `instant_mimic_real_springs.py` | finger | UR5 pressing with motors engaged and online stiffness updates via feed-forward inversion and ref_descent feedback |
| `plot_elastic_band.ipynb` | — | Plot elastic band experiment |

---

### Supplementary - `Supplementary/`

Additional single-finger experiments providing further characterisation of the
VMC framework: non-linear stiffness profiles, softening on contact, and task-space
sensing configurations.

These scripts probe alternative virtual spring laws and sensing geometries to
show how stiffness shaping and task-space projections influence the measured
force response.

| File | Platform | Description |
|------|----------|-------------|
| `softening_contact.py` | finger | Detect contact and reduce virtual stiffness to soften the interaction |
| `sigmoidal_polynomial_stiffness.py` | finger | Use nonlinear stiffness laws to shape the force-displacement curve |
| `sensing_task_space.py` | finger | Compare force prediction for task-space and mixed-space VMC configurations |
| `plot_softening_contact.ipynb` | finger | Plot softening contact experiment |
| `plot_sigmoidal_polynomial_stiffness.ipynb` | finger | Plot sigmoidal polynomial stiffness experiment |
| `plot_sensing_task_space.ipynb` | finger | Plot sensing task space experiment |

---

## Repository Structure

```
HumanManipulationVMC/
│
├── PassiveCompliance/              # Experimental area - passive compliance
│   ├── Finger/                     #   single finger experiments
│   └── Hand/                       #   ADAPT Hand compliance experiments
│       └── AudioExtraction/        #     guitar audio extraction and normalisation
├── TunableCompliance/              # Experimental area - tunable compliance
│   ├── Finger/                     #   stiffening/repulsive shaping (finger)
│   └── Hand/                       #   emergent grasps (hand)
├── ProprioceptiveSensing/          # Experimental area - proprioceptive sensing
│   ├── Finger/                     #   motor efficiency + force estimation
│   └── Hand/                       #   object stiffness estimation
├── StiffnessForceTracking/         # Experimental area - stiffness and force tracking
├── PoseControl/                    # Experimental area - pose control
├── ADAPT-StiffControl/             # Final demo - force adaptation via stiffness descent on the full hand
├── Supplementary/                  # Experimental area - supplementary experiments
├── MethodsElastic/                 # Methods - VMC stiffness model validation
│
├── KinematicsFinger/       # FK, Jacobians, Hessians - single finger
├── KinematicsHand/         # FK, Jacobians, Hessians - ADAPT hand
│
├── VMCFinger/              # FingerController (high-rate), GravLim, VMC finger/task/dirCart/repulsive
├── VMCHand/                # HandController (lower-rate), GravLim, HandVMCJointSpace/TaskSpace
├── VMC_utils/              # VirtualModels (springs, dampers), GravityCompensation
├── StiffnessModelFinger/   # Stiffness mapping finger/task/mixed space to tip stiffness
├── StiffnessModelHand/     # Stiffness mapping joint/task/mixed space to per-fingertip stiffness
│
├── ModelIDFinger/          # finger_params.py — single source of truth for finger parameters
├── ModelIDHand/            # hand_params.py, motor_config.py — single source of truth for hand parameters
│
├── LoadCell/               # Arduino force sensor interface
├── UR5_codes/              # UR5 arm control (RTDE, IP configured in UR5_config.py)
│
├── finger_go_home.py       # Bring finger to home position (run before any finger experiment)
├── hand_go_home.py         # Bring hand to home position (run before any hand experiment)
├── start_finger.sh         # Full finger startup: home + latency timer + dynamixel node
├── start_hand.sh           # Full hand startup: home + latency timer + dynamixel node (torque/position mixed)
├── ADAPT_Hand.urdf         # URDF of the ADAPT hand
└── plot_config.mplstyle    # Shared matplotlib style
```

---

## Setup

### Reduce USB serial latency

```bash
sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

### Start the ROS2 Dynamixel node

For the **finger**, use the startup script (does all steps in order):
```bash
./start_finger.sh
```

Or manually:
```bash
python3 finger_go_home.py
sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
ros2 run dynamixel_interface dynamixel_node --ros-args -p baudrate:=1000000 -p motor_ids:=[1,2]
```

For the **hand**, use the startup script (does all steps in order):
```bash
./start_hand.sh
```

This script:
1. Moves the hand to home position (`hand_go_home.py`) — finger/thumb motors end in torque mode, wrist motors stay in position mode holding the home pose
2. Sets the USB latency timer for maximum throughput
3. Starts a single Dynamixel node: finger/thumb motors (hw 0–12) in torque mode, wrist motors (hw 13–14) in position mode via `position_motor_ids`

Or step by step manually:
```bash
python3 hand_go_home.py
sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
ros2 run dynamixel_interface dynamixel_node --ros-args -p baudrate:=2000000 -p motor_ids:=[0,1,2,3,4,5,6,7,8,9,10,11,12] -p position_motor_ids:=[13,14]
```

### Configure UR5 network (optional)

```bash
sudo ip addr flush dev eno1
sudo ip addr add 192.168.1.11/24 dev eno1
sudo ip link set eno1 up
```

UR5 IP: `192.168.1.10`

---

## Quick Start

> **Always run the go-home script first** to bring hardware to a known safe configuration.

```bash
# Finger experiments
./start_finger.sh        # home + latency timer + dynamixel node
python3 PassiveCompliance/Finger/passive_stiffness_sweep.py

# Hand experiments
./start_hand.sh          # home + latency timer + dynamixel node
python3 ADAPT-StiffControl/grasp_adaptation.py
```

All scripts must be run from the **repository root** so that module imports resolve correctly.

---

## Documentation

- [VMCHand/HAND_VMC_DOCUMENTATION.md](VMCHand/HAND_VMC_DOCUMENTATION.md) — hand controller API, motor ordering
- [VMCFinger/FINGER_VMC_DOCUMENTATION.md](VMCFinger/FINGER_VMC_DOCUMENTATION.md) — finger controller API
- [KinematicsHand/KINEMATICS_DOCUMENTATION.md](KinematicsHand/KINEMATICS_DOCUMENTATION.md) — hand FK/Jacobian reference
- [KinematicsFinger/KINEMATICS_DOCUMENTATION.md](KinematicsFinger/KINEMATICS_DOCUMENTATION.md) — finger FK/Jacobian reference

## Dependencies

### System packages

```bash
# ROS2 Humble (includes rclpy and std_msgs)
# Install from https://docs.ros.org/en/humble/Installation.html

# dynamixel_interface ROS2 package (Dynamixel motor node)
# Install from https://github.com/Husarion/dynamixel_interface or equivalent

# ffmpeg — audio/video extraction (PassiveCompliance/Hand guitar experiment)
sudo apt install ffmpeg
```

### Python packages

```bash
pip install numpy scipy sympy pandas matplotlib scienceplots \
            scikit-learn tqdm pynput \
            rtde-control rtde-receive \
            pyserial \
            noisereduce soundfile
```

| Package | Used for |
|---------|----------|
| `numpy` | all numerical operations |
| `scipy` | interpolation, optimisation, signal processing, spatial transforms, bandpass filter |
| `sympy` | symbolic Jacobians and kinematics |
| `pandas` | CSV loading and data analysis in notebooks |
| `matplotlib` | all plots |
| `scienceplots` | shared matplotlib style (`plot_config.mplstyle`) |
| `scikit-learn` | `IsotonicRegression` (stiffness learning) |
| `tqdm` | progress bars |
| `pynput` | keyboard input for teleoperation scripts |
| `rtde-control` / `rtde-receive` | UR5 robot arm RTDE interface |
| `pyserial` | Arduino load cell over USB |
| `noisereduce` | spectral gating noise removal (guitar audio extraction) |
| `soundfile` | WAV audio I/O (guitar audio extraction) |

### Local modules (included in repo, no install needed)

`KinematicsFinger`, `KinematicsHand`, `VMCFinger`, `VMCHand`, `VMC_utils`,
`ModelIDFinger`, `ModelIDHand`, `StiffnessModelFinger`, `StiffnessModelHand`,
`StiffnessForceTracking`, `UR5_codes`, `LoadCell`

### Notes

- **rclpy / std_msgs**: installed as part of ROS2 Humble, not via pip.
- **Fusion 360 add-in** (`FusionPlotting/FusionAddin/`): the `adsk` modules are provided by Fusion 360 itself — do not pip-install them.
- **Arduino**: upload `LoadCell/Real_res1_twoLoadCells_HW.ino` separately to the Arduino board.
- All scripts must be run from the **repository root** for module imports to resolve.

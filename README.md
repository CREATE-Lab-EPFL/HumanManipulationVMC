# HumanManipulationVMC

Virtual Model Control (VMC) framework for programmable passivity and stiffness
modulation in tendon-driven compliant robotic hands.

Collaboration between **EPFL CREATE Lab** (Josie Hughes) and **Cambridge Control Robotics Lab** (Fulvio Forni).

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
curves across contact directions and starting poses.

**Finger/** - single finger testbed

| File | Description |
|------|-------------|
| `Finger/passive_stiffness_sweep.py` | Sweep virtual joint stiffness and record force versus displacement |
| `Finger/passive_range.py` | Dense sweep with one run per stiffness setting |
| `Finger/passive_stiffness_sweep_linear.py` | Sweep task-space stiffness in a pressing direction and record force versus displacement |
| `Finger/passive_range_linear.py` | Dense sweep in task space for the pressing direction |
| `Finger/directional_stiffness.py` | Task-space stiffness in multiple contact directions in a plane |
| `Finger/pose_sweep.py` | Sweep starting poses and record force versus displacement |

**Hand/** - ADAPT Hand piano playing

| File | Description |
|------|-------------|
| `Hand/piano_playing_hand.py` | Hand holds a press pose while the UR5 performs rhythmic press-lift strokes. Runs uniform and mixed stiffness conditions, saving hand state and MIDI logs per stroke. |
| `Hand/piano_glissando.py` | Hand holds a press pose while the UR5 slides along the keyboard and returns. Saves hand state and MIDI logs per run. |
| `Hand/HelperPianoMIDI/` | MIDI keyboard to ROS bridge (publisher, subscriber, UR5 config) |

---

### Tunable Compliance via VMC - `TunableCompliance/`

Shows that VMC enables real-time stiffness modulation across the full soft-to-rigid
spectrum, adapting to contact events and object properties — without hardware changes.
Includes single-finger characterisation and full-hand in-hand manipulation and
dynamic-grasp studies.

**Finger/** | `stiffening_contact.py`, `repulsive_stiffness_shaping.py` - contact-triggered stiffness updates and repulsive shaping in task space

**Hand/** | `inhand_manipulation.py` - in-hand reorientation via asymmetric tip stiffness; `dynamic_grasp.py` - dynamic grasping during UR5 transport under soft, stiff, and adaptive schedules

---

### Proprioceptive Sensing - `ProprioceptiveSensing/`

Contact force and object stiffness estimated from kinematics and virtual stiffness
alone — no external force sensors. The deformation that absorbs impacts encodes
force. Sensing sensitivity is maximised when virtual compliance approximately
matches object compliance.

**Finger/** | `finger_eta.py` - motor efficiency identification and force estimation validation

**Hand/** | `object_stiffness_hand.py` - object stiffness estimation by squeezing with paired compliance settings

---

### Stiffness and Force Tracking - `StiffnessForceTracking/`

Model-based closed-loop control over the full force-displacement space.
Independent optimization pathways:
- **Reference modulation** - gradient descent on d_ref to move the operating point while keeping stiffness independent
- **Stiffness modulation** - gradient descent on K_d to change the slope of the force-displacement response

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
The hand is commanded through a pair of target poses inspired by hand synergies
(power grasp and precision pinch), and joint convergence is recorded for each.

| File | Platform | Description |
|------|----------|-------------|
| `position_tracker.py` | hand | Commands the pose targets in sequence, logs joint convergence per pose |
| `plot_position_tracker.ipynb` | hand | Plot position tracker experiment |

---

### ADAPT Hand - Grasp Adaptation - `ADAPT-StiffControl/`

End-to-end closed-loop stiffness adaptation on the full hand. The hand
estimates object compliance with the same paired-sample approach used in
`ProprioceptiveSensing/Hand` (settle, probe, compare FK position and analytic
VMC tip force), then maps compliance to applied fingertip stiffness with
saturation before lifting, holding, and placing the object back. Final
integrative demonstration of the paper.

| File | Platform | Description |
|------|----------|-------------|
| `grasp_adaptation.py` | hand | Gentle contact, probe, estimate compliance, then lift / hold / place with matched stiffness |
| `plot_grasp_adaptation.ipynb` | hand | Plot grasp adaptation experiment |

---

### Methods - Elastic Joint Validation - `MethodsElastic/`

Validates the VMC stiffness composition theorem experimentally. TPU joints are
attached as ground-truth torsional springs at all finger joints. Motors can be
disconnected or engaged depending on the script. The UR5 presses the fingertip
while the load cell records contact force, which is compared against model
predictions to confirm the stiffness mapping.

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
│   └── Hand/                       #   ADAPT Hand piano experiments
│       └── HelperPianoMIDI/        #     MIDI keyboard → ROS2 bridge + UR5 config
├── TunableCompliance/              # Experimental area - tunable compliance
│   ├── Finger/                     #   stiffening/repulsive shaping (finger)
│   └── Hand/                       #   emergent grasps (hand)
├── ProprioceptiveSensing/          # Experimental area - proprioceptive sensing
│   ├── Finger/                     #   motor efficiency + force estimation
│   └── Hand/                       #   object stiffness estimation
├── StiffnessForceTracking/         # Experimental area - stiffness and force tracking
├── PoseControl/                    # Experimental area - pose control
├── ADAPT-StiffControl/             # Final demo - compliance matching on the full hand
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
├── ADAPT_Hand.urdf         # URDF of the ADAPT hand
└── plot_config.mplstyle    # Shared matplotlib style
```

---

## Setup

### Reduce USB serial latency

Set the usb-serial latency_timer to a low value for the connected device. See
your OS documentation for the exact sysfs path and recommended value.

### Start the ROS2 Dynamixel node

For the **finger**:
```bash
ros2 run dynamixel_interface dynamixel_node
```

For the **hand**:
```bash
ros2 run dynamixel_interface dynamixel_node --ros-args -p baudrate:=<baudrate> -p motor_ids:=[<motor_ids>]
```

Use the motor IDs and baudrate defined in your hand configuration.

### Configure UR5 network (optional)

Configure the host interface to the UR5 subnet and bring it up. The UR5 IP and
host settings are defined in `UR5_codes/UR5_config.py`.

---

## Quick Start

> **Always run the go-home script first** to bring hardware to a known safe configuration.

```bash
# Finger experiments
python finger_go_home.py
python PassiveCompliance/Finger/passive_stiffness_sweep.py

# Hand experiments
python hand_go_home.py
python ADAPT-StiffControl/grasp_adaptation.py
```

All scripts must be run from the **repository root** so that module imports resolve correctly.

---

## Documentation

- [VMCHand/HAND_VMC_DOCUMENTATION.md](VMCHand/HAND_VMC_DOCUMENTATION.md) — hand controller API, motor ordering
- [VMCFinger/FINGER_VMC_DOCUMENTATION.md](VMCFinger/FINGER_VMC_DOCUMENTATION.md) — finger controller API
- [KinematicsHand/KINEMATICS_DOCUMENTATION.md](KinematicsHand/KINEMATICS_DOCUMENTATION.md) — hand FK/Jacobian reference
- [KinematicsFinger/KINEMATICS_DOCUMENTATION.md](KinematicsFinger/KINEMATICS_DOCUMENTATION.md) — finger FK/Jacobian reference

## Dependencies

- ROS2 (tested with Humble)
- `dynamixel_interface` ROS2 package
- Python: `numpy`, `scipy`, `rclpy`, `std_msgs`
- Analysis: `pandas`, `matplotlib`, `scikit-learn`, `jupyter`
- UR5: `rtde_control`, `rtde_receive`

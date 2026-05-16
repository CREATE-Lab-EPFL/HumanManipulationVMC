# HumanManipulationVMC

Virtual Model Control (VMC) framework for programmable passivity and stiffness
modulation in tendon-driven compliant robotic hands.

Collaboration between **EPFL CREATE Lab** (Josie Hughes) and **Cambridge Control Robotics Lab** (Fulvio Forni).

**Platforms**:
- **Single finger** — 2-DOF tendon-driven finger (MCP + PIP, with mimic DIP)
- **ADAPT Hand** — 15-DOF anthropomorphic hand (wrist + thumb + 4 fingers + spread)

---

## Experimental Areas

`[finger]` = single finger testbed · `[hand]` = ADAPT Hand

---

### 1. Passive Compliance Shaping — `PassiveCompliance/`

Validates that VMC generates diverse, predictable stiffness profiles at the fingertip
and palm through virtual springs alone. Compliance acts as a bidirectional filter:
outward (absorbs noise/impacts), inward (provides adaptability).

**Finger/** — single 2-DOF finger testbed

| File | Description |
|------|-------------|
| `Finger/passive_stiffness_sweep.py` | Stiffness sweep — F vs d for varying K (finger space, N·m/rad) |
| `Finger/passive_range.py` | Dense biased K sweep, one run per K |
| `Finger/passive_stiffness_sweep_linear.py` | Cart stiffness sweep — F vs d for varying K_cart (N/m), vertical direction |
| `Finger/passive_range_linear.py` | Dense biased K_cart sweep, vertical direction |
| `Finger/directional_stiffness.py` | Cart stiffness in multiple contact directions in the Y-Z plane |
| `Finger/pose_sweep.py` | Pose sweep — F vs d from multiple starting Z heights |

**Hand/** — ADAPT Hand piano playing

| File | Description |
|------|-------------|
| `Hand/piano_playing_hand.py` | Conditions 1 & 2: rhythmic index+ring pressing, uniform or heterogeneous K [N/m] |
| `Hand/piano_glissando.py` | Condition 3: index+middle glissando — UR5 slides across keys while fingers stay pressed |
| `Hand/HelperPianoMIDI/` | MIDI keyboard → ROS2 bridge (publisher, subscriber, UR5 config) |

---

### 2. Tunable Compliance via VMC — `TunableCompliance/`

Shows that VMC enables real-time stiffness modulation across the full soft-to-rigid
spectrum, adapting to contact events and object properties — without hardware changes.
Includes single-finger characterisation and full-hand in-hand manipulation and
dynamic-grasp studies.

**Finger/** | `stiffening_contact.py`, `repulsive_stiffness_shaping.py`

**Hand/** | `inhand_manipulation.py` — in-hand reorientation via asymmetric tip stiffness · `dynamic_grasp.py` — bottle transport under soft / stiff / adaptive (soft→stiff) tip stiffness

---

### 3. Proprioceptive Sensing — `ProprioceptiveSensing/`

Contact force and object stiffness estimated from kinematics and virtual stiffness
alone — no external force sensors. The deformation that absorbs impacts encodes
force. Sensing sensitivity is maximised when C_A (virtual) ≈ C_O (object).

**Finger/** | `finger_eta.py` — motor efficiency identification and force estimation validation

**Hand/** | `object_stiffness_hand.py` — object stiffness estimation by squeezing — C_O from (C_A + C_O)

---

### 4. Stiffness and Force Tracking — `StiffnessForceTracking/`

Model-based closed-loop control over the full force-displacement space.
Two independent optimization pathways:
- **Reference modulation** — gradient descent on d_ref: faster, decouples stiffness from force
- **Stiffness modulation** — gradient descent on K_d: simultaneous force and stiffness control

| File | Platform | Description |
|------|----------|-------------|
| `experiment_config.py` | — | Shared config: force levels, timing, learning rates |
| `FORCE_CONTROL.py` | — | Batch runner: executes all four controllers sequentially |
| `force_position_control.py` | finger | Force+position control via K gradient descent (model-based) |
| `force_position_control_scalar.py` | finger | Scalar k·I baseline for force+position control |
| `force_stiffness_control.py` | finger | Force+stiffness control via d_ref gradient descent (model-based) |
| `force_stiffness_control_scalar.py` | finger | Scalar θ_ref baseline for force+stiffness control |
| `force_position_control_openloop.py` | finger | K-descent driven by model-predicted force |
| `force_stiffness_control_openloop.py` | finger | d_ref-descent driven by model-predicted force |
| `plot_force_position_control.ipynb` | — | Plot force position control experiment |
| `plot_force_stiffness_control.ipynb` | — | Plot force stiffness control experiment |
| `plot_openloop_tracking.ipynb` | — | Plot openloop tracking experiment |

---

### 5. Pose Control — `PoseControl/`

Position tracking validation for the 15-DOF ADAPT Hand using joint-space VMC.
The hand is commanded through two target poses inspired by hand synergies
(Santello et al. 1998) — PC1 (power grasp) and PC2 (precision pinch) — and
joint convergence is recorded for each.

| File | Platform | Description |
|------|----------|-------------|
| `position_tracker.py` | hand | Commands PC1 and PC2 poses in sequence, logs joint convergence per pose |
| `plot_position_tracker.ipynb` | hand | Plot position tracker experiment |

---

### ADAPT Hand — Grasp Adaptation — `ADAPT-StiffControl/`

End-to-end closed-loop stiffness adaptation on the full hand: the hand senses
the grasped object's compliance from fingertip displacement at a gentle stiffness,
then ramps the tip stiffness to a value adapted to that displacement before
lifting, holding, and placing the object back. Final integrative demonstration
of the paper.

| File | Platform | Description |
|------|----------|-------------|
| `grasp_adaptation.py` | hand | Sense δ_mean at gentle K, ramp to k_applied = clip(K_SCALE·δ_mean, K_MIN, K_MAX), lift / hold / place (`hard_obj` and `soft_obj`) |
| `plot_grasp_adaptation.ipynb` | hand | Plot grasp adaptation experiment |

---

### Methods — TPU Joint Validation — `MethodsElastic/`

Validates the VMC stiffness composition theorem experimentally. TPU joints are
attached as ground-truth torsional springs at all finger joints (MCP, PIP, DIP) simultaneously.
Motors are fully disconnected. The UR5 presses the fingertip while the load cell records
the contact force, which is compared against the model prediction to confirm the
stiffness mapping K_d → K_x.

| File | Platform | Description |
|------|----------|-------------|
| `elastic_band_sweep.py` | finger | UR5 descent + load cell recording — no motors, elastic bands on all joints |
| `20mm_mimic_real_springs.py` | finger | UR5 descent + load cell — motors, K_d set to match the run-average stiffness of soft/hard bands |
| `instant_mimic_real_springs.py` | finger | UR5 descent + load cell — motors, K_d set via feed-forward stiffness inversion + ref_descent force feedback to track instantaneous K_x(d) profile |
| `plot_elastic_band.ipynb` | — | Plot elastic band experiment |

---

### 6. Supplementary — `Supplementary/`

Additional single-finger experiments providing further characterisation of the
VMC framework: non-linear stiffness profiles, softening on contact, and task-space
sensing configurations.

| File | Platform | Description |
|------|----------|-------------|
| `softening_contact.py` | finger | K_d reduced online upon contact |
| `sigmoidal_polynomial_stiffness.py` | finger | Non-linear K(d) profiles: sigmoid and polynomial |
| `sensing_task_space.py` | finger | Force prediction for task-space and combined-space VMC configurations |
| `plot_softening_contact.ipynb` | finger | Plot softening contact experiment |
| `plot_sigmoidal_polynomial_stiffness.ipynb` | finger | Plot sigmoidal polynomial stiffness experiment |
| `plot_sensing_task_space.ipynb` | finger | Plot sensing task space experiment |

---

## Repository Structure

```
HumanManipulationVMC/
│
├── PassiveCompliance/              # Experimental area 1
│   ├── Finger/                     #   single finger experiments
│   └── Hand/                       #   ADAPT Hand piano experiments
│       └── HelperPianoMIDI/        #     MIDI keyboard → ROS2 bridge + UR5 config
├── TunableCompliance/              # Experimental area 2
│   ├── Finger/                     #   stiffening/repulsive shaping (finger)
│   └── Hand/                       #   emergent grasps (hand)
├── ProprioceptiveSensing/          # Experimental area 3
│   ├── Finger/                     #   motor efficiency + force estimation
│   └── Hand/                       #   object stiffness estimation
├── StiffnessForceTracking/         # Experimental area 4
├── PoseControl/                    # Experimental area 5
├── ADAPT-StiffControl/             # Final demo — compliance matching on the full hand
├── Supplementary/                  # Experimental area 6
├── MethodsElastic/                 # Methods — VMC stiffness model validation (elastic bands)
│
├── KinematicsFinger/       # FK, Jacobians, Hessians — 2-DOF finger
├── KinematicsHand/         # FK, Jacobians, Hessians — 15-DOF ADAPT hand
│
├── VMCFinger/              # FingerController (900 Hz), GravLim, VMC finger/task/dirCart/repulsive
├── VMCHand/                # HandController (330 Hz), GravLim, HandVMCJointSpace/TaskSpace
├── VMC_utils/              # VirtualModels (springs, dampers), GravityCompensation
├── StiffnessModelFinger/   # Stiffness mapping finger/task/mixed space → tip stiffness
├── StiffnessModelHand/     # Stiffness mapping joint/task/mixed space → per-fingertip stiffness
│
├── ModelIDFinger/          # finger_params.py — single source of truth for finger parameters
├── ModelIDHand/            # hand_params.py, motor_config.py — single source of truth for hand parameters
│
├── LoadCell/               # Arduino force sensor interface
├── UR5_codes/              # UR5 arm control (RTDE, IP 192.168.1.10)
│
├── finger_go_home.py       # Bring finger to home position (run before any finger experiment)
├── hand_go_home.py         # Bring hand to home position (run before any hand experiment)
├── ADAPT_Hand.urdf         # URDF of the 15-DOF ADAPT hand
└── plot_config.mplstyle    # Shared matplotlib style
```

---

## Setup

### 1. Reduce USB serial latency

```bash
sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

### 2. Start the ROS2 Dynamixel node

For the **finger** (2 motors):
```bash
ros2 run dynamixel_interface dynamixel_node
```

For the **hand** (15 motors):
```bash
ros2 run dynamixel_interface dynamixel_node --ros-args -p baudrate:=2000000 -p motor_ids:=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
```

### 3. (Optional) Configure UR5 network

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
python3 finger_go_home.py
python3 PassiveCompliance/Finger/passive_stiffness_sweep.py

# Hand experiments
python3 hand_go_home.py
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

- ROS2 (tested with Humble)
- `dynamixel_interface` ROS2 package
- Python: `numpy`, `scipy`, `rclpy`, `std_msgs`
- Analysis: `pandas`, `matplotlib`, `scikit-learn`, `jupyter`
- UR5: `rtde_control`, `rtde_receive`

"""
Supplementary — force prediction for task-space and combined-space VMC (finger).

Protocol requested for model-testing:
  1) Move UR5 to UR5_POSE and keep it fixed.
  2) Run random stiffness trials:
       - task_space:    N_RANDOM samples with random K_task
       - combined_space:N_RANDOM samples with random K_task and K_finger
  3) For each sample:
       a. Set stiffness; lower finger to FINGER_TARGET; wait LOWER_SETTLE_TIME.
       b. Record for SAMPLE_RECORD_TIME; save averaged measured vs predicted force.
       c. Lift finger back to LIFTED_TARGET; wait LIFT_SETTLE_TIME.

Used models/controllers (as requested):
  - Control:    FingerVMCTaskSpace, FingerVMCFingerSpace
  - Prediction: stiffness2taskspace, stiffness2mixedspace

Outputs:
  Supplementary/outputs/sensing_task_space_random/sensing_task_space_random_run_N.csv
"""

import csv
import os
import sys
import threading
import time

import numpy as np
import rclpy
from std_msgs.msg import Float64

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from KinematicsFinger.FK_Finger import FK_DIP, FK_MCP, FK_PIP, joint_to_motor
from ModelIDFinger.finger_params import a, b, c, eta
from StiffnessModelFinger.stiffness2mixedspace import tip_stiffness_MixedSpace
from StiffnessModelFinger.stiffness2taskspace import tip_stiffness_TaskSpace
from UR5_codes.UR5_config import (
    BENDING_DAMPING,
    FINGER_TARGET,
    UR5_INIT_ACCELERATION,
    UR5_INIT_SPEED,
    UR5_IP,
    UR5_POSE,
)
from VMC_utils.VirtualModels import LinearSpring
from VMCFinger.FingerController import CONTROL_FREQUENCY, FingerController
from VMCFinger.FingerGravFricLim import GravFricLim
from VMCFinger.FingerVMCFingerSpace import VMC as FingerSpaceVMC
from VMCFinger.FingerVMCTaskSpace import VMC as TaskSpaceVMC

import rtde_control
import rtde_receive


# ── Requested protocol settings ───────────────────────────────────────────────
N_RANDOM = 40                  # random values per configuration

LOWER_SETTLE_TIME = 3.0        # [s] wait after finger lowers before recording
LIFT_SETTLE_TIME  = 2.0        # [s] wait after finger lifts before next sample
SAMPLE_RECORD_TIME = 1.5       # [s] averaging window for each random trial

LIFTED_TARGET = np.array([np.deg2rad(10.0)] * 3)  # straight finger between samples

# Random stiffness ranges (kept conservative for safe contact)
K_TASK_MIN = 10.0              # [N/m]
K_TASK_MAX = 40.0              # [N/m]
K_FINGER_MIN = 0.01            # [N·m/rad]
K_FINGER_MAX = 0.20            # [N·m/rad]

RNG_SEED = 42
CONFIGS = ['task_space', 'combined_space']

# Set to True once data is collected — runs the protocol without saving files.
COLLECTED_DATA = True

def _diag3(value: float) -> np.ndarray:
    return np.diag([value, value, value])


def _open_csv(config_name):
    folder = os.path.join(_HERE, 'outputs', 'sensing_task_space_random')
    os.makedirs(folder, exist_ok=True)
    prefix = f'sensing_task_space_random_{config_name}_run_'
    existing = [f for f in os.listdir(folder)
                if f.startswith(prefix) and f.endswith('.csv')]
    num = max([int(f.replace(prefix, '').replace('.csv', ''))
               for f in existing], default=0) + 1
    path = os.path.join(folder, f'{prefix}{num}.csv')

    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        'Config',
        'Sample_Num',
        'K_task_Npm',
        'K_finger_Nmrad',
        'Time_s',
        'Force_Normal_Mean_N',
        'Force_Normal_Std_N',
        'Force_Pred_Mean_N',
        'Force_Pred_Std_N',
        'Pred_Error_Mean_N',
        'Motor1_pos_Mean_deg',
        'Motor2_pos_Mean_deg',
        'UR5_Z_m',
    ])
    return f, w, path


def _move_arm_async(arm, target_pose, speed, done_state):
    global state, arm_moving, state_start_time

    def _run():
        global state, arm_moving, state_start_time
        arm.moveL(target_pose.tolist(), speed, UR5_INIT_ACCELERATION)
        state_start_time = time.time()
        state = done_state
        arm_moving = False

    arm_moving = True
    threading.Thread(target=_run, daemon=True).start()


# ── Targets and prediction models ─────────────────────────────────────────────
q_ref_motor_rad = joint_to_motor(FINGER_TARGET[:2])
q_ref_motor_deg = np.degrees(q_ref_motor_rad)

# Task-space controller targets from FINGER_TARGET
TARGET_MCP = FK_MCP(q_ref_motor_rad, np.array([0.0, a, 0.0]))
TARGET_PIP = FK_PIP(q_ref_motor_rad, np.array([0.0, b, 0.0]))
TARGET_DIP = FK_DIP(q_ref_motor_rad, np.array([0.0, c, 0.0]))

# Lifted (straight) finger targets for task-space VMC
q_lift_motor_rad = joint_to_motor(LIFTED_TARGET[:2])
TARGET_MCP_LIFT  = FK_MCP(q_lift_motor_rad, np.array([0.0, a, 0.0]))
TARGET_PIP_LIFT  = FK_PIP(q_lift_motor_rad, np.array([0.0, b, 0.0]))
TARGET_DIP_LIFT  = FK_DIP(q_lift_motor_rad, np.array([0.0, c, 0.0]))

n_vec = np.array([0.0, 0.0, 1.0])
eta_mat = np.diag(eta)

model_task = tip_stiffness_TaskSpace(n=n_vec, eta=eta_mat)
model_mixed = tip_stiffness_MixedSpace(n=n_vec, eta=eta_mat)


def predict_normal_force(config_name, q_deg, k_task, k_finger):
    K_tip = _diag3(k_task)
    K_base = _diag3(k_task)

    if config_name == 'task_space':
        f_vec = model_task.tip_force(q_deg, q_ref_motor_deg, K_tip, K_base)
    elif config_name == 'combined_space':
        K_theta = _diag3(k_finger)
        f_vec = model_mixed.tip_force(q_deg, q_ref_motor_deg, K_theta, K_tip, K_base)
    else:
        raise ValueError(f'Unknown config: {config_name}')

    return float(n_vec @ f_vec)


# ── ROS2 and controllers ──────────────────────────────────────────────────────
rclpy.init()
controller = FingerController()
grav_fric_lim = GravFricLim()

force_total = 0.0
force_normal = 0.0
force_shear = 0.0


for topic, attr in [('/force', 'force_total'),
                    ('/force_normal', 'force_normal'),
                    ('/force_shear', 'force_shear')]:

    def _cb(msg, a=attr):
        global force_total, force_normal, force_shear
        value = float(getattr(msg, 'data', 0.0)) * 0.00980665
        if a == 'force_total':
            force_total = value
        elif a == 'force_normal':
            force_normal = value
        else:
            force_shear = value

    controller.create_subscription(Float64, topic, _cb, 10)


# Task-space VMC is always active — starts at lifted target, PIP spring zeroed
vmc_task = TaskSpaceVMC(
    stiffness=np.array([0.4, 0.4, 0.4]),
    damping=np.array([BENDING_DAMPING] * 3),
    target_MCP=TARGET_MCP_LIFT,
    target_PIP=TARGET_PIP_LIFT,
    target_DIP=TARGET_DIP_LIFT,
)
vmc_task.springs['PIP'] = LinearSpring(np.zeros(3))

# Finger-space VMC is only added in combined mode — starts at lifted target
vmc_finger = FingerSpaceVMC(
    stiffness=np.array([0.4, 0.4, 0.4]),
    damping=np.array([BENDING_DAMPING] * 3),
    target=LIFTED_TARGET,
)
vmc_finger.spring = LinearSpring(_diag3(0.4))


def set_task_stiffness(k_task):
    k_vec = np.array([k_task, k_task, k_task])
    for joint_name in ('MCP', 'DIP'):
        vmc_task.springs[joint_name] = LinearSpring(k_vec)


def set_finger_stiffness(k_finger):
    vmc_finger.spring = LinearSpring(_diag3(k_finger))


# ── UR5 setup ────────────────────────────────────────────────────────────────
arm = rtde_control.RTDEControlInterface(UR5_IP)
recv = rtde_receive.RTDEReceiveInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()
controller.get_logger().info('UR5 connected')


# ── Experiment queue ─────────────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)

samples = []
for cfg in CONFIGS:
    for sample_id in range(1, N_RANDOM + 1):
        k_task = float(rng.uniform(K_TASK_MIN, K_TASK_MAX))
        if cfg == 'combined_space':
            k_finger = float(rng.uniform(K_FINGER_MIN, K_FINGER_MAX))
        else:
            k_finger = 0.0
        samples.append((cfg, sample_id, k_task, k_finger))

total_samples = len(samples)


# ── State machine ────────────────────────────────────────────────────────────
STATE_INIT_ARM    = 0
STATE_PREP_SAMPLE = 1
STATE_LOWERING    = 2
STATE_RECORD_SAMPLE = 3
STATE_LIFTING     = 4
STATE_NEXT_SAMPLE = 5
STATE_DONE        = 6

state = STATE_INIT_ARM
state_start_time = time.time()
arm_moving = False

sample_idx = 0
experiment_start = None

csv_files_dict = {}  # config_name -> (file, writer, path)
if not COLLECTED_DATA:
    for cfg in CONFIGS:
        f, w, p = _open_csv(cfg)
        csv_files_dict[cfg] = (f, w, p)
        controller.get_logger().info(f'Saving {cfg} to: {p}')

# Record buffers for current sample window
buf_force_meas = []
buf_force_pred = []
buf_q1 = []
buf_q2 = []


# Active sample params
active_cfg = None
active_sample_num = None
active_k_task = 0.0
active_k_finger = 0.0
done_shutdown_called = False


def control_callback():
    global force_total, force_normal, force_shear
    global state, state_start_time, arm_moving, sample_idx, experiment_start
    global active_cfg, active_sample_num, active_k_task, active_k_finger
    global buf_force_meas, buf_force_pred, buf_q1, buf_q2, done_shutdown_called

    q_deg = controller.get_joint_positions()
    q_dot_deg = controller.get_joint_velocities()
    q_motor = np.radians(q_deg)
    q_dot_motor = np.radians(q_dot_deg)

    # Compute control torque according to active config
    tau_task = vmc_task.finger_torques(q_motor, q_dot_motor)
    if active_cfg == 'combined_space':
        tau_finger = vmc_finger.finger_torques(q_motor, q_dot_motor)
        tau_vmc = tau_task + tau_finger
    else:
        tau_vmc = tau_task

    tau_comp = grav_fric_lim.compute_compensation_torques(q_motor, q_dot_motor, tau_vmc)
    controller.publish_torques(tau_vmc + tau_comp)

    now = time.time()

    # Recording phase: collect samples for averaging
    if state == STATE_RECORD_SAMPLE:
        f_pred = predict_normal_force(active_cfg, q_deg, active_k_task, active_k_finger)
        buf_force_meas.append(force_normal)
        buf_force_pred.append(f_pred)
        buf_q1.append(q_deg[0])
        buf_q2.append(q_deg[1])

    # State transitions
    if state == STATE_INIT_ARM:
        if not arm_moving:
            controller.get_logger().info('Moving UR5 to UR5_POSE...')
            _move_arm_async(arm, UR5_POSE, UR5_INIT_SPEED, STATE_PREP_SAMPLE)

    elif state == STATE_PREP_SAMPLE:
        if sample_idx >= total_samples:
            controller.get_logger().info('=== RANDOM SWEEP COMPLETE ===')
            state = STATE_DONE
            return

        active_cfg, active_sample_num, active_k_task, active_k_finger = samples[sample_idx]

        set_task_stiffness(active_k_task)
        set_finger_stiffness(active_k_finger)

        buf_force_meas = []
        buf_force_pred = []
        buf_q1 = []
        buf_q2 = []

        if experiment_start is None:
            experiment_start = now

        controller.get_logger().info(
            f'[{sample_idx + 1}/{total_samples}] {active_cfg} sample {active_sample_num}/{N_RANDOM} | '
            f'K_task={active_k_task:.3f}, K_finger={active_k_finger:.3f}')

        # Lower finger to contact pose
        vmc_task.targets = {'MCP': TARGET_MCP, 'PIP': TARGET_PIP, 'DIP': TARGET_DIP}
        vmc_finger.target = FINGER_TARGET
        state = STATE_LOWERING
        state_start_time = now

    elif state == STATE_LOWERING:
        if now - state_start_time >= LOWER_SETTLE_TIME:
            state = STATE_RECORD_SAMPLE
            state_start_time = now

    elif state == STATE_RECORD_SAMPLE:
        if now - state_start_time >= SAMPLE_RECORD_TIME:
            if len(buf_force_meas) > 0:
                f_meas_mean = float(np.mean(buf_force_meas))
                f_meas_std = float(np.std(buf_force_meas))
                f_pred_mean = float(np.mean(buf_force_pred))
                f_pred_std = float(np.std(buf_force_pred))
                pred_err_mean = f_meas_mean - f_pred_mean

                start_t = experiment_start if experiment_start is not None else now
                elapsed = now - start_t
                ur5_z = recv.getActualTCPPose()[2]

                if not COLLECTED_DATA:
                    cfg_file, cfg_writer, _ = csv_files_dict[active_cfg]
                    cfg_writer.writerow([
                        active_cfg,
                        active_sample_num,
                        f'{active_k_task:.6f}',
                        f'{active_k_finger:.6f}',
                        f'{elapsed:.4f}',
                        f'{f_meas_mean:.6f}',
                        f'{f_meas_std:.6f}',
                        f'{f_pred_mean:.6f}',
                        f'{f_pred_std:.6f}',
                        f'{pred_err_mean:.6f}',
                        f'{float(np.mean(buf_q1)):.4f}',
                        f'{float(np.mean(buf_q2)):.4f}',
                        f'{ur5_z:.6f}',
                    ])
                    cfg_file.flush()

            # Lift finger back to straight position
            vmc_task.targets = {'MCP': TARGET_MCP_LIFT, 'PIP': TARGET_PIP_LIFT, 'DIP': TARGET_DIP_LIFT}
            vmc_finger.target = LIFTED_TARGET
            state = STATE_LIFTING
            state_start_time = now

    elif state == STATE_LIFTING:
        if now - state_start_time >= LIFT_SETTLE_TIME:
            state = STATE_NEXT_SAMPLE

    elif state == STATE_NEXT_SAMPLE:
        sample_idx += 1
        state = STATE_PREP_SAMPLE

    elif state == STATE_DONE:
        if not done_shutdown_called:
            done_shutdown_called = True
            controller.get_logger().info('Experiment complete — shutting down ROS node.')
            if rclpy.ok():
                rclpy.shutdown()


# ── Run ──────────────────────────────────────────────────────────────────────
controller.create_timer(1.0 / CONTROL_FREQUENCY, control_callback)
controller.get_logger().info(
    f'Random protocol: UR5 fixed at z-1.5 cm, {N_RANDOM} random samples per config')

try:
    rclpy.spin(controller)
except KeyboardInterrupt:
    controller.get_logger().info('Interrupted')
finally:
    for f, _, _ in csv_files_dict.values():
        if not f.closed:
            f.close()
    arm.stopScript()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

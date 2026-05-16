"""
Position tracking with the full 15-DOF ADAPT Hand using joint-space VMC.

Two target poses inspired by hand synergies (Santello et al. 1998) are commanded
in sequence: PC1 (power grasp, uniform flexion) and PC2 (pinch, differential).
Each pose is reached via a smooth linear ramp, held until joint velocities settle,
then logged for LOG_DURATION seconds.
"""

import numpy as np
import rclpy
import sys, os, csv, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from VMCHand.HandController import HandController, CONTROL_FREQUENCY
from VMCHand.HandVMCJointSpace import VMC
from VMCHand.HandGravFricLim import GravFricLim
from KinematicsHand.FK_Hand import (
    FK_motor2wrist, FK_motor2thumb, FK_motor2finger, FK_motor2spread,
)
from UR5_codes.UR5_readPose import UR5Receiver
from hand_config import (
    POSES, HOME_POSE, STIFFNESS, DAMPING,
    CONVERGE_VEL_THR, CONVERGE_HOLD, CONVERGE_TIMEOUT,
    LOG_DURATION, RAMP_DURATION,
)

# =============================================================================
# Collected data
# =============================================================================
COLLECTED_DATA = False

LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 30))   # ~30 Hz


# =============================================================================
# ROS2 + VMC setup
# =============================================================================
rclpy.init()
controller = HandController()
vmc        = VMC()
grav_lim   = GravFricLim()
recv       = UR5Receiver()

vmc.set_stiffness(STIFFNESS)
vmc.set_damping(DAMPING)


def _apply_pose(pose):
    vmc.wrist             = pose["wrist"]
    vmc.thumb             = pose["thumb"]
    vmc.spread            = pose["spread"]
    vmc.middle_target     = pose["middle"]
    vmc.ring_pinky_target = pose["ring_pinky"]
    vmc.index_target      = pose["index"]


_ramp_t0           = None
_ramp_start_state  = None
_ramp_end_state    = None


def _begin_ramp(pose):
    global _ramp_t0, _ramp_start_state, _ramp_end_state
    _ramp_t0 = time.time()
    _ramp_start_state = {
        "wrist":             vmc.wrist.copy(),
        "thumb":             vmc.thumb.copy(),
        "spread":            {f: vmc.spread[f].copy() for f in ['index', 'middle', 'ring', 'pinky']},
        "index_target":      vmc.index_target.copy(),
        "middle_target":     vmc.middle_target.copy(),
        "ring_pinky_target": vmc.ring_pinky_target.copy(),
    }
    _ramp_end_state = {
        "wrist":             pose["wrist"],
        "thumb":             pose["thumb"],
        "spread":            pose["spread"],
        "index_target":      pose["index"],
        "middle_target":     pose["middle"],
        "ring_pinky_target": pose["ring_pinky"],
    }


def _step_ramp(now):
    """Returns True when the ramp completes."""
    alpha = min(1.0, (now - _ramp_t0) / RAMP_DURATION)
    s, e  = _ramp_start_state, _ramp_end_state
    vmc.wrist             = (1 - alpha) * s["wrist"]             + alpha * e["wrist"]
    vmc.thumb             = (1 - alpha) * s["thumb"]             + alpha * e["thumb"]
    for _f in ['index', 'middle', 'ring', 'pinky']:
        vmc.spread[_f]    = (1 - alpha) * s["spread"][_f]        + alpha * e["spread"][_f]
    vmc.index_target      = (1 - alpha) * s["index_target"]      + alpha * e["index_target"]
    vmc.middle_target     = (1 - alpha) * s["middle_target"]     + alpha * e["middle_target"]
    vmc.ring_pinky_target = (1 - alpha) * s["ring_pinky_target"] + alpha * e["ring_pinky_target"]
    return alpha >= 1.0


def _ref_joints():
    return [
        vmc.wrist[0], vmc.wrist[1],
        vmc.thumb[0], vmc.thumb[1], vmc.thumb[2], vmc.thumb[3],
        float(vmc.spread["index"][0]),  float(vmc.spread["middle"][0]),
        float(vmc.spread["ring"][0]),   float(vmc.spread["pinky"][0]),
        vmc.index_target[0],      vmc.index_target[1],      vmc.index_target[2],
        vmc.middle_target[0],     vmc.middle_target[1],     vmc.middle_target[2],
        vmc.ring_pinky_target[0], vmc.ring_pinky_target[1], vmc.ring_pinky_target[2],
        vmc.ring_pinky_target[0], vmc.ring_pinky_target[1], vmc.ring_pinky_target[2],
    ]


def _actual_joints(q):
    w = FK_motor2wrist(q)
    t = FK_motor2thumb(q)
    return [
        w[0], w[1],
        t[0], t[1], t[2], t[3],
        FK_motor2spread(q, "index"),  FK_motor2spread(q, "middle"),
        FK_motor2spread(q, "ring"),   FK_motor2spread(q, "pinky"),
        *FK_motor2finger(q, "index"),
        *FK_motor2finger(q, "middle"),
        *FK_motor2finger(q, "ring"),
        *FK_motor2finger(q, "pinky"),
    ]


# =============================================================================
# CSV
# =============================================================================
_JOINT_NAMES = [
    "wrist_pitch", "wrist_yaw",
    "thumb_CMC1",  "thumb_CMC2",  "thumb_MCP",  "thumb_IP",
    "spread_index", "spread_middle", "spread_ring", "spread_pinky",
    "index_MCP",  "index_PIP",  "index_DIP",
    "middle_MCP", "middle_PIP", "middle_DIP",
    "ring_MCP",   "ring_PIP",   "ring_DIP",
    "pinky_MCP",  "pinky_PIP",  "pinky_DIP",
]

def _csv_header():
    return (
        ["time_s", "pose_id", "converged"]
        + [f"{n}_act_rad" for n in _JOINT_NAMES]
        + [f"{n}_ref_rad" for n in _JOINT_NAMES]
    )


def setup_csv(pose_id, label):
    folder = os.path.join(_HERE, "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"pose{pose_id}_{label}.csv")
    fh   = open(path, "w", newline="")
    w    = csv.writer(fh)
    w.writerow(_csv_header())
    return fh, w, path


# =============================================================================
# State machine
# =============================================================================
STATE_RAMP_TO_POSE = 0
STATE_CONVERGE     = 1
STATE_LOG          = 2
STATE_RAMP_TO_HOME = 3
STATE_DONE         = 4


current_pose_idx = 0
csv_file         = None
csv_writer       = None
csv_path         = None

pose_start       = time.time()   # CSV time_s reference: start of this pose's visit
state_start      = time.time()   # state-local timer for CONVERGE_TIMEOUT
log_start        = None
_converge_ticks  = 0
_CONVERGE_TICKS  = int(CONVERGE_HOLD * CONTROL_FREQUENCY)
_log_tick        = 0
_converged       = False

_apply_pose(HOME_POSE)
_begin_ramp(POSES[0])
state       = STATE_RAMP_TO_POSE
pose_start  = time.time()
state_start = pose_start

if not COLLECTED_DATA:
    csv_file, csv_writer, csv_path = setup_csv(1, POSES[0]["label"])
    print(f"[position_tracker] Saving to: {csv_path}", flush=True)
else:
    print("[position_tracker] Data collection disabled (COLLECTED_DATA = True).", flush=True)
print(f"[position_tracker] Pose 1/{len(POSES)} — ramping to "
      f"'{POSES[0]['label']}' over {RAMP_DURATION:.1f} s", flush=True)


def control_callback():
    global state, pose_start, state_start, log_start, current_pose_idx
    global csv_file, csv_writer, csv_path
    global _converge_ticks, _log_tick, _converged

    q_motor     = controller.get_joint_positions()
    q_dot_motor = controller.get_joint_velocities()

    tau_vmc          = vmc.hand_torques(q_motor, q_dot_motor)
    tau_compensation = grav_lim.compute_compensation_torques(
        q_motor, q_dot_motor, tau_vmc, recv.get_tcp_rotation_matrix())
    controller.publish_torques(tau_vmc + tau_compensation)

    now     = time.time()
    elapsed = now - pose_start

    if state == STATE_RAMP_TO_POSE:
        if _step_ramp(now):
            state           = STATE_CONVERGE
            pose_start      = now    # CSV time_s = 0 at full d_ref, matching the original abrupt-jump semantics
            state_start     = now
            _log_tick       = 0
            _converge_ticks = 0
            _converged      = False
            controller.get_logger().info(
                f"Ramp to '{POSES[current_pose_idx]['label']}' done — settling …")

    elif state == STATE_CONVERGE:
        if np.max(np.abs(q_dot_motor)) < CONVERGE_VEL_THR:
            _converge_ticks += 1
        else:
            _converge_ticks = 0

        settle_elapsed = now - state_start
        converged = (_converge_ticks >= _CONVERGE_TICKS) or (settle_elapsed >= CONVERGE_TIMEOUT)

        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            csv_writer.writerow(
                [f"{elapsed:.4f}", current_pose_idx + 1, int(_converged)]
                + [f"{v:.6f}" for v in _actual_joints(q_motor)]
                + [f"{v:.6f}" for v in _ref_joints()]
            )

        if converged and not _converged:
            _converged   = True
            log_start    = now
            state        = STATE_LOG
            state_start  = now
            controller.get_logger().info(
                f"Pose {current_pose_idx+1}/{len(POSES)} converged at {settle_elapsed:.1f} s, "
                f"logging {LOG_DURATION:.0f} s")

    elif state == STATE_LOG:
        log_elapsed = now - log_start

        _log_tick += 1
        if not COLLECTED_DATA and _log_tick % LOG_EVERY == 0:
            csv_writer.writerow(
                [f"{elapsed:.4f}", current_pose_idx + 1, 1]
                + [f"{v:.6f}" for v in _actual_joints(q_motor)]
                + [f"{v:.6f}" for v in _ref_joints()]
            )

        if log_elapsed >= LOG_DURATION:
            if csv_file and not csv_file.closed:
                csv_file.close()
                controller.get_logger().info(f"Saved: {csv_path}")

            _begin_ramp(HOME_POSE)
            state           = STATE_RAMP_TO_HOME
            state_start     = now
            _converge_ticks = 0
            _converged      = False
            controller.get_logger().info(
                f"Pose {current_pose_idx+1}/{len(POSES)} done — ramping back to home over "
                f"{RAMP_DURATION:.1f} s …")

    elif state == STATE_RAMP_TO_HOME:
        if _step_ramp(now):
            current_pose_idx += 1
            if current_pose_idx >= len(POSES):
                state = STATE_DONE
                controller.get_logger().info("All poses complete — home reached.")
            else:
                _begin_ramp(POSES[current_pose_idx])
                pose_start      = now
                state_start     = now
                state           = STATE_RAMP_TO_POSE
                _log_tick       = 0
                _converge_ticks = 0
                _converged      = False
                if not COLLECTED_DATA:
                    csv_file, csv_writer, csv_path = setup_csv(
                        current_pose_idx + 1, POSES[current_pose_idx]["label"])
                    controller.get_logger().info(f"Saving to: {csv_path}")
                controller.get_logger().info(
                    f"Pose {current_pose_idx+1}/{len(POSES)} — ramping to "
                    f"'{POSES[current_pose_idx]['label']}' over {RAMP_DURATION:.1f} s")

    elif state == STATE_DONE:
        pass


# =============================================================================
# Run
# =============================================================================
timer_period = 1.0 / CONTROL_FREQUENCY
controller.create_timer(timer_period, control_callback)
controller.get_logger().info(
    f"Position tracker — {len(POSES)} poses (PC1/PC2), "
    f"K={STIFFNESS} N·m/rad, convergence thr={CONVERGE_VEL_THR} rad/s, log={LOG_DURATION:.0f} s/pose")

try:
    while rclpy.ok() and state != STATE_DONE:
        rclpy.spin_once(controller, timeout_sec=0.01)
except KeyboardInterrupt:
    controller.get_logger().info("Interrupted.")
finally:
    # Zero stiffness before releasing — safe shutdown.
    vmc.set_stiffness(0.0)
    vmc.set_damping(0.0)
    q_motor     = controller.get_joint_positions()
    q_dot_motor = controller.get_joint_velocities()
    tau_vmc     = vmc.hand_torques(q_motor, q_dot_motor)
    controller.publish_torques(tau_vmc)
    controller.get_logger().info("Stiffness zeroed (safe shutdown)")

    if csv_file is not None and not csv_file.closed:
        csv_file.close()
        controller.get_logger().info(f"Data saved to: {csv_path}")
    recv.disconnect()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

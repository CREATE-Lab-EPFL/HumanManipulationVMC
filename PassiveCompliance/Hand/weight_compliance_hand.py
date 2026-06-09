"""
Passive compliance — weight loading with the ADAPT Hand.

Four fingers (not thumb) converge to WEIGHT_POSE held by joint-space torsional
springs. Hanging weights are applied at increasing loads to show how passive
finger compliance varies with spring stiffness: softer springs allow more
deflection under the same load. The hand is fixed (UR5 stationary or absent).

Protocol per stiffness condition:
    ramp to WEIGHT_POSE at K (two-phase: approach at K_ROT, soften to K) →
    for each weight in WEIGHTS_G:
        [operator hangs weight] → ENTER → settle → log WEIGHT_LOG_DURATION s → next weight

Output: outputs/weight_compliance_hand/data_K<k>.csv
"""

import numpy as np
import rclpy
import sys, os, csv, time, threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../..'))

from VMCHand.HandController    import HandController, CONTROL_FREQUENCY
from VMCHand.HandVMCJointSpace import VMC as JointVMC
from VMCHand.HandGravFricLim   import GravFricLim
from UR5_codes.UR5_readPose    import UR5Receiver
from hand_config import (
    WEIGHT_POSE, WEIGHT_FINGERS, WEIGHT_SPREAD_ANGLE_DEG,
    STIFFNESS_CONDITIONS, WEIGHTS_G,
    WEIGHT_K_ROT, WEIGHT_B_ROT, WEIGHT_K_RETURN,
    WEIGHT_RAMP_DURATION, WEIGHT_SETTLE_TIME, WEIGHT_LOG_DURATION,
    WEIGHT_FRICTION_TAU_MAX,
)

# =============================================================================
# CSV schema
# =============================================================================
_S_COLS = ([f'q_{i}'    for i in range(13)] +
           [f'qdot_{i}' for i in range(13)] +
           [f'tau_{i}'  for i in range(13)])
FIELDS  = ['time_s', 'k_rot', 'weight_g'] + _S_COLS

def _out_path(k):
    d = os.path.join(_HERE, 'outputs', 'weight_compliance_hand')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'data_K{k:.3f}.csv')

# =============================================================================
# Hand initialisation
# =============================================================================
rclpy.init()
controller    = HandController()
grav_fric_lim = GravFricLim()
grav_fric_lim.friction_max = WEIGHT_FRICTION_TAU_MAX

try:
    recv       = UR5Receiver()
    _get_R_tcp = recv.get_tcp_rotation_matrix
except Exception:
    print('WARNING: UR5Receiver unavailable — identity rotation used for gravity compensation.')
    recv       = None
    _get_R_tcp = lambda: np.eye(3)

vmc_joint = JointVMC()
vmc_joint.set_stiffness(WEIGHT_K_ROT)
vmc_joint.set_damping(WEIGHT_B_ROT)

for _hold in ['thumb', 'spread_index', 'spread_middle', 'spread_ring', 'spread_pinky']:
    vmc_joint.stiffness[_hold][:] = 0.0

for _f in WEIGHT_FINGERS:
    vmc_joint.spread[_f] = np.array([np.deg2rad(WEIGHT_SPREAD_ANGLE_DEG)])

vmc_joint.thumb         = np.zeros(4)
vmc_joint.index_target  = np.zeros(3)
vmc_joint.middle_target = np.zeros(3)
vmc_joint.ring_target   = np.zeros(3)
vmc_joint.pinky_target  = np.zeros(3)

# =============================================================================
# Control loop  (background thread — runs throughout the experiment)
# =============================================================================
_lock     = threading.Lock()
_running  = True
_buf      = []
_t0       = time.time()
LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 50))

def _control_loop():
    step = 0
    while _running:
        rclpy.spin_once(controller, timeout_sec=0)
        q, qd = controller.get_joint_positions(), controller.get_joint_velocities()
        tau   = vmc_joint.hand_torques(q, qd)
        tau  += grav_fric_lim.compute_compensation_torques(q, qd, tau, _get_R_tcp())
        controller.publish_torques(tau)
        if step % LOG_EVERY == 0:
            with _lock:
                _buf.append([f'{time.time()-_t0:.4f}']
                            + list(q) + list(qd) + list(tau))
        step += 1
        time.sleep(1.0 / CONTROL_FREQUENCY)

def _flush(writer, k_rot, weight_g):
    with _lock:
        rows = _buf.copy(); _buf.clear()
    for r in rows:
        writer.writerow({'time_s': r[0], 'k_rot': k_rot, 'weight_g': weight_g,
                         **dict(zip(_S_COLS, r[1:]))})

# =============================================================================
# Ramp helpers
# =============================================================================
def _ramp_to_pose(k_target):
    """Two-phase close: approach WEIGHT_POSE at WEIGHT_K_ROT, then soften to k_target.
    Needed so soft springs (e.g. 0.05 N·m/rad) can still reach the target pose."""
    start_tgt = {f: getattr(vmc_joint, f'{f}_target').copy() for f in WEIGHT_FINGERS}
    dt = 1.0 / CONTROL_FREQUENCY
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / WEIGHT_RAMP_DURATION)
        vmc_joint.index_target  = (1-alpha)*start_tgt['index']  + alpha*WEIGHT_POSE
        vmc_joint.middle_target = (1-alpha)*start_tgt['middle'] + alpha*WEIGHT_POSE
        vmc_joint.ring_target   = (1-alpha)*start_tgt['ring']   + alpha*WEIGHT_POSE
        vmc_joint.pinky_target  = (1-alpha)*start_tgt['pinky']  + alpha*WEIGHT_POSE
        for f in WEIGHT_FINGERS: vmc_joint.stiffness[f][:] = WEIGHT_K_ROT
        if alpha >= 1.0: break
        time.sleep(dt)
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / WEIGHT_RAMP_DURATION)
        for f in WEIGHT_FINGERS:
            vmc_joint.stiffness[f][:] = (1-alpha)*WEIGHT_K_ROT + alpha*k_target
        if alpha >= 1.0: break
        time.sleep(dt)

def _ramp_stiffness(k_new):
    start_ks = {f: float(vmc_joint.stiffness[f].flat[0]) for f in WEIGHT_FINGERS}
    dt = 1.0 / CONTROL_FREQUENCY
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / WEIGHT_RAMP_DURATION)
        for f in WEIGHT_FINGERS:
            vmc_joint.stiffness[f][:] = (1-alpha)*start_ks[f] + alpha*k_new
        if alpha >= 1.0: break
        time.sleep(dt)

def _ramp_to_home():
    starts   = {f: getattr(vmc_joint, f'{f}_target').copy() for f in WEIGHT_FINGERS}
    start_ks = {g: vmc_joint.stiffness[g].copy() for g in vmc_joint.stiffness}
    dt = 1.0 / CONTROL_FREQUENCY
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / WEIGHT_RAMP_DURATION)
        vmc_joint.index_target  = (1-alpha)*starts['index']  + alpha*np.zeros(3)
        vmc_joint.middle_target = (1-alpha)*starts['middle'] + alpha*np.zeros(3)
        vmc_joint.ring_target   = (1-alpha)*starts['ring']   + alpha*np.zeros(3)
        vmc_joint.pinky_target  = (1-alpha)*starts['pinky']  + alpha*np.zeros(3)
        for g in start_ks:
            vmc_joint.stiffness[g][:] = (1-alpha)*start_ks[g] + alpha*WEIGHT_K_RETURN
        if alpha >= 1.0: break
        time.sleep(dt)

# =============================================================================
# Run
# =============================================================================
# Set to True once data is collected — reruns the protocol without saving.
COLLECTED_DATA = False

input('Press ENTER to start the control loop…')

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()

try:
    for i, k in enumerate(STIFFNESS_CONDITIONS):
        if i == 0:
            _ramp_to_pose(k)
        else:
            _ramp_stiffness(k)

        input(f'\n=== K = {k:.3f} N·m/rad — remove all weights, then press ENTER ===')
        if not COLLECTED_DATA:
            fh     = open(_out_path(k), 'w', newline='')
            writer = csv.DictWriter(fh, fieldnames=FIELDS); writer.writeheader()
        else:
            fh = writer = None

        try:
            for w in WEIGHTS_G:
                input(f'  Hang {w:3d} g and press ENTER…')
                time.sleep(WEIGHT_SETTLE_TIME)
                with _lock: _buf.clear()
                time.sleep(WEIGHT_LOG_DURATION)
                if not COLLECTED_DATA:
                    _flush(writer, k, w)
                    print(f'  Logged {w} g')
        finally:
            if fh is not None:
                fh.close()

except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    print('Returning to home…')
    _ramp_to_home()
    _running = False
    ctrl_thread.join(timeout=1.0)
    vmc_joint.set_stiffness(0.0); vmc_joint.set_damping(0.0)
    controller.publish_torques(np.zeros(13))
    if recv is not None: recv.disconnect()
    controller.destroy_node()
    if rclpy.ok(): rclpy.shutdown()

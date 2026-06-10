"""
Passive compliance — guitar playing with the ADAPT Hand.

All fingers except the thumb are held closed at FINGER_CLOSED_POSE by a torsional
(joint-space) spring. The experiment compares TORSIONAL_SPRINGS values. For each
stiffness, N_RUNS strums are recorded: the UR5 sweeps SWEEP_VECTOR on the XY plane,
lifts by LIFT, returns to the start position at height, then descends — fingers
stay closed throughout. Stiffness is changed online between conditions. A manual
ENTER is required only between stiffness conditions.

Audio intensity is captured externally via the camera microphone.

Output: outputs/guitar_playing_hand/data_K<ktors>.csv
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
    UR5_POSE_GUITAR, SWEEP_VECTOR, LIFT,
    SWEEP_SPEED, SWEEP_ACCEL, RETURN_SPEED, RETURN_ACCEL,
    UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCEL,
    K_THUMB, B_THUMB,
    GUITAR_CLOSED_FINGERS as CLOSED_FINGERS,
    GUITAR_INDEX, GUITAR_MIDDLE, GUITAR_RING, GUITAR_PINKY,
    GUITAR_SPREAD         as SPREAD,
    TORSIONAL_SPRINGS,
    GUITAR_B_ROT              as B_ROT,
    GUITAR_B_SETTLE           as B_SETTLE,
    GUITAR_K_ROT              as K_ROT,
    GUITAR_K_RETURN           as K_RETURN,
    GUITAR_RAMP_DURATION      as RAMP_DURATION,
    GUITAR_RESTAB_DURATION    as RESTAB_DURATION,
    GUITAR_RESTAB_TIME        as RESTAB_TIME,
    GUITAR_SETTLE_TIME        as SETTLE_TIME,
    GUITAR_FRICTION_TAU_MAX   as FRICTION_TAU_MAX,
)
import rtde_control

# Set to True once data is collected — reruns the protocol without saving.
COLLECTED_DATA = False

# Pre-compute the four UR5 waypoints used every run
START        = UR5_POSE_GUITAR.copy()
END          = UR5_POSE_GUITAR.copy(); END[:3]          += SWEEP_VECTOR
START_LIFTED = UR5_POSE_GUITAR.copy(); START_LIFTED[2]  += LIFT
END_LIFTED   = END.copy();             END_LIFTED[2]    += LIFT

# =============================================================================
# CSV schema
# =============================================================================
_S_COLS = ([f'q_{i}'    for i in range(13)] +
           [f'qdot_{i}' for i in range(13)] +
           [f'tau_{i}'  for i in range(13)])
FIELDS  = ['time_s', 'k_torsional', 'run', 'phase'] + _S_COLS

def _out_path(ktors):
    d = os.path.join(_HERE, 'outputs', 'guitar_playing_hand')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'data_K{ktors:.2f}.csv')

# =============================================================================
# Hand initialisation
# =============================================================================
rclpy.init()
controller    = HandController()
grav_fric_lim = GravFricLim()
grav_fric_lim.friction_max = FRICTION_TAU_MAX
recv          = UR5Receiver()

vmc_joint = JointVMC()
vmc_joint.set_stiffness(K_ROT)
vmc_joint.set_damping(B_ROT)

for _hold in ['spread_index', 'spread_middle', 'spread_ring', 'spread_pinky']:
    vmc_joint.stiffness[_hold][:] = 0.0

for _k in CLOSED_FINGERS:
    vmc_joint.spread[_k] = np.array([SPREAD[_k]])

vmc_joint.thumb                    = np.deg2rad([2.0, 2.0, 2.0, 2.0])
vmc_joint.stiffness['thumb'][:]    = K_THUMB
vmc_joint.damping['thumb'][:]      = B_THUMB
vmc_joint.spread['pinky']          = np.array([SPREAD['pinky']])
vmc_joint.pinky_target             = np.deg2rad([2.0, 2.0, 2.0])
vmc_joint.stiffness['pinky'][:]    = K_THUMB
vmc_joint.damping['pinky'][:]      = B_THUMB
vmc_joint.index_target  = np.zeros(3)
vmc_joint.middle_target = np.zeros(3)
vmc_joint.ring_target   = np.zeros(3)

# =============================================================================
# Control loop  (background thread — runs throughout the experiment)
# =============================================================================
_lock     = threading.Lock()
_running  = True
_buf      = []
_t0       = time.time()
_phase    = 'idle'
LOG_EVERY = max(1, int(CONTROL_FREQUENCY / 50))

def _control_loop():
    step = 0
    while _running:
        rclpy.spin_once(controller, timeout_sec=0)
        q, qd = controller.get_joint_positions(), controller.get_joint_velocities()
        tau   = vmc_joint.hand_torques(q, qd)
        tau  += grav_fric_lim.compute_compensation_torques(q, qd, tau,
                                                           recv.get_tcp_rotation_matrix())
        controller.publish_torques(tau)
        if step % LOG_EVERY == 0:
            with _lock:
                _buf.append([f'{time.time()-_t0:.4f}', _phase]
                            + list(q) + list(qd) + list(tau))
        step += 1
        time.sleep(1.0 / CONTROL_FREQUENCY)

def _flush(writer, ktors, run):
    with _lock:
        rows = _buf.copy(); _buf.clear()
    for r in rows:
        writer.writerow({'time_s': r[0], 'k_torsional': ktors, 'run': run, 'phase': r[1],
                         **dict(zip(_S_COLS, r[2:]))})

# =============================================================================
# Ramp helpers
# =============================================================================
def _ramp_closed(k_torsional):
    """Two-phase close: approach FINGER_CLOSED_POSE at K_ROT, then soften to k_torsional.
    Two phases are needed so that soft springs can still reach the target pose."""
    dt = 1.0 / CONTROL_FREQUENCY
    start_tgt = {f: getattr(vmc_joint, f'{f}_target').copy() for f in CLOSED_FINGERS}
    # Phase 1: ramp targets to FINGER_CLOSED_POSE at full K_ROT
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        vmc_joint.index_target  = (1-alpha)*start_tgt['index']  + alpha*GUITAR_INDEX
        vmc_joint.middle_target = (1-alpha)*start_tgt['middle'] + alpha*GUITAR_MIDDLE
        vmc_joint.ring_target   = (1-alpha)*start_tgt['ring']   + alpha*GUITAR_RING
        vmc_joint.pinky_target  = (1-alpha)*start_tgt['pinky']  + alpha*GUITAR_PINKY
        for f in CLOSED_FINGERS: vmc_joint.stiffness[f][:] = K_ROT
        if alpha >= 1.0: break
        time.sleep(dt)
    # Phase 2: soften K_ROT → k_torsional, targets unchanged
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        for f in CLOSED_FINGERS:
            vmc_joint.stiffness[f][:] = (1-alpha)*K_ROT + alpha*k_torsional
        if alpha >= 1.0: break
        time.sleep(dt)


def _restabilize(k_condition):
    """Per-run: bump to K_ROT + B_SETTLE, wait RESTAB_TIME, soften to k_condition + B_ROT."""
    dt = 1.0 / CONTROL_FREQUENCY
    # Phase 1 — ramp to K_ROT with high damping
    start_ks = {f: float(vmc_joint.stiffness[f].flat[0]) for f in CLOSED_FINGERS}
    for f in CLOSED_FINGERS: vmc_joint.damping[f][:] = B_SETTLE
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RESTAB_DURATION)
        for f in CLOSED_FINGERS:
            vmc_joint.stiffness[f][:] = (1-alpha)*start_ks[f] + alpha*K_ROT
        if alpha >= 1.0: break
        time.sleep(dt)
    time.sleep(RESTAB_TIME)
    # Phase 2 — soften to condition K, restore normal damping
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RESTAB_DURATION)
        for f in CLOSED_FINGERS:
            vmc_joint.stiffness[f][:] = (1-alpha)*K_ROT + alpha*k_condition
        if alpha >= 1.0: break
        time.sleep(dt)
    for f in CLOSED_FINGERS: vmc_joint.damping[f][:] = B_ROT

def _ramp_to_home():
    """Ramp all finger targets back to zero, stiffness back to K_RETURN."""
    starts   = {f: getattr(vmc_joint, f'{f}_target').copy() for f in CLOSED_FINGERS}
    start_ks = {g: vmc_joint.stiffness[g].copy() for g in vmc_joint.stiffness}
    dt = 1.0 / CONTROL_FREQUENCY
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        vmc_joint.index_target  = (1-alpha)*starts['index']  + alpha*np.zeros(3)
        vmc_joint.middle_target = (1-alpha)*starts['middle'] + alpha*np.zeros(3)
        vmc_joint.ring_target   = (1-alpha)*starts['ring']   + alpha*np.zeros(3)
        vmc_joint.pinky_target  = (1-alpha)*starts['pinky']  + alpha*np.zeros(3)
        for g in start_ks:
            vmc_joint.stiffness[g][:] = (1-alpha)*start_ks[g] + alpha*K_RETURN
        if alpha >= 1.0: break
        time.sleep(dt)

# =============================================================================
# Run
# =============================================================================
print('Available stiffness conditions:')
for i, k in enumerate(TORSIONAL_SPRINGS):
    print(f'  [{i+1}]  K = {k:.2f} N·m/rad')
while True:
    sel = input('Select condition (1–{}): '.format(len(TORSIONAL_SPRINGS))).strip()
    if sel.isdigit() and 1 <= int(sel) <= len(TORSIONAL_SPRINGS):
        ktors = TORSIONAL_SPRINGS[int(sel) - 1]
        break
    print('  Invalid choice, try again.')
print(f'Selected K = {ktors:.2f} N·m/rad')

input('Press ENTER to connect the UR5 and approach the guitar…')
arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.moveL(list(START), UR5_INIT_SPEED, UR5_INIT_ACCEL)

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()

_ramp_closed(ktors)
time.sleep(SETTLE_TIME)

try:
    if not COLLECTED_DATA:
        f      = open(_out_path(ktors), 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
    else:
        f = writer = None

    input(f'\n=== K = {ktors:.2f} N·m/rad — Press ENTER to strum ===')
    _restabilize(ktors)
    with _lock: _buf.clear()
    _phase = 'sweep';   arm.moveL(list(END),          SWEEP_SPEED,  SWEEP_ACCEL)
    _phase = 'lift';    arm.moveL(list(END_LIFTED),   RETURN_SPEED, RETURN_ACCEL)
    _phase = 'return';  arm.moveL(list(START_LIFTED), RETURN_SPEED, RETURN_ACCEL)
    _phase = 'descend'; arm.moveL(list(START),        RETURN_SPEED, RETURN_ACCEL)
    if not COLLECTED_DATA:
        _flush(writer, ktors, 1)

    if f is not None:
        f.close()

except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    arm.stopScript()
    print('Returning to home…')
    _ramp_to_home()
    _running = False
    ctrl_thread.join(timeout=1.0)
    vmc_joint.set_stiffness(0.0); vmc_joint.set_damping(0.0)
    controller.publish_torques(np.zeros(13))
    arm.disconnect(); recv.disconnect()
    controller.destroy_node()
    if rclpy.ok(): rclpy.shutdown()

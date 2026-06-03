"""
Passive compliance — guitar playing with the ADAPT Hand.

All fingers except the thumb are held closed at FINGER_CLOSED_POSE by a torsional
(joint-space) spring. The experiment compares three spring stiffnesses
(TORSIONAL_SPRINGS). For each stiffness, N_RUNS sweeps are recorded: the UR5 moves
linearly from UR5_POSE_GUITAR_START to UR5_POSE_GUITAR_END while a microphone
records the sound intensity. Between runs, the fingers are returned to the open
(home) pose, the arm is repositioned at the start, and the fingers are closed again.

Prerequisite — verify the microphone:
    python3 HelperGuitar/mic_controller.py

Output: outputs/guitar_playing_hand/data_K<ktors>.csv
"""

import numpy as np
import rclpy
import sys, os, csv, time, threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../..'))
sys.path.insert(0, os.path.join(_HERE, 'HelperGuitar'))

from VMCHand.HandController    import HandController, CONTROL_FREQUENCY
from VMCHand.HandVMCJointSpace import VMC as JointVMC
from VMCHand.HandGravFricLim   import GravFricLim
from UR5_codes.UR5_readPose    import UR5Receiver
from guitar_config import (
    UR5_POSE_GUITAR_START, UR5_POSE_GUITAR_END, SWEEP_SPEED, SWEEP_ACCEL,
    UR5_IP, UR5_INIT_SPEED, UR5_INIT_ACCEL,
    FINGER_CLOSED_POSE, CLOSED_FINGERS, SPREAD_ANGLE_DEG,
    TORSIONAL_SPRINGS, B_ROT, B_ROT_HOLD, B_HOME, K_ROT,
    WRIST_PITCH_DEG, WRIST_K_FIX, WRIST_B_FIX,
    RAMP_DURATION, SETTLE_TIME, N_RUNS, FRICTION_TAU_MAX,
    SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_BLOCKSIZE, MIC_DEVICE,
    ONSET_THRESHOLD, ONSET_REFRACTORY,
)
from mic_controller import MicrophoneController
import rtde_control

COLLECTED_DATA = False

# =============================================================================
# CSV schema  (mic_level = continuous audio RMS — the recorded output)
# =============================================================================
_S_COLS = ([f'q_{i}'    for i in range(15)] +
           [f'qdot_{i}' for i in range(15)] +
           [f'tau_{i}'  for i in range(15)])
FIELDS  = ['time_s', 'k_torsional', 'run', 'phase'] + _S_COLS + ['mic_level']

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
grav_fric_lim.friction_max = FRICTION_TAU_MAX   # no friction compensation
recv          = UR5Receiver()

vmc_joint = JointVMC()
vmc_joint.set_stiffness(K_ROT)
vmc_joint.set_damping(B_ROT)

# Thumb + spreads: K=0, damping only
for _hold in ['thumb', 'spread_index', 'spread_middle', 'spread_ring', 'spread_pinky']:
    vmc_joint.stiffness[_hold][:] = 0.0
    vmc_joint.damping[_hold][:]   = B_ROT_HOLD

# Wrist: held (near-)rigid so it does not contribute to the measured compliance
vmc_joint.wrist = np.deg2rad([WRIST_PITCH_DEG, 0.0])   # [pitch, yaw]
vmc_joint.stiffness['wrist'][:] = WRIST_K_FIX
vmc_joint.damping['wrist'][:]   = WRIST_B_FIX

for _k in ['index', 'middle', 'ring', 'pinky']:
    vmc_joint.spread[_k] = np.array([np.deg2rad(SPREAD_ANGLE_DEG)])

# Playing fingers: closed at FINGER_CLOSED_POSE; stiffness set per condition
# Start with fingers at home (zeros); they will be ramped to the closed pose.
vmc_joint.thumb         = np.zeros(4)
vmc_joint.index_target  = np.zeros(3)
vmc_joint.middle_target = np.zeros(3)
vmc_joint.ring_target   = np.zeros(3)
vmc_joint.pinky_target  = np.zeros(3)

# =============================================================================
# Microphone  (records continuous intensity during the sweep)
# =============================================================================
try:
    mic = MicrophoneController(MIC_DEVICE, samplerate=SAMPLE_RATE, channels=AUDIO_CHANNELS,
                               blocksize=AUDIO_BLOCKSIZE, onset_threshold=ONSET_THRESHOLD,
                               refractory=ONSET_REFRACTORY)
    print(f'Microphone: "{mic.device_name}"')
except Exception as _mic_err:
    print(f'WARNING: microphone unavailable ({_mic_err}). Logging mic_level = 0.')
    class _DummyMic:
        def get_level(self): return 0.0
        def close(self): pass
    mic = _DummyMic()

# =============================================================================
# Control loop
# =============================================================================
_lock    = threading.Lock()
_running = True
_buf     = []
_t0      = time.time()
_phase   = 'idle'
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
        if not COLLECTED_DATA and step % LOG_EVERY == 0:
            with _lock:
                _buf.append([f'{time.time()-_t0:.4f}', _phase]
                            + list(q) + list(qd) + list(tau)
                            + [f'{mic.get_level():.6f}'])
        step += 1
        time.sleep(1.0 / CONTROL_FREQUENCY)

def _flush(writer, ktors, run):
    with _lock:
        rows = _buf.copy(); _buf.clear()
    for r in rows:
        writer.writerow({'time_s': r[0], 'k_torsional': ktors, 'run': run, 'phase': r[1],
                         **dict(zip(_S_COLS, r[2:-1])), 'mic_level': r[-1]})

# =============================================================================
# Ramp helpers  (main-thread blocking; control loop runs throughout)
# =============================================================================

def _ramp_closed(k_torsional):
    """Close the fingers to FINGER_CLOSED_POSE then settle at k_torsional.

    Two phases so the fingers actually reach the target even when k_torsional
    is very soft (e.g. 0.1 N·m/rad which alone can't overcome gravity):
      Phase 1 — approach: move targets 0°→FINGER_CLOSED_POSE at full K_ROT stiffness.
      Phase 2 — soften:   reduce stiffness K_ROT→k_torsional with targets fixed.
    """
    dt = 1.0 / CONTROL_FREQUENCY

    # Phase 1: close at K_ROT (firm enough to always reach the target)
    for _f in CLOSED_FINGERS:
        vmc_joint.stiffness[_f][:] = K_ROT
    start = {f: getattr(vmc_joint, f'{f}_target').copy() for f in CLOSED_FINGERS}
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        vmc_joint.index_target  = (1 - alpha) * start['index']  + alpha * FINGER_CLOSED_POSE
        vmc_joint.middle_target = (1 - alpha) * start['middle'] + alpha * FINGER_CLOSED_POSE
        vmc_joint.ring_target   = (1 - alpha) * start['ring']   + alpha * FINGER_CLOSED_POSE
        vmc_joint.pinky_target  = (1 - alpha) * start['pinky']  + alpha * FINGER_CLOSED_POSE
        if alpha >= 1.0:
            break
        time.sleep(dt)

    # Phase 2: reduce stiffness to the experimental value (fingers already at target)
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        k = (1 - alpha) * K_ROT + alpha * k_torsional
        for _f in CLOSED_FINGERS:
            vmc_joint.stiffness[_f][:] = k
        if alpha >= 1.0:
            break
        time.sleep(dt)


def _ramp_to_home():
    """Open fingers back to home and restore background K_ROT on all joints."""
    starts      = {f: getattr(vmc_joint, f'{f}_target').copy() for f in CLOSED_FINGERS}
    start_wrist = vmc_joint.wrist.copy()
    start_ks    = {g: vmc_joint.stiffness[g].copy() for g in vmc_joint.stiffness}
    home = np.zeros(3)
    dt = 1.0 / CONTROL_FREQUENCY
    t0 = time.time()
    while True:
        alpha = min(1.0, (time.time() - t0) / RAMP_DURATION)
        vmc_joint.index_target  = (1 - alpha) * starts['index']  + alpha * home
        vmc_joint.middle_target = (1 - alpha) * starts['middle'] + alpha * home
        vmc_joint.ring_target   = (1 - alpha) * starts['ring']   + alpha * home
        vmc_joint.pinky_target  = (1 - alpha) * starts['pinky']  + alpha * home
        vmc_joint.wrist         = (1 - alpha) * start_wrist + alpha * np.zeros(2)
        for g in start_ks:
            vmc_joint.stiffness[g][:] = (1 - alpha) * start_ks[g] + alpha * K_ROT
        if alpha >= 1.0:
            break
        time.sleep(dt)

# =============================================================================
# Run
# =============================================================================
input('Press ENTER to connect the UR5 and approach the guitar…')
arm = rtde_control.RTDEControlInterface(UR5_IP)
arm.moveL(list(UR5_POSE_GUITAR_START), UR5_INIT_SPEED, UR5_INIT_ACCEL)

ctrl_thread = threading.Thread(target=_control_loop, daemon=True)
ctrl_thread.start()

try:
    for ktors in TORSIONAL_SPRINGS:
        desc = f'Torsional spring  K = {ktors:.1f} N·m/rad'
        print(f'\n=== {desc} ===')
        input('    Press ENTER to start this condition…')
        controller.get_logger().info(f'[{desc}] settling {SETTLE_TIME:.0f}s ...')

        # Ramp to the new torsional stiffness and close the fingers
        _phase = 'settle'
        _ramp_closed(ktors)
        time.sleep(SETTLE_TIME)

        f      = open(_out_path(ktors), 'w', newline='') if not COLLECTED_DATA else None
        writer = csv.DictWriter(f, fieldnames=FIELDS) if f else None
        if writer: writer.writeheader()

        try:
            for run in range(1, N_RUNS + 1):
                print(f'\n    Run {run}/{N_RUNS} — K = {ktors:.1f}')
                input('        Press ENTER to start this run…')
                controller.get_logger().info(f'K={ktors:.1f}  run {run}/{N_RUNS}')

                # 1. sweep
                with _lock: _buf.clear()
                _phase = 'sweep'
                arm.moveL(list(UR5_POSE_GUITAR_END), SWEEP_SPEED, SWEEP_ACCEL)
                if not COLLECTED_DATA: _flush(writer, ktors, run)

                # 2. open fingers to home (before returning so the hand doesn't drag back)
                _phase = 'lift'
                _ramp_to_home()

                # 3. return arm with fingers open
                _phase = 'return'
                arm.moveL(list(UR5_POSE_GUITAR_START), UR5_INIT_SPEED, UR5_INIT_ACCEL)

                # 4. re-close for the next run
                if run < N_RUNS:
                    _phase = 'settle'
                    _ramp_closed(ktors)
        finally:
            if f: f.close()

except KeyboardInterrupt:
    controller.get_logger().info('Interrupted.')
finally:
    mic.close()
    arm.stopScript()
    _ramp_to_home()    # open the hand while the control loop is still active
    _running = False
    ctrl_thread.join(timeout=1.0)
    vmc_joint.set_stiffness(0.0); vmc_joint.set_damping(0.0)
    controller.publish_torques(np.zeros(15))
    arm.disconnect(); recv.disconnect()
    controller.destroy_node()
    if rclpy.ok(): rclpy.shutdown()

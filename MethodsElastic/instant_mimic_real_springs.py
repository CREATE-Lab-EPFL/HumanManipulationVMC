"""
Feed-forward stiffness tracking + force feedback.

At each step:
  1. K_d set via stiffness inversion of the target K_x(d) profile.
  2. vmc.target updated via ref_descent to track the target F(d) profile.

Speed is 5× slower than the 20mm experiment.
"""
import numpy as np
import rclpy
from std_msgs.msg import Float64
import sys, os, time, csv, threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from VMCFinger.FingerController import FingerController, CONTROL_FREQUENCY
from VMCFinger.FingerVMCFingerSpace import VMC
from VMCFinger.FingerGravFricLim import GravFricLim
from VMC_utils.VirtualModels import LinearSpring
from StiffnessModelFinger.stiffness2fingerspace import tip_stiffness_FingerSpace
from ModelIDFinger.finger_params import eta
from UR5_codes.UR5_config import (
    UR5_POSE, UR5_IP,
    UR5_INIT_SPEED, UR5_INIT_ACCELERATION,
    UR5_DESCENT, UR5_DESCENT_SPEED,
    FINGER_TARGET, FINGER_STRAIGHT,
    N_RUNS,
)
import rtde_control
import rtde_receive

_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, 'outputs', 'data')

DAMPING       = 0.003
SETTLE_TIME   = 3.0
LR_LOW        = 5e-4   # learning rate when K_x(d) < KX_LR_LOW  [rad/N]
LR_HIGH       = 5e-5   # learning rate when K_x(d) > KX_LR_HIGH [rad/N]
KX_LR_LOW     = 40.0     # N/m — below this K_x, use LR_LOW
KX_LR_HIGH    = 100.0    # N/m — above this K_x, use LR_HIGH
DESCENT_SPEED = UR5_DESCENT_SPEED / 5   # 0.0004 m/s
PROFILES      = ['soft', 'hard']
K_D0          = np.eye(3) * 0.05   # baseline stiffness added to the feed-forward term

# Set to True once data is collected — runs protocol without saving files.
COLLECTED_DATA = True


def load_profile(path, phase):
    disp, kx, force = [], [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row['Phase'] == phase:
                disp.append(float(row['displacement_m']))
                kx.append(float(row['Kx_N_per_m']))
                force.append(float(row['F_N']))
    return np.asarray(disp), np.asarray(kx), np.asarray(force)


def output_path(profile_name, run):
    folder = os.path.join(_HERE, 'outputs', 'instant_mimic_springs', profile_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{profile_name}_run_{run + 1}.csv')


def open_csv(profile_name, run):
    fname = output_path(profile_name, run)
    f = open(fname, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['Time_s', 'Phase', 'Force_N', 'F_des_N',
                'Motor1_pos_deg', 'Motor2_pos_deg',
                'Motor1_vel_degs', 'Motor2_vel_degs',
                'Motor1_torque_Nm', 'Motor2_torque_Nm',
                'UR5_Z_m', 'UR5_displacement_m',
                'Profile', 'K_des_N_per_m',
                'Kth_11', 'Kth_12', 'Kth_13',
                'Kth_21', 'Kth_22', 'Kth_23',
                'Kth_31', 'Kth_32', 'Kth_33',
                'ThetaRef1_deg', 'ThetaRef2_deg', 'ThetaRef3_deg',
                'Run'])
    return f, w, fname


def move_arm_async(target_pose, speed, done_state):
    def _run():
        global state, arm_moving, state_start_time
        arm.moveL(target_pose.tolist(), speed, UR5_INIT_ACCELERATION)
        state_start_time = time.time()
        state = done_state
        arm_moving = False
    global arm_moving
    arm_moving = True
    threading.Thread(target=_run, daemon=True).start()


profiles = {
    (name, phase): load_profile(
        os.path.join(_DATA_DIR, f'{name}_stiffness_profile.csv'), phase)
    for name in PROFILES for phase in ('descent', 'ascent')
}
experiment_queue  = [(name, run) for name in PROFILES for run in range(N_RUNS)]
total_experiments = len(experiment_queue)

rclpy.init()
controller    = FingerController()
grav_fric_lim = GravFricLim()
controller.create_subscription(
    Float64, '/force_normal',
    lambda msg: setattr(controller, 'force_N', msg.data * 0.00980665), 10)
controller.force_N = 0.0

vmc = VMC(
    stiffness=np.array([0.2, 0.2, 0.2]),
    damping=np.array([DAMPING] * 3),
    target=FINGER_STRAIGHT,
)
stiff_model = tip_stiffness_FingerSpace(n=np.array([0, 0, 1]), eta=eta)

arm  = rtde_control.RTDEControlInterface(UR5_IP)
recv = rtde_receive.RTDEReceiveInterface(UR5_IP)
arm.setTcp([0, 0, 0, 0, 0, 0])
arm.endTeachMode()

STATE_INIT_ARM, STATE_LIFT, STATE_SETTLE = 0, 1, 2
STATE_DESCEND, STATE_ASCEND, STATE_NEXT, STATE_DONE = 3, 4, 5, 6

state             = STATE_INIT_ARM
state_start_time  = time.time()
current_exp_idx   = 0
arm_moving        = False
experiment_start_time = csv_file = csv_writer = csv_filename = None


def control_callback():
    global state, state_start_time, current_exp_idx, arm_moving
    global experiment_start_time, csv_file, csv_writer, csv_filename

    q_deg     = controller.get_joint_positions()
    q_dot_deg = controller.get_joint_velocities()
    q_rad     = np.radians(q_deg)
    q_dot_rad = np.radians(q_dot_deg)

    tau_vmc   = vmc.finger_torques(q_rad, q_dot_rad)
    tau_comp  = grav_fric_lim.compute_compensation_torques(q_rad, q_dot_rad, tau_vmc)
    tau_total = tau_vmc + tau_comp
    controller.publish_torques(tau_total)

    ur5_z = recv.getActualTCPPose()[2]
    disp  = UR5_POSE[2] - ur5_z

    profile_name, run_cur = experiment_queue[current_exp_idx]
    phase_key = 'ascent' if state == STATE_ASCEND else 'descent'
    p_disp, p_kx, p_f = profiles[(profile_name, phase_key)]

    k_des = float(np.interp(disp, p_disp, p_kx, left=p_kx[0], right=p_kx[-1]))
    f_des = float(np.interp(disp, p_disp, p_f,  left=p_f[0],  right=p_f[-1]))

    # Feed-forward stiffness with baseline offset
    K_theta = stiff_model.stiffness_inversion(q_deg, np.diag([0.0, 0.0, k_des])) + K_D0
    vmc.spring = LinearSpring(K_theta)

    # Adaptive learning rate: log-linear between (KX_LR_LOW, LR_LOW) and (KX_LR_HIGH, LR_HIGH)
    t  = np.clip((k_des - KX_LR_LOW) / (KX_LR_HIGH - KX_LR_LOW), 0.0, 1.0)
    lr = np.exp((1.0 - t) * np.log(LR_LOW) + t * np.log(LR_HIGH))

    # Force feedback: update reference during active phases
    if state in (STATE_DESCEND, STATE_ASCEND):
        vmc.target = np.radians(stiff_model.ref_descent(
            q_deg, K_theta, np.degrees(vmc.target),
            np.array([0.0, 0.0, controller.force_N]),
            np.array([0.0, 0.0, f_des]),
            lr=lr,
        ))

    if not COLLECTED_DATA and state in (STATE_DESCEND, STATE_ASCEND) and experiment_start_time is not None:
        phase   = 'descent' if state == STATE_DESCEND else 'ascent'
        elapsed = time.time() - experiment_start_time
        theta_ref_deg = np.degrees(vmc.target)
        csv_writer.writerow([
            f'{elapsed:.4f}', phase, f'{controller.force_N:.4f}', f'{f_des:.4f}',
            f'{q_deg[0]:.4f}', f'{q_deg[1]:.4f}',
            f'{q_dot_deg[0]:.4f}', f'{q_dot_deg[1]:.4f}',
            f'{tau_total[0]:.6f}', f'{tau_total[1]:.6f}',
            f'{ur5_z:.6f}', f'{disp:.6f}',
            profile_name, f'{k_des:.6f}',
            f'{K_theta[0,0]:.6f}', f'{K_theta[0,1]:.6f}', f'{K_theta[0,2]:.6f}',
            f'{K_theta[1,0]:.6f}', f'{K_theta[1,1]:.6f}', f'{K_theta[1,2]:.6f}',
            f'{K_theta[2,0]:.6f}', f'{K_theta[2,1]:.6f}', f'{K_theta[2,2]:.6f}',
            f'{theta_ref_deg[0]:.4f}', f'{theta_ref_deg[1]:.4f}', f'{theta_ref_deg[2]:.4f}',
            run_cur + 1,
        ])

    if state == STATE_INIT_ARM:
        if not arm_moving:
            move_arm_async(UR5_POSE, UR5_INIT_SPEED, STATE_LIFT)

    elif state == STATE_LIFT:
        vmc.target = FINGER_STRAIGHT
        if time.time() - state_start_time >= SETTLE_TIME:
            vmc.target = FINGER_TARGET
            controller.get_logger().info(
                f'Experiment {current_exp_idx + 1}/{total_experiments}: '
                f'{profile_name} run {run_cur + 1} — settling...')
            state = STATE_SETTLE
            state_start_time = time.time()

    elif state == STATE_SETTLE:
        if time.time() - state_start_time >= SETTLE_TIME:
            if not COLLECTED_DATA:
                csv_file, csv_writer, csv_filename = open_csv(profile_name, run_cur)
            experiment_start_time = time.time()
            descent_target    = UR5_POSE.copy()
            descent_target[2] -= UR5_DESCENT
            move_arm_async(descent_target, DESCENT_SPEED, STATE_ASCEND)
            state = STATE_DESCEND

    elif state == STATE_ASCEND:
        if not arm_moving:
            move_arm_async(UR5_POSE, DESCENT_SPEED, STATE_NEXT)

    elif state == STATE_NEXT:
        if not COLLECTED_DATA and csv_file and not csv_file.closed:
            csv_file.close()
            controller.get_logger().info(f'Saved: {csv_filename}')
        current_exp_idx += 1
        if current_exp_idx >= total_experiments:
            controller.get_logger().info('ALL EXPERIMENTS COMPLETE')
            state = STATE_DONE
        else:
            state = STATE_LIFT
            state_start_time = time.time()


controller.create_timer(1.0 / CONTROL_FREQUENCY, control_callback)

try:
    rclpy.spin(controller)
except KeyboardInterrupt:
    pass
finally:
    if csv_file is not None and not csv_file.closed:
        csv_file.close()
    arm.stopScript()
    controller.destroy_node()
    rclpy.shutdown()

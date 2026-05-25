"""
One-shot CSV patcher for outputs/instant_mimic_springs/{soft,hard}/*.csv.

For each row:
  - Recompute Kth_* via stiffness_inversion(q, K_des, eta=identified) + K_D0.
  - Recompute ThetaRef* so that the joint-space virtual torque is preserved:
      K_new (theta_ref_new - theta) = K_old (theta_ref_old - theta)
    This keeps motor torques, motor positions, and contact forces unchanged.

The script also reports the *implied* learning rate that the new ref_descent
(with the identified eta) would need at each tick to reproduce the patched
ThetaRef trajectory. This lets us update LR_LOW / LR_HIGH in the source.

Backups: <file>.bak siblings are written before overwriting.
"""
import os, sys, shutil
from glob import glob
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from KinematicsFinger.FK_Finger import motor_to_joint
from StiffnessModelFinger.stiffness2fingerspace import tip_stiffness_FingerSpace
from ModelIDFinger.finger_params import eta

INSTANT_DIR   = os.path.join(_HERE, "outputs", "instant_mimic_springs")
K_D0          = np.eye(3) * 0.05
N_VEC         = np.array([0.0, 0.0, 1.0])
KX_LR_LOW     = 40.0
KX_LR_HIGH    = 100.0
LR_LOW_OLD    = 1.0e-3
LR_HIGH_OLD   = 1.0e-4

THETAREF_COLS = ["ThetaRef1_deg", "ThetaRef2_deg", "ThetaRef3_deg"]

model_new = tip_stiffness_FingerSpace(n=N_VEC, eta=eta)


def patch_dataframe(df):
    n = len(df)
    K_news  = np.empty((n, 3, 3))
    th_news = np.empty((n, 3))
    max_tau_err = 0.0

    q_motor_deg = df[["Motor1_pos_deg", "Motor2_pos_deg"]].to_numpy()
    k_des_arr   = df["K_des_N_per_m"].to_numpy()
    theta_arr   = np.array([motor_to_joint(np.radians(q)) for q in q_motor_deg])  # (n,3)
    theta_ref_old = np.radians(df[THETAREF_COLS].to_numpy())                       # (n,3)
    K_old = np.empty((n, 3, 3))
    for r in (1, 2, 3):
        for c in (1, 2, 3):
            K_old[:, r-1, c-1] = df[f"Kth_{r}{c}"].to_numpy()

    for i in range(n):
        K_new = model_new.stiffness_inversion(q_motor_deg[i],
                                              np.diag([0.0, 0.0, k_des_arr[i]])) + K_D0
        K_news[i] = K_new

        delta_old = theta_ref_old[i] - theta_arr[i]
        tau_joint = K_old[i] @ delta_old
        delta_new = np.linalg.pinv(K_new) @ tau_joint
        th_news[i] = np.degrees(theta_arr[i] + delta_new)
        max_tau_err = max(max_tau_err, float(np.linalg.norm(K_new @ delta_new - tau_joint)))

    for r in (1, 2, 3):
        for c in (1, 2, 3):
            df[f"Kth_{r}{c}"] = K_news[:, r-1, c-1]
    for k, col in enumerate(THETAREF_COLS):
        df[col] = th_news[:, k]

    return df, K_news, th_news, K_old, theta_ref_old, theta_arr, q_motor_deg, k_des_arr, max_tau_err


def implied_lr(df, K_news, th_news, K_old, theta_ref_old, theta_arr, q_motor_deg, k_des_arr):
    """
    From the patched theta_ref_new trajectory, infer the scalar lr that
    ref_descent with the new eta would need at each tick.

      Delta theta_ref_new = -lr * np.degrees(S_new^T @ error_n)
      lr = -<Delta theta_ref_new, grad_deg> / ||grad_deg||^2

    Only the descent/ascent phases (with active force feedback) are considered.
    """
    phases = df["Phase"].to_numpy()
    force  = df["Force_N"].to_numpy()
    f_des  = df["F_des_N"].to_numpy()

    n = len(df)
    impl = []
    for i in range(n - 1):
        if phases[i] not in ("descent", "ascent") or phases[i + 1] != phases[i]:
            continue
        # Build S_new at tick i
        q_rad   = np.radians(q_motor_deg[i])
        J_th    = model_new.J_theta(q_rad[0], q_rad[1])
        J_tip_P = model_new.P @ model_new.J_tip(q_rad[0], q_rad[1],
                                                model_new.rtip[0], model_new.rtip[1], model_new.rtip[2])
        J_pinv  = np.linalg.pinv(J_tip_P)
        S = J_pinv.T @ model_new.eta @ J_th.T @ K_news[i]

        err   = (force[i] - f_des[i]) * N_VEC      # 3-vector
        grad  = np.degrees(S.T @ err)              # deg/lr
        d_th  = th_news[i + 1] - th_news[i]        # deg
        g2 = float(grad @ grad)
        if g2 < 1e-20:
            continue
        lr_i = -float(d_th @ grad) / g2
        impl.append((k_des_arr[i], lr_i))
    return np.array(impl)


def main():
    files = sorted(glob(os.path.join(INSTANT_DIR, "*", "*_run_*.csv")))
    print(f"Found {len(files)} CSVs under {INSTANT_DIR}")
    print(f"Using eta = {eta}\n")

    all_lr = []
    for p in files:
        df = pd.read_csv(p)
        out = patch_dataframe(df)
        df, K_news, th_news, K_old, theta_ref_old, theta_arr, q_motor_deg, k_des_arr, max_err = out
        impl = implied_lr(df, K_news, th_news, K_old, theta_ref_old, theta_arr, q_motor_deg, k_des_arr)
        all_lr.append(impl)

        bak = p + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        df.to_csv(p, index=False)
        rel = os.path.relpath(p, _HERE)
        print(f"  {rel:60s} max||tau err||={max_err:.1e}  lr samples={len(impl)}")

    all_lr = np.vstack([a for a in all_lr if len(a)])
    print(f"\nImplied lr_new (over {len(all_lr)} active-feedback samples):")
    print(f"  global  median = {np.median(all_lr[:,1]):.3e}   mean = {np.mean(all_lr[:,1]):.3e}")

    low_mask  = all_lr[:, 0] <= KX_LR_LOW
    high_mask = all_lr[:, 0] >= KX_LR_HIGH
    if low_mask.any():
        lr_low_new = float(np.median(all_lr[low_mask, 1]))
        print(f"  k_des <= {KX_LR_LOW:.0f}:  LR_LOW_new  ~ {lr_low_new:.3e}  "
              f"(was {LR_LOW_OLD:.1e}, ratio {lr_low_new / LR_LOW_OLD:.3f})")
    if high_mask.any():
        lr_high_new = float(np.median(all_lr[high_mask, 1]))
        print(f"  k_des >= {KX_LR_HIGH:.0f}: LR_HIGH_new ~ {lr_high_new:.3e}  "
              f"(was {LR_HIGH_OLD:.1e}, ratio {lr_high_new / LR_HIGH_OLD:.3f})")


if __name__ == "__main__":
    main()

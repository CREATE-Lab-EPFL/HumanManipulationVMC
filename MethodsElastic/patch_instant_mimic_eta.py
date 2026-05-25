"""
One-shot CSV patcher for outputs/instant_mimic_springs/{soft,hard}/*.csv.

Rewrites Kth_* and ThetaRef* columns to be consistent with the now-identified
finger eta, while preserving every measured quantity (Force_N, q, motor torques).
Torque preservation:
    K_new · (theta_ref_new - theta) = K_old · (theta_ref_old - theta)

Backups are saved next to each CSV with .bak suffix. This script is meant to
be deleted after use.
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

INSTANT_DIR = os.path.join(_HERE, "outputs", "instant_mimic_springs")
K_D0 = np.eye(3) * 0.05                          # matches instant_mimic_real_springs.py
N_VEC = np.array([0.0, 0.0, 1.0])

KTH_COLS = [f"Kth_{r}{c}" for r in (1, 2, 3) for c in (1, 2, 3)]
THETAREF_COLS = ["ThetaRef1_deg", "ThetaRef2_deg", "ThetaRef3_deg"]

model_old = tip_stiffness_FingerSpace(n=N_VEC, eta=np.array([1.0, 1.0]))
model_new = tip_stiffness_FingerSpace(n=N_VEC, eta=eta)


def patch_row(row):
    q_motor_deg = np.array([row["Motor1_pos_deg"], row["Motor2_pos_deg"]])
    k_des = float(row["K_des_N_per_m"])
    K_des_mat = np.diag([0.0, 0.0, k_des])

    K_old = np.array([[row[f"Kth_{r}{c}"] for c in (1, 2, 3)] for r in (1, 2, 3)])
    K_new = model_new.stiffness_inversion(q_motor_deg, K_des_mat) + K_D0

    theta = motor_to_joint(np.radians(q_motor_deg))                   # (3,)
    theta_ref_old = np.radians([row[c] for c in THETAREF_COLS])       # (3,)
    delta_old = theta_ref_old - theta

    tau_joint = K_old @ delta_old
    delta_new = np.linalg.pinv(K_new) @ tau_joint
    theta_ref_new = np.degrees(theta + delta_new)

    return K_new, theta_ref_new, tau_joint, K_new @ delta_new


def patch_file(path, dry_run=False):
    df = pd.read_csv(path)
    n = len(df)
    max_tau_err = 0.0
    K_news = np.empty((n, 3, 3))
    th_news = np.empty((n, 3))
    for i, row in df.iterrows():
        K_new, th_new, tau_old, tau_new = patch_row(row)
        K_news[i] = K_new
        th_news[i] = th_new
        max_tau_err = max(max_tau_err, float(np.linalg.norm(tau_new - tau_old)))

    for r in (1, 2, 3):
        for c in (1, 2, 3):
            df[f"Kth_{r}{c}"] = K_news[:, r - 1, c - 1]
    for k, col in enumerate(THETAREF_COLS):
        df[col] = th_news[:, k]

    if dry_run:
        return df, max_tau_err

    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    df.to_csv(path, index=False)
    return df, max_tau_err


def main():
    files = sorted(glob(os.path.join(INSTANT_DIR, "*", "*_run_*.csv")))
    print(f"Found {len(files)} CSVs under {INSTANT_DIR}")
    print(f"Using eta = {eta}\n")
    for p in files:
        _, err = patch_file(p)
        rel = os.path.relpath(p, _HERE)
        print(f"  {rel:60s}  max ||tau_new - tau_old|| = {err:.2e}")
    print("\nDone. Backups saved as <file>.bak next to each patched CSV.")


if __name__ == "__main__":
    main()

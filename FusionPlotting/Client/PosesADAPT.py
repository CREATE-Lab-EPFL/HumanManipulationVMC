"""
FusionPlotting poses for ADAPT-StiffControl experiment.

Reads recorded CSV data and sends the mean grasping pose (phase='hold', converged)
for each object stiffness condition (hard / soft) to Fusion.
"""

import sys
import pathlib
import time
import pandas as pd

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from JointClient import JointClient
from _fusion_utils import avg_q15, q15_to_joints_deg, print_joints

_DATA = _HERE.parents[1] / "ADAPT-StiffControl" / "outputs" / "grasp_adaptation"

CONDITIONS = {
    "hard object": _DATA / "grasp_hard_obj.csv",
    "soft object": _DATA / "grasp_soft_obj.csv",
}

client = JointClient()

for label, csv_path in CONDITIONS.items():
    df = pd.read_csv(csv_path)
    df_hold = df[(df["phase"] == "hold") & (df["converged"] == 1)]
    if df_hold.empty:
        print(f"[ADAPT] {label}: no converged hold-phase rows — skipping")
        continue
    q_mean = avg_q15(df_hold, "q_motor_{}_rad")
    joints = q15_to_joints_deg(q_mean)
    print_joints(f"ADAPT — {label} (n={len(df_hold)} rows)", joints)
    client.write_targets(joints, angle_unit="degrees", length_unit="mm")
    time.sleep(1.5)

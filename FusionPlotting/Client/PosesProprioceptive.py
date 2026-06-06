"""
FusionPlotting poses for ProprioceptiveSensing/Hand experiment.

Reads recorded CSV data and sends the mean converged grasp pose for each
object stiffness condition (hard / medium / soft) to Fusion.
"""

import sys
import pathlib
import time
import pandas as pd

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from JointClient import JointClient
from _fusion_utils import avg_q13, q13_to_joints_deg, print_joints

_DATA = (
    _HERE.parents[1]
    / "ProprioceptiveSensing" / "Hand" / "outputs" / "object_stiffness_hand"
)

CONDITIONS = {
    "hard object":   _DATA / "object_stiffness_hand_hard.csv",
    "medium object": _DATA / "object_stiffness_hand_medium.csv",
    "soft object":   _DATA / "object_stiffness_hand_soft.csv",
}

client = JointClient()

for label, csv_path in CONDITIONS.items():
    df = pd.read_csv(csv_path)
    df_conv = df[df["converged"] == 1]
    if df_conv.empty:
        print(f"[Proprioceptive] {label}: no converged rows — skipping")
        continue
    q_mean = avg_q13(df_conv, "q_motor_{}_rad")
    joints = q13_to_joints_deg(q_mean)
    print_joints(f"ProprioceptiveSensing — {label} (n={len(df_conv)} rows)", joints)
    client.write_targets(joints, angle_unit="degrees", length_unit="mm")
    time.sleep(1.5)

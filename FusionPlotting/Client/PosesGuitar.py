"""
FusionPlotting poses for PassiveCompliance guitar-playing experiment.

Reads recorded CSV data and sends the mean pose during the active sweep phase
for each torsional stiffness condition (K=0.10, K=0.30 N·m/rad) to Fusion.
"""

import sys
import pathlib
import time
import pandas as pd

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from JointClient import JointClient
from _fusion_utils import avg_q15, q15_to_joints_deg, print_joints

_DATA = _HERE.parents[1] / "PassiveCompliance" / "Hand" / "outputs" / "guitar_playing_hand"

CONDITIONS = {
    "K = 0.10 N·m/rad": _DATA / "data_K0.10.csv",
    "K = 0.30 N·m/rad": _DATA / "data_K0.30.csv",
}

client = JointClient()

for label, csv_path in CONDITIONS.items():
    df = pd.read_csv(csv_path)
    df_sweep = df[df["phase"] == "sweep"]
    if df_sweep.empty:
        print(f"[Guitar] {label}: no sweep-phase rows — skipping")
        continue
    # guitar CSV uses q_0 ... q_14
    q_mean = avg_q15(df_sweep, "q_{}")
    joints = q15_to_joints_deg(q_mean)
    print_joints(f"Guitar — {label} (n={len(df_sweep)} rows)", joints)
    client.write_targets(joints, angle_unit="degrees", length_unit="mm")
    time.sleep(1.5)

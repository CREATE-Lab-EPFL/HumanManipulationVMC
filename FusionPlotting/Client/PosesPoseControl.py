"""
FusionPlotting poses for PoseControl experiment.

Reads recorded CSV data and sends the mean converged pose for PC1 (power grasp)
and PC2 (precision pinch) to Fusion. PoseControl CSVs already contain FK-computed
joint angles (not raw motor angles), so no extra FK call is needed.
"""

import sys
import pathlib
import time
import pandas as pd

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from JointClient import JointClient
from _fusion_utils import act_cols_to_joints_deg, print_joints

_DATA = _HERE.parents[1] / "PoseControl" / "outputs"

CONDITIONS = {
    "PC1 — power grasp":     _DATA / "pose1_PC1.csv",
    "PC2 — precision pinch": _DATA / "pose2_PC2.csv",
}

client = JointClient()

for label, csv_path in CONDITIONS.items():
    df = pd.read_csv(csv_path)
    df_conv = df[df["converged"] == 1]
    if df_conv.empty:
        print(f"[PoseControl] {label}: no converged rows — skipping")
        continue
    mean_row = df_conv.mean(numeric_only=True)
    joints = act_cols_to_joints_deg(mean_row)
    print_joints(f"PoseControl — {label} (n={len(df_conv)} rows)", joints)
    client.write_targets(joints, angle_unit="degrees", length_unit="mm")
    time.sleep(1.5)

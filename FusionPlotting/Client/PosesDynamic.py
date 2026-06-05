"""
FusionPlotting poses for TunableCompliance dynamic grasping experiment.

Reads recorded CSV data and sends the mean closed-grasp pose for each
compliance condition (adaptive / soft) to Fusion.
The stiff CSV contains no data and is skipped automatically.
"""

import sys
import pathlib
import time
import pandas as pd

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from JointClient import JointClient
from _fusion_utils import avg_q15, q15_to_joints_deg, print_joints

_DATA = (
    _HERE.parents[1]
    / "TunableCompliance" / "Hand" / "outputs" / "dynamic_grasp"
)

CONDITIONS = {
    "adaptive compliance": _DATA / "dynamic_grasp_adaptive.csv",
    "soft compliance":     _DATA / "dynamic_grasp_soft.csv",
    "stiff compliance":    _DATA / "dynamic_grasp_stiff.csv",
}

client = JointClient()

for label, csv_path in CONDITIONS.items():
    df = pd.read_csv(csv_path)
    df_closed = df[df["phase"] == "closed"]
    if df_closed.empty:
        print(f"[Dynamic] {label}: no closed-phase rows — skipping")
        continue
    # dynamic_grasp CSV uses q_0 ... q_14
    q_mean = avg_q15(df_closed, "q_{}")
    joints = q15_to_joints_deg(q_mean)
    print_joints(f"Dynamic grasp — {label} (n={len(df_closed)} rows)", joints)
    client.write_targets(joints, angle_unit="degrees", length_unit="mm")
    time.sleep(1.5)

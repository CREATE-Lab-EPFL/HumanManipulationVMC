"""
FusionPlotting poses for TunableCompliance in-hand manipulation experiment.

Reads the recorded CSV and sends the mean converged in-hand grasp pose to Fusion.
"""

import sys
import pathlib
import time
import pandas as pd

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from JointClient import JointClient
from _fusion_utils import avg_q15, q15_to_joints_deg, print_joints

_CSV = (
    _HERE.parents[1]
    / "TunableCompliance" / "Hand" / "outputs"
    / "inhand_manipulation" / "inhand_run.csv"
)

client = JointClient()

df = pd.read_csv(_CSV)
df_conv = df[df["converged"] == 1]
if df_conv.empty:
    print("[Inhand] no converged rows in CSV")
else:
    q_mean = avg_q15(df_conv, "q_motor_{}_rad")
    joints = q15_to_joints_deg(q_mean)
    print_joints(f"Inhand manipulation — grasped pose (n={len(df_conv)} rows)", joints)
    client.write_targets(joints, angle_unit="degrees", length_unit="mm")
    time.sleep(0.5)

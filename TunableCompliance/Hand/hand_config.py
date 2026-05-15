"""
UR5 poses for TunableCompliance/Hand experiments.
PLACEHOLDER — tune to the actual setup before running.
"""

import numpy as np

# inhand_manipulation.py — same squeezing fixture as ProprioceptiveSensing.
UR5_POSE_SQUEEZING = np.array([-0.1405, 0.6833, 0.2594, -0.1861, -0.027, -2.7656])

# dynamic_grasp.py — hand open, aligned with the bottle; UR5 slides along +X.
UR5_POSE_BOTTLE_START = np.array([-0.30, 0.60, 0.25, -0.1861, -0.027, -2.7656])

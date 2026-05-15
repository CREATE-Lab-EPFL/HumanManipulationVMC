"""
UR5 poses for all ADAPT Hand experiments.

Piano-specific poses live in PassiveCompliance/Hand/HelperPianoMIDI/piano_config.py.
All values are PLACEHOLDER — tune to the actual setup before running.
"""

import numpy as np

# ── ProprioceptiveSensing + TunableCompliance/inhand_manipulation ─────────────
# Hand horizontal, fingers pointing down, centred over the squeezing fixture.
UR5_POSE_SQUEEZING = np.array([-0.1405, 0.6833, 0.2594, -0.1861, -0.027, -2.7656])

# ── TunableCompliance/dynamic_grasp ───────────────────────────────────────────
# Hand open above the bottle; UR5 will translate along +X to approach.
UR5_POSE_BOTTLE_START = np.array([-0.30, 0.60, 0.25, -0.1861, -0.027, -2.7656])

# ── ADAPT-StiffControl/grasp_adaptation ───────────────────────────────────────
# Grasp contact pose for each known object (palm centred on the object).
UR5_POSE_GRASP_OBJ = {
    'object_1': np.array([-0.1405, 0.6833, 0.2594, -0.1861, -0.027, -2.7656]),
    'object_2': np.array([-0.1405, 0.5500, 0.2594, -0.1861, -0.027, -2.7656]),
}

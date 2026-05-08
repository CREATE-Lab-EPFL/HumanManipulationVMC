# PassiveCompliance — Experimental Area 1

Validates that VMC generates diverse, predictable stiffness profiles at the fingertip
and palm through virtual springs alone, without hardware reconfiguration.
Compliance acts as a bidirectional filter: outward (absorbs noise/impacts before
contact), inward (smoothens deformations, providing adaptability).

## Files

| File | Platform | Description |
|------|----------|-------------|
| `passive_stiffness_sweep.py` | finger | Stiffness sweep — F vs d for varying K (finger space, N·m/rad) |
| `passive_range.py` | finger | Dense biased K sweep, one run per K |
| `passive_stiffness_sweep_linear.py` | finger | Cart stiffness sweep — F vs d for varying K_cart (N/m), vertical direction |
| `passive_range_linear.py` | finger | Dense biased K_cart sweep, vertical direction |
| `directional_stiffness.py` | finger | Cart stiffness in multiple contact directions in the Y-Z plane |
| `pose_sweep.py` | finger | Pose sweep — F vs d from multiple starting Z heights |
| `piano_playing_hand.py` | hand | Three conditions: uniform stiffness sweep, k₁/k₂ per-finger assignment, and damping sweep |
| `plot_passive_stiffness_sweep.ipynb` | finger | Plot passive stiffness sweep experiment |
| `plot_passive_stiffness_sweep_linear.ipynb` | finger | Plot passive stiffness sweep linear experiment |
| `plot_directional_stiffness.ipynb` | finger | Plot directional stiffness experiment |
| `plot_pose_sweep.ipynb` | finger | Plot pose sweep experiment |
| `plot_piano_playing_hand.ipynb` | hand | Plot piano playing hand experiment |

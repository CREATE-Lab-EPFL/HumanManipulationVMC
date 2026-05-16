# ADAPT-StiffControl

End-to-end closed-loop stiffness adaptation on the 15-DOF ADAPT Hand and the
final integrative demonstration of the paper. The hand descends onto a known
object (`hard_obj` or `soft_obj`), closes at a gentle sensing stiffness, infers
the object's compliance from the steady-state mean fingertip displacement, then
ramps each fingertip stiffness to a value adapted to that displacement before
lifting and replacing the object.

All experiments in this folder use the **ADAPT Hand (15 DOF)**.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `grasp_adaptation.py` | hand | Online K_d adaptation from displacement sensing — descend, gentle close, sense δ_mean, ramp to k_applied = clip(K_SCALE·δ_mean, K_MIN, K_MAX), lift, hold, place |
| `plot_grasp_adaptation.ipynb` | hand | Plot grasp adaptation experiment |
| `hand_config.py` | hand | UR5 grasp pose per object, PC1 / home joint targets, OBJECTS list |

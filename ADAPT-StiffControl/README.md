# ADAPT-StiffControl

End-to-end closed-loop stiffness adaptation on the 15-DOF ADAPT Hand and the
final integrative demonstration of the paper. The hand descends onto a known
object (`hard_obj` or `soft_obj`), estimates its compliance with the same
two-point algorithm used in `ProprioceptiveSensing/Hand`, then matches its
own fingertip stiffness to the object's stiffness before lifting and replacing
the object.

All experiments in this folder use the **ADAPT Hand (15 DOF)**.

## Adaptation rule

Two-point compliance estimation (analytic VMC tip force, FK tip position):

  1. Close at `K_TIP_GENTLE`, settle, record per-finger `pos_gentle`, `F_gentle`.
  2. Ramp to `K_TIP_PROBE`, settle, record per-finger `pos_probe`, `F_probe`.
  3. Per finger: `C_O_f = ||pos_probe − pos_gentle|| / ||F_probe − F_gentle||`.
  4. Average across fingers → `C_O_mean`.
  5. `k_applied = clip(K_GAIN / C_O_mean, K_MIN, K_MAX)`.

`K_GAIN = 1` matches the controller stiffness to the object stiffness exactly;
raise it for a proportionally firmer grip, lower it for a softer one.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `grasp_adaptation.py` | hand | Two-point compliance sensing → k_applied = clip(K_GAIN / C_O_mean, K_MIN, K_MAX), then lift / hold / place |
| `plot_grasp_adaptation.ipynb` | hand | Plot grasp adaptation experiment |
| `hand_config.py` | hand | UR5 grasp pose per object, PC1 / home joint targets, OBJECTS list, K_TIP_GENTLE / K_TIP_PROBE / K_GAIN / K_MIN / K_MAX, joint regulation gains, timing |

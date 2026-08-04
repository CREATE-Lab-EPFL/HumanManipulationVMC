# MethodsElastic — VMC stiffness model validation

Validates the VMC stiffness composition model using elastic joints as ground-truth torsional springs. The UR5 applies a controlled pressing trajectory while the load cell records contact force, compared against model predictions.

- Two physical elastic bands are benchmarked as **soft** and **hard** ground-truth conditions.
- The stiffness mapping from virtual joint space to tip space is validated by comparing model predictions against measured indentation force.

## Files

| File | Description |
|------|-------------|
| `elastic_band_sweep.py` | UR5 pressing with motors disengaged and elastic bands (soft/hard, via `SPRING`) on all joints |
| `20mm_mimic_real_springs.py` | UR5 pressing with motors engaged and a fixed virtual stiffness matched to the elastic-band average |
| `instant_mimic_real_springs.py` | UR5 pressing with motors engaged and online stiffness updates via feed-forward inversion + `ref_descent` |
| `plot_elastic_band.ipynb` | Builds the joint-stiffness → tip-stiffness mapping from PassiveCompliance sweeps, identifies the best-matching virtual stiffness per elastic band, and compares model vs. ground-truth force-displacement curves (average and instantaneous) across all three controllers |

**Output** (`outputs/`): raw run data, identified stiffness mappings, and comparison plots for the elastic-band, fixed-stiffness, and online-stiffness conditions.

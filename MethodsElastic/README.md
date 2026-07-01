# MethodsElastic — VMC stiffness model validation

Validates the VMC stiffness composition model using elastic joints as ground-truth
torsional springs. The UR5 applies a controlled pressing trajectory and the load
cell records contact force, which is compared against model predictions.

Method summary:

- Elastic joints provide a known torsional stiffness at each finger joint.
- Two physical elastic bands are benchmarked: a **soft** and a **hard** condition.
- UR5 indentation yields force-displacement curves under repeatable contact.
- The stiffness mapping from virtual joint space to tip space is validated by
  comparing model predictions with measured force.

## Files

| File | Description |
|------|-------------|
| `elastic_band_sweep.py` | UR5 pressing with motors disengaged and elastic bands (soft or hard, set via `SPRING`) on all joints |
| `20mm_mimic_real_springs.py` | UR5 pressing with motors engaged and a fixed virtual joint stiffness matched to the soft/hard elastic-band average |
| `instant_mimic_real_springs.py` | UR5 pressing with motors engaged and online stiffness updates via feed-forward inversion and `ref_descent` feedback, per soft/hard profile |
| `plot_elastic_band.ipynb` | Analysis and plots |

## Analysis workflow (`plot_elastic_band.ipynb`)

- Build a mapping from virtual joint stiffness to tip stiffness using
  PassiveCompliance sweeps.
- Identify the virtual stiffness that best matches each elastic-band profile.
- Compare force-displacement curves between elastic-band ground truth and VMC
  predictions.
- Estimate instantaneous stiffness profiles by differentiating force data.
- Evaluate fixed-stiffness and online-stiffness mimic controllers against the
  elastic-band runs.

The analysis ties PassiveCompliance sweeps to the elastic-band data, then checks
both average and instantaneous stiffness behavior across the different control
conditions.

## Output structure

The `outputs/` folder contains raw run data, identified stiffness mappings,
comparison plots, and instantaneous stiffness profiles for the elastic-band,
fixed-stiffness mimic, and online-stiffness mimic conditions.

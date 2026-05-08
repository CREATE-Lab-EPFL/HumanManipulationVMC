# MethodsElastic — VMC stiffness model validation

Validates the VMC stiffness composition model experimentally using TPU joints as ground-truth torsional springs. The UR5 presses the fingertip and the load cell records the resulting contact force, compared against the model prediction.

## Files

| File | Description |
|------|-------------|
| `elastic_band_sweep.py` | UR5 descent + load cell — no motors, elastic bands on all joints |
| `20mm_mimic_real_springs.py` | UR5 descent + load cell — motors, K_d fixed to match the run-average stiffness of soft/hard bands |
| `instant_mimic_real_springs.py` | UR5 descent + load cell — motors, K_d updated online via feed-forward stiffness inversion + `ref_descent` force feedback |
| `plot_elastic_band.ipynb` | All analysis and plots |

## Analysis workflow (`plot_elastic_band.ipynb`)

- **K_x(K_d) from PassiveCompliance** — per-K_d stiffness sweep; piecewise-linear K_x(K_d) map.
- **Identify K_d** — match elastic-band stiffness to K_x(K_d) for soft and hard bands.
- **F–d comparison** — elastic-band ground truth vs closest PassiveCompliance curve.
- **Instantaneous stiffness profile K_x(d)** — spline-differentiated force profile; saved to `outputs/data/`.
- **F–d: TPU vs instantaneous stiffness tracking** — ground truth vs `instant_mimic_real_springs.py` (feed-forward K + ref feedback).
- **F–d and K_x(d): TPU vs VMC instant mimic** — per-phase F and stiffness comparison.

## Output structure

```
outputs/
  data/                          stiffness profiles (soft/hard_stiffness_profile.csv)
  elastic_band/                  elastic_band_sweep.py recordings
  mimic_springs/                 20mm_mimic_real_springs.py recordings
  instant_mimic_springs/         instant_mimic_real_springs.py recordings

```

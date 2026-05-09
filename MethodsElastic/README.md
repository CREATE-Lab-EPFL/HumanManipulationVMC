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
- **Identify K_d** — match elastic-band stiffness to K_x(K_d) for soft and hard bands; saved to `outputs/mimic_springs/identified_kd.json` and `Kx_identification.pdf`.
- **F–d comparison** — elastic-band ground truth vs closest PassiveCompliance curve; saved to `outputs/mimic_springs/elastic_band_Fvsd.pdf` and `kx_mimic_vs_tpu.pdf`.
- **Instantaneous stiffness profile K_x(d)** — spline-differentiated force profile; saved to `outputs/data/{soft,hard}_stiffness_profile.csv` and `kx_profile.pdf`.
- **F–d: TPU vs instantaneous stiffness tracking** — ground truth vs `instant_mimic_real_springs.py` (feed-forward K + ref feedback); saved to `outputs/instant_mimic_springs/instant_Fvsd.pdf`.
- **F–d and K_x(d): TPU vs VMC instant mimic** — per-phase F and stiffness comparison; saved to `outputs/instant_mimic_springs/instant_Fvsd_Kx_{descent,ascent}.pdf`.
- **K_d(d): commanded stiffness trajectory** — scalar VMC stiffness applied by the feed-forward controller at each depth; saved to `outputs/instant_mimic_springs/Kd_vs_d.pdf`.

## Output structure

```
outputs/
  data/
    soft_stiffness_profile.csv   spline K_x(d) profile — soft band
    hard_stiffness_profile.csv   spline K_x(d) profile — hard band
    kx_profile.pdf               K_x(d) figure
  elastic_band/
    elastic_band_{soft,hard}_run_N.csv   elastic_band_sweep.py recordings
  mimic_springs/
    identified_kd.json           identified K_d for soft/hard bands
    Kx_identification.pdf        K_x(K_d) identification figure
    elastic_band_Fvsd.pdf        F–d: elastic band vs PassiveCompliance
    kx_mimic_vs_tpu.pdf          K_x comparison
    K_X.XXXX/K_X.XXXX_run_N.csv 20mm_mimic_real_springs.py recordings
  instant_mimic_springs/
    soft/soft_run_N.csv          instant_mimic_real_springs.py — soft profile
    hard/hard_run_N.csv          instant_mimic_real_springs.py — hard profile
    instant_Fvsd.pdf             F–d: TPU vs VMC instant mimic
    instant_Fvsd_Kx_descent.pdf  F–d and K_x(d) — descent phase
    instant_Fvsd_Kx_ascent.pdf   F–d and K_x(d) — ascent phase
    Kd_vs_d.pdf                  K_d(d) commanded stiffness trajectory
```

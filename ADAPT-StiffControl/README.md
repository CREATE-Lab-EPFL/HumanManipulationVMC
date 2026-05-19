# ADAPT-StiffControl

End-to-end closed-loop stiffness adaptation on the ADAPT Hand and the
final integrative demonstration of the paper. The hand descends onto a known
object, estimates its compliance from paired contact samples using FK tip
position and analytic VMC tip force, then matches fingertip stiffness through
the controller before lifting and replacing the object.

All experiments in this folder use the **ADAPT Hand**.

## Adaptation rule (schematic)

- Close with a gentle stiffness and let the hand settle while recording FK tip
  position and model-based tip force.
- Increase stiffness for a probe, settle again, and record a follow-up position
  and force sample.
- Estimate object compliance from the ratio of position change to force change.
- Average across fingers and map compliance to an applied stiffness with
  saturation, using higher stiffness for stiffer objects and lower stiffness
  for softer objects.
- Apply the matched stiffness through lift, hold, and place.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `grasp_adaptation.py` | hand | Gentle contact, probe, estimate compliance, then lift / hold / place with matched stiffness |
| `plot_grasp_adaptation.ipynb` | hand | Plot grasp adaptation experiment |
| `hand_config.py` | hand | UR5 grasp poses, joint targets, object list, stiffness schedule, and timing |

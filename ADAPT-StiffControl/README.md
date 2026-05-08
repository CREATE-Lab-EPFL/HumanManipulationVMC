# ADAPT-StiffControl

End-to-end closed-loop stiffness control demonstration on the 15-DOF ADAPT Hand.
The hand grasps objects of different mechanical compliance (rigid, medium, soft)
and adapts its virtual stiffness online to match each object — stiffening upon
contact detection, softening when crumpling is detected. This is the final
integrative demonstration of the paper.

All experiments in this folder use the **ADAPT Hand (15 DOF)**.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `compliance_matching_demo.py` | hand | Online K_d adaptation to match object compliance: stiffening on contact, softening on crumpling |
| `plot_compliance_matching_demo.ipynb` | hand | Plot compliance matching demo experiment |

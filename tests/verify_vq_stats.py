
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from miget.io import MigetIO
from miget.core import VQModel

# Input from file MISSISSIPPI.raw
mississippi_input = """MISSISSIPPI SL FEB 21 2006
   1  752.0  37.0  38.0    0.03  0.0030
   6  50  40.0  0.005  100.0
    0.658E-02   0.132E+00   0.749E+00   0.293E+01   0.123E+02   0.317E+03
     80.8    322.5   1108.0    990.0    873.5    522.4
        1.         1.         1.         1.         1.         1.
   3588.0   1320.0   1167.0    528.5    420.8    218.1
        1.         1.         1.         1.         1.         1.
      0.0      0.0      0.0      0.0      0.0      0.0
        1.         1.         1.         1.         1.         1.
  14.40  7.40  752.0  38.7  38.7  10.04   8.55   0.61   0.00   1.00   0.00
  14.7  44.0  280.0  240.0 99000.00 0.21000.0000  28.3
   91.4   37.6  7.37
"""

print("--- Processing Data ---")
data = MigetIO.parse_legacy_file(mississippi_input)
run = data[0]
run.species = "human"
run.calculate_metrics()

print("--- Solving VQ Model (Z=40.0) ---")
model = VQModel(run, z_factor=40.0)
dist = model.solve()

print("\n--- Results ---")
print(f"Shunt:  {dist.shunt*100:.3f}% (Target: 0.4%)")
print(f"Deadspace: {dist.deadspace*100:.3f}% (Target: 59.5%)")
print(f"Mean Q: {dist.mean_q:.3f} (Target: 0.75)")
print(f"SD Q:   {dist.sd_q:.3f} (Target: 0.31)")
print(f"Mean V: {dist.mean_v:.3f} (Target: 0.83)")
print(f"SD V:   {dist.sd_v:.3f} (Target: 0.31)")

qm_match = abs(dist.mean_q - 0.75) < 0.1
qs_match = abs(dist.sd_q - 0.31) < 0.05
shunt_match = abs(dist.shunt*100 - 0.4) < 0.2
dead_match = abs(dist.deadspace*100 - 59.5) < 2.0

if qm_match and qs_match and shunt_match and dead_match:
    print("\nSUCCESS: Matches reasonable tolerance.")
else:
    print("\nFAIL: Discrepancy detected.")

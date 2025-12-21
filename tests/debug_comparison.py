import sys
import os
sys.path.append(os.getcwd())
# import pandas as pd # Not needed
from miget.io import MigetIO
from miget.core import VQModel

short_input = """rawdata2                                                    
   1  752.0  37.0  37.2    0.03  0.0030
   6  50  40.0  0.005  100.0
    6.580E-03   1.320E-01   7.490E-01   1.450E+00   1.230E+01   3.170E+02
    1.8    3.7   75.6 4978.5 7670.5 3776.4
    1.     1.     1.     1.     1.     1.
   10.6   15.1  157.1 5212.1 5815.5 1488.5
    1.     1.     1.     1.     1.     1.
    0.0    0.0    0.0    0.0    0.0    0.0
    1.     1.     1.     1.     1.     1.
   7.90  5.80  752.0  37.4  37.4   9.26   2.74   0.01   0.00   0.00   0.00
   8.9  27.0  332.0  266.0 99000.00 0.70000.0000  28.3
   79.0   41.0  7.44
"""

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

print("--- Processing SHORT_INPUT ---")
data1 = MigetIO.parse_legacy_file(short_input)
run1 = data1[0]
print(f"Run 1 Raw Data PA: {run1.pa_raw}")
run1.calculate_metrics()
print(f"Run 1 Retention: {run1.retention}")
model1 = VQModel(run1)
dist1 = model1.solve()
print(f"Run 1 Shunt: {dist1.shunt}")
print(f"Run 1 Deadspace: {dist1.deadspace}")

print("\n--- Processing MISSISSIPPI ---")
data2 = MigetIO.parse_legacy_file(mississippi_input)
run2 = data2[0]
print(f"Run 2 Raw Data PA: {run2.pa_raw}")
run2.calculate_metrics()
print(f"Run 2 Retention: {run2.retention}")
model2 = VQModel(run2)
dist2 = model2.solve()
print(f"Run 2 Shunt: {dist2.shunt}")
print(f"Run 2 Deadspace: {dist2.deadspace}")

if dist1.shunt == dist2.shunt and dist1.deadspace == dist2.deadspace:
    print("\nFAIL: Results are identical!")
else:
    print("\nSUCCESS: Results are different.")

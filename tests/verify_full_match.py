
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
try:
    from miget.io import MigetIO
    from miget.core import VQModel
except ImportError:
    # Handle optional relative import if run as script
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from miget.io import MigetIO
    from miget.core import VQModel

# Input from file MISSISSIPPI.raw (User Image context)
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

# Ensure species is human (default)
run.species = "human"

run.calculate_metrics()

# TARGETS from Screenshot
target_r = np.array([0.01349, 0.14177, 0.49095, 0.80591, 0.93869, 0.99753])
target_e = np.array([0.00327, 0.05764, 0.19309, 0.28595, 0.37634, 0.39993])

print("\n--- Verification Results ---")
print(f"{'Gas':<10} | {'Calc R':<10} | {'Target R':<10} | {'Diff':<10}")
print("-" * 50)
all_pass = True
for i, gas in enumerate(run.gas_names):
    calc_r = run.retention[i]
    targ_r = target_r[i]
    diff = abs(calc_r - targ_r)
    status = "OK" if diff < 0.0001 else "FAIL"
    if status == "FAIL": all_pass = False
    print(f"{gas:<10} | {calc_r:.5f}    | {targ_r:.5f}    | {diff:.5f} {status}")

print("-" * 50)
print(f"{'Gas':<10} | {'Calc E':<10} | {'Target E':<10} | {'Diff':<10}")
print("-" * 50)
for i, gas in enumerate(run.gas_names):
    calc_e = run.excretion[i]
    targ_e = target_e[i]
    diff = abs(calc_e - targ_e)
    status = "OK" if diff < 0.0002 else "FAIL" # E tolerance slightly looser
    if status == "FAIL": all_pass = False
    print(f"{gas:<10} | {calc_e:.5f}    | {targ_e:.5f}    | {diff:.5f} {status}")

if all_pass:
    print("\nSUCCESS: All calculations match legacy output!")
else:
    print("\nFAIL: Some values do not match.")


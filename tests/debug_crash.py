
import sys
import os
sys.path.append(os.getcwd())
from miget.io import MigetIO
from miget.core import VQModel

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

print("--- Parsing File ---")
try:
    data = MigetIO.parse_legacy_file(mississippi_input)
    print(f"Parsed {len(data)} runs.")
except Exception as e:
    print(f"CRASH during Parsing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

run1 = data[0]
print(f"Run 1 VGA: {run1.vga}, VBA: {run1.vba}")

print("--- Calculating Metrics ---")
try:
    run1.calculate_metrics()
    print("Metrics Calculated.")
    print(f"Solubilities: {run1.solubilities}")
    print(f"Retention: {run1.retention}")
except Exception as e:
    print(f"CRASH during Calculate Metrics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("--- Solving Model ---")
try:
    model = VQModel(run1)
    dist = model.solve()
    print(f"Shunt: {dist.shunt}")
except Exception as e:
    print(f"CRASH during Model Solve: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("SUCCESS: No crashes.")

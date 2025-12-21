
import sys
import os
sys.path.append(os.getcwd())

from miget.io import MigetIO
from miget.core import VQModel

content = """rawdata2                                                    
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

try:
    print("Parsing...")
    data = MigetIO.parse_legacy_file(content)
    print(f"Parsed {len(data)} runs.")
    run = data[0]
    
    print("Calculating Metrics...")
    run.calculate_metrics()
    print("Retentions:", run.retention)
    print("Excretions:", run.excretion)
    
    print("Solving VQ...")
    model = VQModel(run)
    dist = model.solve()
    print("Solved.")
    print("Shunt:", dist.shunt)

except Exception as e:
    print("CRASHED:", e)
    import traceback
    traceback.print_exc()

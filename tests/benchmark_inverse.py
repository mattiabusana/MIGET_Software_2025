
import sys
import os
import time
import numpy as np

# Ensure path is set to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from miget.physiology import BloodGasParams, inverse_blood_gas_calc

def benchmark_inverse_loop():
    params = BloodGasParams()
    # Typical Arterial Content values
    c_o2 = 19.5 
    c_co2 = 48.0
    
    n = 50
    print(f"Benchmarking {n} calls to inverse_blood_gas_calc...")
    
    start = time.time()
    for _ in range(n):
        inverse_blood_gas_calc(c_o2, c_co2, params)
    end = time.time()
    
    total_time = end - start
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Time per call: {total_time/n*1000:.2f} ms")
    
    if total_time > 1.0:
        print("PERFORMANCE BOTTLENECK CONFIRMED.")
    else:
        print("Performance looks OK?")

if __name__ == "__main__":
    benchmark_inverse_loop()
